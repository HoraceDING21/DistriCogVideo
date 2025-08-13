# Distributed CogVideoX Transformer 3D Model
# Implements patch-based parallelism with communication in self-attention layers

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, Tuple, Dict, Any, Union

from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXTransformer3DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import logging

from distri_cogvideo_utils import DistriCogVideoConfig
from distri_cogvideo_math import apply_distributed_rope_cogvideo
from einops import rearrange

logger = logging.get_logger(__name__)


class BaseModule(nn.Module):
    """
    Base module for distributed components, similar to FLUX implementation
    """
    def __init__(self, module, distri_config):
        super().__init__()
        self.module = module
        self.distri_config = distri_config
        self.comm_manager = None
        self.counter = 0
        self.idx = None
        self.buffer_list = None
    
    def set_comm_manager(self, comm_manager):
        self.comm_manager = comm_manager
    
    def set_counter(self, counter=0):
        self.counter = counter


class DistriCogVideoXAttnProcessor2_0(BaseModule):
    """
    Distributed attention processor for CogVideoX using communication patterns from FLUX.
    Implements proper KV communication and buffer management.
    """
    
    def __init__(self, attention_module, distri_config: DistriCogVideoConfig):
        super(DistriCogVideoXAttnProcessor2_0, self).__init__(attention_module, distri_config)
        
        # Store original attention module
        self.attention = attention_module
        self.distri_config = distri_config
        
        # Get dtype and device from original module
        original_dtype = next(attention_module.parameters()).dtype
        original_device = next(attention_module.parameters()).device
        
        # Create separate Q and KV projections for distributed communication
        hidden_size = attention_module.to_q.in_features
        self.to_q = nn.Linear(
            hidden_size, 
            attention_module.inner_dim, 
            bias=attention_module.to_q.bias is not None,
            dtype=original_dtype,
            device=original_device
        )
        self.to_kv = nn.Linear(
            hidden_size, 
            attention_module.inner_dim * 2,  # For both K and V
            bias=attention_module.to_k.bias is not None,
            dtype=original_dtype,
            device=original_device
        )
        
        # Copy weights from original attention module
        if hasattr(attention_module, 'to_qkv'):
            # Fused QKV case
            qkv_weight = attention_module.to_qkv.weight
            qkv_bias = attention_module.to_qkv.bias
            
            # Split QKV weights
            q_weight, k_weight, v_weight = torch.chunk(qkv_weight, 3, dim=0)
            self.to_q.weight.data.copy_(q_weight)
            self.to_kv.weight.data.copy_(torch.cat([k_weight, v_weight], dim=0))
            
            if qkv_bias is not None:
                q_bias, k_bias, v_bias = torch.chunk(qkv_bias, 3, dim=0)
                self.to_q.bias.data.copy_(q_bias)
                self.to_kv.bias.data.copy_(torch.cat([k_bias, v_bias], dim=0))
        else:
            # Separate Q, K, V case
            self.to_q.weight.data.copy_(attention_module.to_q.weight)
            self.to_kv.weight.data.copy_(torch.cat([
                attention_module.to_k.weight, 
                attention_module.to_v.weight
            ], dim=0))
            
            if attention_module.to_q.bias is not None:
                self.to_q.bias.data.copy_(attention_module.to_q.bias)
                self.to_kv.bias.data.copy_(torch.cat([
                    attention_module.to_k.bias,
                    attention_module.to_v.bias
                ], dim=0))
        
        # Store other components
        self.to_out = attention_module.to_out
        self.heads = attention_module.heads
        self.inner_dim = attention_module.inner_dim
        self.scale = attention_module.scale
        self.head_dim = self.inner_dim // self.heads
        
        # Store cross attention flag (important for RoPE application)
        self.is_cross_attention = getattr(attention_module, 'is_cross_attention', False)
        
        # For norm in CogVideoX (if exists)
        if hasattr(attention_module, 'norm_q'):
            self.norm_q = attention_module.norm_q
            self.norm_k = attention_module.norm_k
        else:
            self.norm_q = None
            self.norm_k = None
    
    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        
        distri_config = self.distri_config
        
        # Wait for previous communication to complete (only the processor needs this)
        if self.comm_manager is not None and self.comm_manager.handles is not None and self.idx is not None:
            if self.comm_manager.handles[self.idx] is not None:
                self.comm_manager.handles[self.idx].wait()
                self.comm_manager.handles[self.idx] = None
        
        # Get text sequence length - critical for proper RoPE application
        text_seq_length = encoder_hidden_states.size(1)
        
        b, image_sequence_length, _ = hidden_states.shape
        
        if distri_config.global_world_size > 1 and self.buffer_list is None:
            # Register tensor for img_kv communication only (not txt_kv)
            if self.comm_manager.buffer_list is None:
                self.idx = self.comm_manager.register_tensor(
                    shape=(b, image_sequence_length, self.to_kv.out_features),
                    torch_dtype=hidden_states.dtype,
                    layer_type="cogvideox_attn"
                )
            else:
                self.buffer_list = self.comm_manager.get_buffer_list(self.idx)
        
        # Concatenate text and image sequences like in standard CogVideoXAttnProcessor2_0
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        
        batch_size, sequence_length, _ = hidden_states.shape
        
        # Compute query locally (no communication needed)
        query = self.to_q(hidden_states)
        query = query.view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        
        # Compute key-value for the complete sequence (like standard CogVideoXAttnProcessor2_0)
        kv = self.to_kv(hidden_states)
        
        # Split into text and image parts for separate handling
        text_kv = kv[:, :text_seq_length, :]  # Text K,V - no communication needed
        image_kv = kv[:, text_seq_length:, :]  # Image K,V - needs communication for global context
        
        # OPTIMIZATION: Ensure image_kv is contiguous for communication if needed
        if distri_config.global_world_size > 1 and not image_kv.is_contiguous():
            image_kv = image_kv.contiguous()
        
        # Handle distributed key-value processing ONLY for image part
        # SIMPLE: Each GPU has local image patches, concatenate to get global image_kv
        if distri_config.global_world_size == 1:
            # Single GPU - no communication needed
            full_image_kv = image_kv
        else:
            # Communication: simple concatenation of local image patches
            if self.buffer_list is None:  # buffer not created yet (pre-run phase)
                gathered_image_kv = [image_kv for _ in range(distri_config.global_world_size)]
            elif distri_config.mode == "full_sync" or self.counter <= distri_config.warmup_steps:
                # Synchronous communication
                dist.all_gather(self.buffer_list, image_kv, group=distri_config.all_processes_group, async_op=False)
                gathered_image_kv = self.buffer_list
            else:
                # Optimized mode - use cached buffers with asynchronous updates
                gathered_image_kv = [buffer for buffer in self.buffer_list]
                gathered_image_kv[distri_config.global_rank] = image_kv
                if distri_config.mode != "no_sync":
                    # Enqueue for asynchronous communication
                    self.comm_manager.enqueue(self.idx, image_kv)
            
            # SIMPLE CONCATENATION: Equal split means simple concatenation preserves order
            full_image_kv = torch.cat(gathered_image_kv, dim=1)

        # Reconstruct full K,V by concatenating text and image parts
        # Now we have: local Q (from this GPU's slice) + global K,V (from all GPUs)
        full_kv = torch.cat([text_kv, full_image_kv], dim=1)
        
        # Split into key and value
        key, value = torch.chunk(full_kv, 2, dim=-1)
        
        # Reshape key and value
        key = key.view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        
        # Apply normalization if available
        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)
        
        # Apply 3D rotary position embeddings ONLY to the image part (after text_seq_length)
        # This matches the behavior of standard CogVideoXAttnProcessor2_0
        if image_rotary_emb is not None:
            # Apply distributed RoPE using the math function
            query, key = apply_distributed_rope_cogvideo(
                query=query,
                key=key,
                image_rotary_emb=image_rotary_emb,
                text_seq_length=text_seq_length,
                distri_config=distri_config,
                is_cross_attention=self.is_cross_attention
            )
        
        # Compute attention with local Q and global K,V
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * self.head_dim)
        
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        
        # Split back to encoder_hidden_states and hidden_states like in standard processor
        # Return only the local part that corresponds to this GPU's slice
        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        
        self.counter += 1
        return hidden_states, encoder_hidden_states


class DistriCogVideoXBlock(nn.Module):
    """
    Distributed implementation of CogVideoXBlock.
    The block itself doesn't need communication management - only the attention processor does.
    """
    
    def __init__(self, module, distri_config: DistriCogVideoConfig):
        super(DistriCogVideoXBlock, self).__init__()
        
        self.distri_config = distri_config
        
        # Store original module components
        self.norm1 = module.norm1
        self.attn1 = module.attn1
        self.norm2 = module.norm2
        self.ff = module.ff
        
        # Replace attention with distributed version
        self.distributed_attn = DistriCogVideoXAttnProcessor2_0(
            module.attn1, distri_config
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # Apply normalization and modulation
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, temb
        )
        
        # Apply distributed attention (communication is handled inside the processor)
        # The distributed processor returns both hidden_states and encoder_hidden_states
        attn_hidden_states, attn_encoder_hidden_states = self.distributed_attn(
            self.attn1,
            norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **(attention_kwargs or {}),
        )
        
        # Apply attention gates
        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states
        
        # Apply second normalization and modulation
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb
        )
        
        # Feed-forward
        text_seq_length = encoder_hidden_states.size(1)
        norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        ff_output = self.ff(norm_hidden_states)
        
        # Split FF output back to text and image parts
        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]
        
        return hidden_states, encoder_hidden_states
    
    def set_comm_manager(self, comm_manager):
        """Set communication manager only for the attention processor"""
        self.distributed_attn.set_comm_manager(comm_manager)
    
    def set_counter(self, counter=0):
        """Set counter only for the attention processor"""
        self.distributed_attn.set_counter(counter)


class DistriCogVideoXTransformer3DModel(CogVideoXTransformer3DModel):
    """
    Distributed version of CogVideoXTransformer3DModel with communication coordination.
    Inherits directly from CogVideoXTransformer3DModel to preserve all PyTorch functionality.
    """
    
    def __init__(self, config, distri_config: DistriCogVideoConfig):
        # Initialize as a standard CogVideoX transformer with the same config
        super().__init__(**config)
        
        # Store distributed config and communication state
        self.distri_config = distri_config
        self.comm_manager = None
        self.counter = 0
        self.buffer_list = None  # Initialize buffer list for distributed gathering

    @classmethod
    def load_from_standard_cogvideox(cls, model: CogVideoXTransformer3DModel, distri_config: DistriCogVideoConfig):
        """
        Convert a standard CogVideoXTransformer3DModel to a distributed one.
        Following FLUX pattern: inherit from original class and replace specific modules.
        
        Args:
            model: Standard CogVideoX transformer model
            distri_config: Distributed configuration
            
        Returns:
            DistriCogVideoXTransformer3DModel with distributed components
        """
        # Get model dtype and device to preserve
        model_dtype = next(model.parameters()).dtype
        model_device = next(model.parameters()).device
        
        # Create a new DistriCogVideoXTransformer3DModel with the same config
        distri_model = cls(model.config, distri_config)
        
        # Copy all parameters from original model (preserves weights)
        distri_model.load_state_dict(model.state_dict())
        
        # CRITICAL: Preserve config modifications from original model
        # Especially use_rotary_positional_embeddings setting
        distri_model.config.use_rotary_positional_embeddings = model.config.use_rotary_positional_embeddings
        
        # Ensure distributed-specific attributes are properly initialized after load_state_dict
        distri_model.buffer_list = None
        distri_model.comm_manager = None
        distri_model.counter = 0
        
        # Ensure the distributed model has the same dtype as the original
        distri_model = distri_model.to(dtype=model_dtype, device=model_device)
        
        # Replace transformer blocks with distributed versions
        if distri_config.global_world_size > 1:
            # Convert transformer blocks to distributed versions
            new_blocks = nn.ModuleList()
            for i, block in enumerate(distri_model.transformer_blocks):
                distri_block = DistriCogVideoXBlock(block, distri_config)
                # Ensure distributed block has correct dtype
                distri_block = distri_block.to(dtype=model_dtype, device=model_device)
                new_blocks.append(distri_block)
            distri_model.transformer_blocks = new_blocks
            
            if distri_config.verbose and distri_config.global_rank == 0:
                print(f"Converted {len(distri_model.transformer_blocks)} CogVideoXBlocks to distributed versions")
        
        if distri_config.verbose and distri_config.global_rank == 0:
            print(f"Successfully converted CogVideoXTransformer3DModel to distributed version")
            print(f"Using {distri_config.global_world_size} GPUs with equal patch splitting")
        
        return distri_model
    
    def set_comm_manager(self, comm_manager):
        """Set communication manager for the model and propagate to attention processors."""
        self.comm_manager = comm_manager
        
        # Propagate only to distributed blocks
        for block in self.transformer_blocks:
            if isinstance(block, DistriCogVideoXBlock):
                block.set_comm_manager(comm_manager)
    
    def set_counter(self, counter=0):
        """Reset or set step counter for this model and propagate to attention processors"""
        self.counter = counter
        
        # Propagate only to distributed blocks (which will pass it to their attention processors)
        for block in self.transformer_blocks:
            if isinstance(block, DistriCogVideoXBlock):
                block.set_counter(counter)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        ofs: Optional[Union[int, float, torch.LongTensor]] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        record: bool = False,  # For buffer registration compatibility
    ):
        """
        Distributed forward pass with hidden_states splitting.
        Split hidden_states (image patches) across GPUs, keep encoder_hidden_states (text) full.
        """
        batch_size, num_frames, channels, height, width = hidden_states.shape
        
        # 1. Time embedding (from parent class logic)
        timesteps = timestep
        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        if self.ofs_embedding is not None:
            ofs_emb = self.ofs_proj(ofs)
            ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
            ofs_emb = self.ofs_embedding(ofs_emb)
            emb = emb + ofs_emb

        # 2. Patch embedding (creates patchified sequence)
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        image_hidden_states = hidden_states[:, text_seq_length:]  # Image patches only

        # 3. SPLIT: Divide image patches across GPUs (equal split)
        if self.distri_config.global_world_size > 1:
            total_image_patches = image_hidden_states.shape[1]
            patches_per_gpu = total_image_patches // self.distri_config.global_world_size
            
            start_patch = self.distri_config.global_rank * patches_per_gpu
            end_patch = start_patch + patches_per_gpu
            
            # Last GPU takes remaining patches
            if self.distri_config.global_rank == self.distri_config.global_world_size - 1:
                end_patch = total_image_patches
            
            # Split image patches for this GPU
            local_image_hidden_states = image_hidden_states[:, start_patch:end_patch, :]
            
            # Store info for attention processor
            self.distri_config.local_start_patch = start_patch
            self.distri_config.local_end_patch = end_patch
            self.distri_config.total_image_patches = total_image_patches
            
        else:
            local_image_hidden_states = image_hidden_states

        # 4. Transformer blocks
        # encoder_hidden_states: FULL text embeddings (same on all GPUs)
        # local_image_hidden_states: LOCAL image patches (different per GPU)
        for i, block in enumerate(self.transformer_blocks):
            local_image_hidden_states, encoder_hidden_states = block(
                hidden_states=local_image_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=emb,
                image_rotary_emb=image_rotary_emb,
                attention_kwargs=attention_kwargs,
            )

        # 5. Final norm
        local_image_hidden_states = self.norm_final(local_image_hidden_states)

        # 6. GATHER: Collect local hidden states from all GPUs before unpatchify
        if self.distri_config.global_world_size > 1:
            # Gather all local image hidden states
            if self.buffer_list is None:
                self.buffer_list = [torch.empty_like(local_image_hidden_states) for _ in range(self.distri_config.global_world_size)]
            
            dist.all_gather(self.buffer_list, local_image_hidden_states, group=self.distri_config.all_processes_group, async_op=False)
            full_image_hidden_states = torch.cat(self.buffer_list, dim=1)
        else:
            full_image_hidden_states = local_image_hidden_states

        # 7. Final block and unpatchify (needs full sequence)
        full_image_hidden_states = self.norm_out(full_image_hidden_states, temb=emb)
        full_image_hidden_states = self.proj_out(full_image_hidden_states)

        # 8. Unpatchify
        p = self.config.patch_size
        p_t = self.config.patch_size_t

        if p_t is None:
            output = full_image_hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
            output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
        else:
            output = full_image_hidden_states.reshape(
                batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
            )
            output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

        self.counter += 1

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
    