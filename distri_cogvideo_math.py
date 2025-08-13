"""
Math operations for distributed CogVideoX implementation.
Handles rotary position embeddings (RoPE) and other mathematical operations
across multiple GPUs in distributed training/inference.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Union
import math
from diffusers.models.embeddings import apply_rotary_emb


def apply_distributed_rope_cogvideo(
    query: torch.Tensor,
    key: torch.Tensor,
    image_rotary_emb: Tuple[torch.Tensor, torch.Tensor],
    text_seq_length: int,
    distri_config,
    is_cross_attention: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings in distributed CogVideoX setting.
    
    Key principles:
    - Queries (local patches) use LOCAL position embeddings (their specific positions)
    - Keys (global patches after communication) use GLOBAL position embeddings
    - Only image parts (after text_seq_length) get RoPE applied
    - Text tokens are never rotated (matches standard CogVideoX behavior)
    
    Args:
        query: Query tensor [batch, heads, seq_len, head_dim] (local text + local image)
        key: Key tensor [batch, heads, seq_len, head_dim] (full text + global image after communication)
        image_rotary_emb: Tuple of (cos, sin) rotary embeddings for ALL image positions
        text_seq_length: Length of text sequence (to skip text tokens)
        distri_config: Distributed configuration object
        is_cross_attention: Whether the attention is cross-attention
        
    Returns:
        Tuple of (rotated_query, rotated_key)
    """
    
    # Extract image parts for RoPE application (skip text tokens)
    image_query = query[:, :, text_seq_length:]  # Local image queries
    image_key = key[:, :, text_seq_length:]      # Global image keys (after communication)
    
    if distri_config.global_world_size == 1:
        # Single GPU case - apply RoPE directly to both query and key
        rotated_image_query = apply_rotary_emb(image_query, image_rotary_emb)
        rotated_image_key = apply_rotary_emb(image_key, image_rotary_emb)
    else:
        # Multi-GPU distributed case
        cos, sin = image_rotary_emb
        
        # Calculate LOCAL position indices for queries (equal split)
        # This must match the patch distribution logic in the transformer
        if hasattr(distri_config, 'local_start_patch') and hasattr(distri_config, 'local_end_patch'):
            # Use the exact patch indices calculated in transformer forward pass
            start_patch = distri_config.local_start_patch
            end_patch = distri_config.local_end_patch
            local_positions = torch.arange(start_patch, end_patch, device=query.device, dtype=torch.long)

        else:
            # Fallback: calculate equal distribution (should match transformer logic exactly)
            local_seq_len = image_query.shape[2]
            global_seq_len = image_key.shape[2]
            
            # Match transformer's patch distribution logic exactly
            patches_per_gpu = global_seq_len // distri_config.global_world_size
            start_pos = distri_config.global_rank * patches_per_gpu
            
            # Last GPU gets remaining patches
            if distri_config.global_rank == distri_config.global_world_size - 1:
                end_pos = global_seq_len
            else:
                end_pos = start_pos + patches_per_gpu
            
            local_positions = torch.arange(start_pos, end_pos, device=query.device, dtype=torch.long)
                
            # Sanity check: the calculated range should match the actual sequence length
            if end_pos - start_pos != local_seq_len:
                if distri_config.verbose:
                    print(f"[RoPE WARNING RANK {distri_config.global_rank}] Position calc mismatch: expected {end_pos - start_pos}, got {local_seq_len}")
                # Use actual sequence length for safety
                local_positions = local_positions[:local_seq_len]
        
        local_cos = cos[local_positions]
        local_sin = sin[local_positions]
        
        # Apply RoPE
        # Queries: use LOCAL embeddings (corresponding to their specific patch positions)
        rotated_image_query = apply_rotary_emb(image_query, (local_cos, local_sin))
        
        if not is_cross_attention:
            # Keys: use GLOBAL embeddings (full sequence after communication)
            rotated_image_key = apply_rotary_emb(image_key, image_rotary_emb)
        else:
            rotated_image_key = image_key
    
    # Reconstruct full tensors by concatenating text and image parts
    # Text parts (before text_seq_length) remain unchanged (never rotated in CogVideoX)
    rotated_query = torch.cat([query[:, :, :text_seq_length], rotated_image_query], dim=2)
    rotated_key = torch.cat([key[:, :, :text_seq_length], rotated_image_key], dim=2)
    
    return rotated_query, rotated_key
