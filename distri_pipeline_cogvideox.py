# Distributed CogVideoX Pipeline
# Implements multi-GPU video generation with patch-based parallelism

import inspect
import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from transformers import T5EncoderModel, T5Tokenizer

from diffusers.models import AutoencoderKLCogVideoX
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.cogvideo.pipeline_output import CogVideoXPipelineOutput
from diffusers.schedulers import CogVideoXDPMScheduler, CogVideoXDDIMScheduler
from diffusers.utils import is_torch_xla_available, logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXTransformer3DModel

from distri_cogvideox_transformer_3d import DistriCogVideoXTransformer3DModel
from distri_cogvideo_utils import DistriCogVideoConfig, CogVideoCommManager

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)


def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    """Helper function for grid resizing"""
    tw = tgt_width
    th = tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """Retrieve timesteps from scheduler"""
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class DistriCogVideoXPipeline(DiffusionPipeline):
    """
    Distributed pipeline for text-to-video generation using CogVideoX.
    
    This pipeline implements multi-GPU collaborative video generation using
    patch-based parallelism with communication in self-attention layers.
    """
    
    _optional_components = []
    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds", 
        "negative_prompt_embeds",
    ]
    
    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKLCogVideoX,
        transformer: Union[CogVideoXTransformer3DModel, DistriCogVideoXTransformer3DModel],
        scheduler: Union[CogVideoXDDIMScheduler, CogVideoXDPMScheduler],
        distri_config: DistriCogVideoConfig,
    ):
        super().__init__()
        
        # Store distributed config FIRST
        self.distri_config = distri_config
        
        self.register_modules(
            tokenizer=tokenizer,
            text_encoder=text_encoder, 
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            distri_config=distri_config,
        )
        
        # Set VAE scale factors
        self.vae_scale_factor_spatial = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self.vae, "config") else 8
        )
        self.vae_scale_factor_temporal = (
            self.vae.config.temporal_compression_ratio if hasattr(self.vae, "config") else 4
        )
        self.vae_scaling_factor_image = self.vae.config.scaling_factor if hasattr(self.vae, "config") else 0.7
        
        # Video processor
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        
        # Communication manager for distributed processing
        self.comm_manager = None
        if distri_config.global_world_size > 1:
            self._setup_communication()
    
    def _setup_communication(self):
        """Setup communication manager for K,V sharing in attention layers"""
        
        self.comm_manager = CogVideoCommManager(self.distri_config)
        
        # Set communication manager on transformer
        if hasattr(self.transformer, 'set_comm_manager'):
            self.transformer.set_comm_manager(self.comm_manager)
    
    @torch.no_grad()
    def prepare(self, **kwargs):
        """
        Prepare the pipeline for distributed inference.
        Creates communication buffers and performs pre-run validation.
        Similar to DistriSDXL prepare logic but adapted for CogVideoX.
        """
        distri_config = self.distri_config
        
        if distri_config.global_rank == 0:
            print("Preparing distributed CogVideoX pipeline...")
        
        # Use default dimensions from config
        height = distri_config.height
        width = distri_config.width
        num_frames = distri_config.num_frames
        
        device = distri_config.device
        
        # Use minimal batch size to save memory during preparation
        batch_size = 2 if distri_config.do_classifier_free_guidance else 1
        
        # Create dummy prompt embeddings (empty prompt)
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt="",
            negative_prompt=None if not distri_config.do_classifier_free_guidance else "",
            do_classifier_free_guidance=distri_config.do_classifier_free_guidance,
            num_videos_per_prompt=1,
            device=device,
        )
        
        if distri_config.do_classifier_free_guidance:
            dummy_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        else:
            dummy_prompt_embeds = prompt_embeds
        
        # Create dummy latents
        latent_channels = self.transformer.config.in_channels
        dummy_latents = self.prepare_latents(
            batch_size, latent_channels, num_frames, height, width,
            dummy_prompt_embeds.dtype, device, None
        )
        
        # Create dummy timestep
        dummy_timestep = torch.zeros([batch_size], device=device, dtype=torch.long)
        
        # Create dummy rotary embeddings if needed
        dummy_image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(height, width, dummy_latents.size(1), device)
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )
        
        # Setup communication manager and register buffers
        comm_manager = None
        if distri_config.global_world_size > 1:
            if self.comm_manager is None:
                self._setup_communication()
            comm_manager = self.comm_manager
            
            if distri_config.global_rank == 0:
                print("Registering communication buffers...")
            
            # Reset counter and run dummy forward pass for buffer registration
            self.transformer.set_counter(0)
            try:
                with torch.no_grad():
                    _ = self.transformer(
                        hidden_states=dummy_latents,
                        encoder_hidden_states=dummy_prompt_embeds,
                        timestep=dummy_timestep,
                        image_rotary_emb=dummy_image_rotary_emb,
                        return_dict=False,
                        record=True
                    )
            except Exception as e:
                if distri_config.global_rank == 0:
                    print(f"Buffer registration completed: {e}")
            
            # Create communication buffers
            if comm_manager.numel > 0:
                comm_manager.create_buffers()
                if distri_config.global_rank == 0:
                    print(f"Communication buffers created: {comm_manager.numel / 1e6:.2f}M parameters")
        
        # Pre-run to warm up the model
        if distri_config.global_rank == 0:
            print("Running pre-run validation...")
        
        self.transformer.set_counter(0)
        try:
            with torch.no_grad():
                _ = self.transformer(
                    hidden_states=dummy_latents,
                    encoder_hidden_states=dummy_prompt_embeds,
                    timestep=dummy_timestep,
                    image_rotary_emb=dummy_image_rotary_emb,
                    return_dict=False,
                    record=True
                )
        except Exception as e:
            if distri_config.global_rank == 0:
                print(f"Pre-run validation completed: {e}")
        
        if distri_config.global_rank == 0:
            print("Pipeline preparation completed successfully")
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, distri_config: DistriCogVideoConfig, **kwargs):
        """
        Create a distributed CogVideoX pipeline from a pretrained model.
        
        Args:
            pretrained_model_name_or_path: Path to pretrained model
            distri_config: Distributed configuration
            **kwargs: Additional arguments for model loading
        """
        device = distri_config.device
        torch_dtype = kwargs.pop("torch_dtype", torch.bfloat16)
        variant = kwargs.pop("variant", None)
        
        if distri_config.global_rank == 0:
            print(f"Loading CogVideoX model from {pretrained_model_name_or_path}")
        
        # Load components
        tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
        text_encoder = T5EncoderModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=torch_dtype
        )
        vae = AutoencoderKLCogVideoX.from_pretrained(
            pretrained_model_name_or_path, subfolder="vae", torch_dtype=torch_dtype
        )
        transformer = CogVideoXTransformer3DModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="transformer", torch_dtype=torch_dtype, variant=variant
        )
        scheduler = CogVideoXDPMScheduler.from_pretrained(
            pretrained_model_name_or_path, subfolder="scheduler"
        )
        
        # Convert transformer to distributed version if multi-GPU
        if distri_config.global_rank == 0:
            print(f"Global world size: {distri_config.global_world_size}")
            if distri_config.global_world_size == 1:
                print("⚠️  Running in single GPU mode. Transformer will NOT be converted to distributed version.")
                print("   To use distributed processing, run with: torchrun --nproc_per_node=N script.py")
            
        if distri_config.global_world_size > 1:
            if distri_config.global_rank == 0:
                print("Converting transformer to distributed version...")
            try:
                transformer = DistriCogVideoXTransformer3DModel.load_from_standard_cogvideox(
                    transformer, distri_config
                )
            except Exception as e:
                if distri_config.global_rank == 0:
                    print(f"❌ Failed to convert transformer to distributed version: {e}")
                    import traceback
                    traceback.print_exc()
                raise
        else:
            if distri_config.global_rank == 0:
                print("Single GPU mode - keeping standard transformer")
        
        # Move components to device
        text_encoder = text_encoder.to(device)
        vae = vae.to(device)
        
        transformer = transformer.to(device)
            
        # Display RoPE setting from model configuration
        if distri_config.global_rank == 0:
            rope_status = "ENABLED" if transformer.config.use_rotary_positional_embeddings else "DISABLED"
            print(f"RoPE status (from model config): {rope_status}")
            
        # Create pipeline
        pipeline = cls(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae, 
            transformer=transformer,
            scheduler=scheduler,
            distri_config=distri_config,
        )
        
        if distri_config.global_rank == 0:
            print(f"Distributed CogVideoX pipeline initialized on {distri_config.global_world_size} GPU(s)")
        
        return pipeline
    
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self.distri_config.device
        dtype = dtype or self.text_encoder.dtype
        
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)
        
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )
        
        prompt_embeds = self.text_encoder(text_input_ids.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        
        # duplicate text embeddings for each generation per prompt
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)
        
        return prompt_embeds
    
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self.distri_config.device
        
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        
        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
        
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            
            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            
            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
        
        return prompt_embeds, negative_prompt_embeds
    
    def prepare_latents(
        self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None
    ):
        """
        Prepare latents for distributed video generation.
        With the new approach, generate FULL latents on all GPUs - no splitting here!
        Splitting happens inside the transformer after patchification.
        """
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # Generate FULL latents on ALL GPUs
        latent_shape = (
            batch_size,
            (num_frames - 1) // self.vae_scale_factor_temporal + 1,
            num_channels_latents,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )

        if latents is None:
            latents = randn_tensor(latent_shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # Scale by scheduler's initial noise sigma
        latents = latents * self.scheduler.init_noise_sigma
        return latents
    
    def _prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grid_height = height // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)
        grid_width = width // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)
        
        p = self.transformer.config.patch_size
        p_t = self.transformer.config.patch_size_t
        
        base_size_width = self.transformer.config.sample_width // p
        base_size_height = self.transformer.config.sample_height // p
        
        if p_t is None:
            # CogVideoX 1.0
            grid_crops_coords = get_resize_crop_region_for_grid(
                (grid_height, grid_width), base_size_width, base_size_height
            )
            freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
                embed_dim=self.transformer.config.attention_head_dim,
                crops_coords=grid_crops_coords,
                grid_size=(grid_height, grid_width),
                temporal_size=num_frames,
                device=device,
            )
        else:
            # CogVideoX 1.5
            base_num_frames = (num_frames + p_t - 1) // p_t
            
            freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
                embed_dim=self.transformer.config.attention_head_dim,
                crops_coords=None,
                grid_size=(grid_height, grid_width),
                temporal_size=base_num_frames,
                grid_type="slice",
                max_size=(base_size_height, base_size_width),
                device=device,
            )
        
        return freqs_cos, freqs_sin
    
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
        
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6.0,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
    ):
        """
        Generate video using the distributed pipeline.
        """
        device = self.distri_config.device
        
        # Use default dimensions if not provided
        height = height or self.transformer.config.sample_height * self.vae_scale_factor_spatial
        width = width or self.transformer.config.sample_width * self.vae_scale_factor_spatial
        num_frames = num_frames or self.transformer.config.sample_frames
        
        # Prepare prompt
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        
        do_classifier_free_guidance = guidance_scale > 1.0
        
        # Encode prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        
        # Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        
        latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        
        # For CogVideoX 1.5, pad latent frames to be divisible by patch_size_t
        patch_size_t = self.transformer.config.patch_size_t
        additional_frames = 0
        if patch_size_t is not None and latent_frames % patch_size_t != 0:
            additional_frames = patch_size_t - latent_frames % patch_size_t
            num_frames += additional_frames * self.vae_scale_factor_temporal

        # === Prepare distributed latents ===
        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        
        # === Calculate RoPE using GLOBAL latent frames (like original pipeline) ===
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )

        # === Prepare communication buffers if not already done ===
        if (self.distri_config.global_world_size > 1 and 
            hasattr(self, 'comm_manager') and 
            self.comm_manager is not None and 
            self.comm_manager.buffer_list is None):
            if self.distri_config.global_rank == 0:
                print("Communication buffers not prepared, running preparation...")
            self.prepare()
        
        # Now prepare prompt_embeds for actual inference
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        
        # Reset counter for inference
        if hasattr(self.transformer, 'set_counter'):
            self.transformer.set_counter(0)
        
        # Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        
        # Progress bar setup
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            old_pred_original_sample = None
            for i, t in enumerate(timesteps):
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                timestep = t.expand(latent_model_input.shape[0])
                
                # Predict noise
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred.float()
                
                # Perform guidance
                if use_dynamic_cfg:
                    guidance_scale = 1 + guidance_scale * (
                        (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                    )
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # Compute previous sample
                if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                else:
                    latents, old_pred_original_sample = self.scheduler.step(
                        noise_pred,
                        old_pred_original_sample,
                        t,
                        timesteps[i - 1] if i > 0 else None,
                        latents,
                        **extra_step_kwargs,
                        return_dict=False,
                    )
                latents = latents.to(prompt_embeds.dtype)
                
                # Callback
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                
                if XLA_AVAILABLE:
                    xm.mark_step()
        
        # Process results
        if self.distri_config.global_rank == 0:
            if self.distri_config.verbose:
                print(f"Processing final latents - shape: {latents.shape}")
                
            if output_type != "latent":
                # Remove padding frames if they were added
                if additional_frames > 0:
                    latents = latents[:, additional_frames:]
                
                # Decode latents - exactly like original pipeline
                latents = latents.permute(0, 2, 1, 3, 4)  # (B, C, F, H, W)
                latents = 1 / self.vae_scaling_factor_image * latents
                frames = self.vae.decode(latents).sample
                video = self.video_processor.postprocess_video(video=frames, output_type=output_type)
            else:
                video = latents
        else:
            # Non-main GPUs return empty result
            video = [] if output_type != "latent" else torch.empty(0)
        
        # Offload models
        self.maybe_free_model_hooks()
        
        if not return_dict:
            return (video,)
        
        return CogVideoXPipelineOutput(frames=video) 