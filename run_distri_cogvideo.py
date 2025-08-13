#!/usr/bin/env python3
"""
Distributed CogVideoX inference script.

This script implements multi-GPU collaborative video generation using CogVideoX
with patch-based parallelism and communication in self-attention layers.

Usage:
    # Single GPU
    python run_distri_cogvideo.py --prompt "A cat playing with a ball" --output_path "output.mp4"
    
    # Multi-GPU
    torchrun --nproc_per_node=2 run_distri_cogvideo.py --prompt "A cat playing with a ball" --output_path "output.mp4"
"""

import os
import torch
import torch.distributed as dist
import argparse
import time
import logging
from pathlib import Path

from diffusers.utils import export_to_video
from diffusers.schedulers import CogVideoXDPMScheduler, CogVideoXDDIMScheduler

from distri_pipeline_cogvideox import DistriCogVideoXPipeline
from distri_cogvideo_utils import DistriCogVideoConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_distributed():
    """Initialize the distributed environment"""
    if not dist.is_initialized():
        try:
            dist.init_process_group("nccl")
            logger.info(f"✅ Distributed initialized: {dist.get_world_size()} GPUs")
            return True
        except Exception as e:
            logger.warning(f"❌ Failed to initialize process group: {e}")
            logger.warning("   This is normal when running single GPU. Use torchrun for multi-GPU.")
            return False
    else:
        logger.info(f"✅ Distributed already initialized: {dist.get_world_size()} GPUs")
        return True


def cleanup_distributed():
    """Clean up the distributed environment"""
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Distributed CogVideoX Video Generation")
    
    # Model configuration
    parser.add_argument("--model_path", type=str, default="THUDM/CogVideoX-2b",
                       help="Path to CogVideoX model")
    parser.add_argument("--prompt", type=str, required=True,
                       help="Text prompt for video generation")
    parser.add_argument("--negative_prompt", type=str, default=None,
                       help="Negative text prompt")
    
    # Video parameters
    parser.add_argument("--height", type=int, default=480,
                       help="Video height (should be multiple of 16)")
    parser.add_argument("--width", type=int, default=720,
                       help="Video width (should be multiple of 16)")
    parser.add_argument("--num_frames", type=int, default=49,
                       help="Number of video frames to generate")
    parser.add_argument("--fps", type=int, default=8,
                       help="Frames per second for output video")
    
    # Sampling parameters
    parser.add_argument("--num_inference_steps", type=int, default=50,
                       help="Number of denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=6.0,
                       help="Guidance scale for sampling")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible generation")
    parser.add_argument("--use_dynamic_cfg", action="store_true",
                       help="Use dynamic classifier-free guidance")
    parser.add_argument("--max_sequence_length", type=int, default=226,
                       help="Maximum sequence length for text encoder")
    parser.add_argument("--scheduler", type=str, default="DPM", choices=["DPM", "DDIM"],
                       help="Scheduler type to use")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"],
                       help="Data type for computation")
    
    # Distributed configuration - Equal patch splitting is always used
    parser.add_argument("--mode", type=str, default="optimized",
                       choices=["full_sync", "optimized", "no_sync"],
                       help="Communication mode")
    parser.add_argument("--warmup_steps", type=int, default=4,
                       help="Number of initial steps with full synchronization")
    parser.add_argument("--comm_checkpoint", type=int, default=60,
                       help="Communication checkpoint frequency")
    
    # Output
    parser.add_argument("--output_path", type=str, default="output_cogvideo.mp4",
                       help="Path to save generated video")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.height % 16 != 0 or args.width % 16 != 0:
        raise ValueError("Height and width must be multiples of 16")
    
    # Use original model path directly
    model_path = args.model_path
    
    # Initialize distributed training if available
    is_distributed = setup_distributed()
    
    if is_distributed:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set model dtype
    torch_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    
    if rank == 0:
        logger.info(f"Starting distributed CogVideoX inference on {world_size} GPU(s)")
        logger.info(f"Prompt: {args.prompt}")
        logger.info(f"Video size: {args.num_frames}x{args.height}x{args.width}")
        logger.info(f"Model: {model_path}")
        logger.info(f"Data type: {torch_dtype}")
    
    # Create distributed configuration
    distri_config = DistriCogVideoConfig(
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        warmup_steps=args.warmup_steps,
        comm_checkpoint=args.comm_checkpoint,
        mode=args.mode,
        verbose=args.verbose and rank == 0,
        do_classifier_free_guidance=(args.guidance_scale > 1.0),
    )
    
    if distri_config.global_rank == 0:
        print("Loading distributed CogVideoX pipeline...")
    
    # Create distributed pipeline - this handles model loading and distributed setup automatically
    pipeline = DistriCogVideoXPipeline.from_pretrained(
        pretrained_model_name_or_path=model_path,
        distri_config=distri_config,
        torch_dtype=torch_dtype,
        variant="fp16" if torch_dtype == torch.float16 else None,
    )
    
    # Set scheduler (similar to original cli_demo.py)
    if args.scheduler == "DDIM":
        pipeline.scheduler = CogVideoXDDIMScheduler.from_config(
            pipeline.scheduler.config, timestep_spacing="trailing"
        )
    else:  # DPM
        pipeline.scheduler = CogVideoXDPMScheduler.from_config(
            pipeline.scheduler.config, timestep_spacing="trailing"
        )
    
    # Enable optimizations (similar to original cli_demo.py)
    # Note: In distributed setting, we may not want CPU offload as models are distributed
    if world_size == 1:
        # Only enable CPU offload for single GPU
        pipeline.enable_sequential_cpu_offload()
        pipeline.vae.enable_slicing()
        pipeline.vae.enable_tiling()
    
    if distri_config.global_rank == 0:
        print("Pipeline loaded and configured successfully")
        print("Starting video generation...")
        start_time = time.time()
    
    # Generate video using distributed pipeline
    with torch.no_grad():
        generator = torch.Generator(device=device).manual_seed(args.seed)
        
        video_frames = pipeline(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            use_dynamic_cfg=args.use_dynamic_cfg,
            generator=generator,
            output_type="pil",
            max_sequence_length=args.max_sequence_length,
        ).frames
    
    if distri_config.global_rank == 0:
        inference_time = time.time() - start_time
        print(f"Inference completed in {inference_time:.2f} seconds")
        
        # Save the video (only main GPU has the result)
        if video_frames and len(video_frames) > 0:
            export_to_video(video_frames[0], args.output_path, fps=args.fps)
            
            total_time = time.time() - start_time
            print(f"Video saved to {args.output_path}")
            print(f"Total generation time: {total_time:.2f} seconds")
            print(f"Time per step: {total_time/args.num_inference_steps:.3f} seconds")
        else:
            print("No video frames generated")
    
    # Cleanup  
    if is_distributed:
        cleanup_distributed()


if __name__ == "__main__":
    main() 