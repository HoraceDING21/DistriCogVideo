# Distributed CogVideoX Utilities
# Configuration and helper functions for multi-GPU video generation

import os
import torch
import torch.distributed as dist
from typing import Dict, List, Optional, Tuple, Union
import math
from diffusers.utils.torch_utils import randn_tensor


class DistriCogVideoConfig:
    """
    Enhanced configuration for distributed CogVideoX processing.
    Supports different parallelism strategies and communication patterns.
    """
    def __init__(
        self,
        height: int = 720,
        width: int = 480,
        num_frames: int = 49,
        warmup_steps: int = 4,
        comm_checkpoint: int = 60,
        mode: str = "optimized",
        verbose: bool = False,
        do_classifier_free_guidance: bool = True,
    ):
        # Initialize distributed environment
        if dist.is_initialized():
            self.global_rank = dist.get_rank()
            self.global_world_size = dist.get_world_size()
            if verbose:
                print(f"✅ Distributed already initialized: rank {self.global_rank}/{self.global_world_size}")
        else:
            if verbose:
                print("Distributed not initialized, attempting to initialize...")
            try:
                dist.init_process_group("nccl")
                self.global_rank = dist.get_rank()
                self.global_world_size = dist.get_world_size()
                if verbose:
                    print(f"✅ Successfully initialized distributed: rank {self.global_rank}/{self.global_world_size}")
            except Exception as e:
                if verbose:
                    print(f"❌ Failed to initialize process group: {e}, falling back to single GPU mode")
                self.global_rank = 0
                self.global_world_size = 1
        
        # Basic configuration
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.warmup_steps = warmup_steps
        self.comm_checkpoint = comm_checkpoint
        self.mode = mode
        self.verbose = verbose
        self.do_classifier_free_guidance = do_classifier_free_guidance
        
        # Device configuration
        self.device = torch.device(f"cuda:{self.global_rank}") if torch.cuda.is_available() else torch.device("cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(self.device)
        
        # Process group configuration
        self.n_device_per_batch = self.global_world_size
        
        # Create process groups
        if self.global_world_size > 1:
            self.all_processes_group = dist.group.WORLD
        else:
            self.all_processes_group = None
        
        # Video-specific calculations
        self.patch_size = 2  # CogVideoX uses 2x2 spatial patches
        self.patch_size_t = 4  # Temporal compression ratio
        
        # Patch division info (set during transformer forward)
        self.local_start_patch = None
        self.local_end_patch = None
        self.total_image_patches = None
        
    
    def split_idx(self) -> int:
        """Return index within the process group for the current process"""
        return self.global_rank


class CogVideoCommManager:
    """
    Communication manager for coordinating distributed tensor operations in CogVideoX.
    Handles efficient buffer management and asynchronous communication.
    """
    def __init__(self, config: DistriCogVideoConfig):
        self.config = config
        self.torch_dtype = None
        self.numel = 0
        self.numel_dict = {}
        
        # Tracking registered tensors
        self.starts = []
        self.ends = []
        self.shapes = []
        self.buffer_list = None
        
        # Communication state
        self.idx_queue = []
        self.handles = None
    
    def register_tensor(self, shape, torch_dtype, layer_type=None):
        """Register a tensor for communication and buffer allocation"""
        if self.torch_dtype is None:
            self.torch_dtype = torch_dtype
        else:
            assert self.torch_dtype == torch_dtype, "All tensors must have the same dtype"
        
        self.starts.append(self.numel)
        numel = 1
        for dim in shape:
            numel *= dim
        self.numel += numel
        
        # Track tensor types for debugging
        if layer_type is not None:
            if layer_type not in self.numel_dict:
                self.numel_dict[layer_type] = 0
            self.numel_dict[layer_type] += numel
        
        self.ends.append(self.numel)
        self.shapes.append(shape)
        return len(self.starts) - 1
    
    def create_buffers(self):
        """Create communication buffers for all processes"""
        config = self.config
        if config.global_rank == 0 and config.verbose:
            print(
                f"Creating buffer with {self.numel / 1e6:.3f}M parameters for {len(self.starts)} tensors on each device."
            )
            for layer_type, numel in self.numel_dict.items():
                print(f"  {layer_type}: {numel / 1e6:.3f}M parameters")
        
        self.buffer_list = [
            torch.empty(self.numel, dtype=self.torch_dtype, device=config.device)
            for _ in range(config.global_world_size)
        ]
        self.handles = [None for _ in range(len(self.starts))]
    
    def get_buffer_list(self, idx):
        """Get buffer views for a registered tensor"""
        if self.buffer_list is None:
            return None
        buffer_list = [t[self.starts[idx]:self.ends[idx]].view(self.shapes[idx]) for t in self.buffer_list]
        return buffer_list
    
    def communicate(self):
        """Execute the actual communication operation"""   
        config = self.config
        start = self.starts[self.idx_queue[0]]
        end = self.ends[self.idx_queue[-1]]
        
        # Get local tensor
        tensor = self.buffer_list[config.global_rank][start:end]
        
        # Get buffer list for all processes
        buffer_list = [t[start:end] for t in self.buffer_list]
        
        # Perform asynchronous all-gather
        handle = dist.all_gather(
            buffer_list, 
            tensor, 
            group=config.all_processes_group, 
            async_op=True
        )
        
        # Set handle for all tensors in the queue
        for i in self.idx_queue:
            self.handles[i] = handle
        
        self.idx_queue = []
    
    def enqueue(self, idx, tensor):
        """Add a tensor to the communication queue"""
        config = self.config
        
        # Handle first element in a new queue
        if idx == 0 and len(self.idx_queue) > 0:
            self.communicate()
            
        # Ensure tensors are added in order
        assert len(self.idx_queue) == 0 or self.idx_queue[-1] == idx - 1
        
        self.idx_queue.append(idx)
        self.buffer_list[config.global_rank][self.starts[idx]:self.ends[idx]].copy_(tensor.flatten())
        
        # Check if we reached checkpoint for communication
        if len(self.idx_queue) == config.comm_checkpoint:
            self.communicate()
    
    def clear(self):
        """Clear all pending operations"""
        if len(self.idx_queue) > 0:
            self.communicate()
            
        if self.handles is not None:
            for i in range(len(self.handles)):
                if self.handles[i] is not None:
                    self.handles[i].wait()
                    self.handles[i] = None 