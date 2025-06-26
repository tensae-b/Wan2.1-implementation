# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial
import copy
from collections import OrderedDict

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn as nn
import torchvision.transforms.functional as TF
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .distributed.fsdp import shard_model
from .modules.clip import CLIPModel
from .modules.model import WanModel,sinusoidal_embedding_1d,MLPProj,rope_params,Head
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class ModelParallelWrapper(nn.Module):
    def __init__(self, model, devices, vae=None, strategy='auto'):
        super().__init__()
        self.devices = devices
        self.num_devices = len(devices)
        self.vae = vae
        self.strategy = strategy
        
        # Store model config and attributes
        self.config = model.config if hasattr(model, 'config') else None
        
        # IMPORTANT: Store model components as attributes, don't create in forward
        self.dim = getattr(model, 'dim', 5120)
        self.patch_size = getattr(model, 'patch_size', (1, 2, 2))
        self.freq_dim = getattr(model, 'freq_dim', 256)
        self.text_dim = getattr(model, 'text_dim', 4096)
        
        # Initialize embeddings as model attributes
        device0 = devices[0]
        in_dim = 36
        
        # Create embeddings once during initialization
        self.patch_embedding = nn.Conv3d(in_dim, self.dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.patch_embedding = self.patch_embedding.to(dtype=torch.bfloat16, device=device0)
        
        self.time_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, self.dim), nn.SiLU(), nn.Linear(self.dim, self.dim)
        ).to(device0)
        
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(self.dim, self.dim * 6)
        ).to(device0)
        
        self.text_embedding = nn.Sequential(
            nn.Linear(self.text_dim, self.dim), nn.GELU(approximate='tanh'), nn.Linear(self.dim, self.dim)
        ).to(device0)
        
        self.img_emb = MLPProj(1280, self.dim).to(device0)
        
        # Create head
        out_dim = 16
        eps = 1e-06
        self.head = Head(self.dim, out_dim, self.patch_size, eps).to(device0)
        
        # Create RoPE frequencies once
        d = self.dim // 40
        self.register_buffer('freqs', torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ], dim=1))
        
        # Get blocks from original model
        if hasattr(model, 'blocks'):
            self.blocks = model.blocks
            self.num_blocks = len(self.blocks)
        else:
            raise ValueError("Model must have 'blocks' attribute")
        
        # Apply distribution strategy
        if strategy == 'auto':
            if self.num_devices == 4:
                self._setup_4gpu_strategy_fixed(model)
            else:
                self._setup_default_strategy(model)
        else:
            self._setup_default_strategy(model)
        
        logging.info(f"Model parallel setup complete with {self.num_devices} GPUs")
        # self._log_gpu_distribution()
    def clear_intermediate_gpu_memory(self):
        """Clear memory on intermediate GPUs to prepare for next timestep"""
        for device_idx in range(len(self.devices) - 1):  # Don't clear final device
            with torch.cuda.device(self.devices[device_idx]):
                torch.cuda.empty_cache()
    def _setup_4gpu_strategy_fixed(self, model):
        """Fixed 4-GPU strategy with proper parameter movement"""
        # Distribute blocks more evenly
        total_blocks = self.num_blocks
        gpu0_blocks = total_blocks // 5  # GPU 0 gets fewer blocks (initial processing)
        remaining_blocks = total_blocks - gpu0_blocks
        blocks_per_gpu = remaining_blocks // 3
        remainder = remaining_blocks % 3
        
        self.device_blocks = [
            list(range(0, gpu0_blocks)),  # GPU 0: fewer blocks
            list(range(gpu0_blocks, gpu0_blocks + blocks_per_gpu + (1 if remainder > 0 else 0))),
            list(range(gpu0_blocks + blocks_per_gpu + (1 if remainder > 0 else 0), 
                      gpu0_blocks + 2*blocks_per_gpu + (2 if remainder > 1 else 1 if remainder > 0 else 0))),
            list(range(gpu0_blocks + 2*blocks_per_gpu + (2 if remainder > 1 else 1 if remainder > 0 else 0), 
                      total_blocks))
        ]
        
        self._distribute_blocks_properly()
        
        # Set device assignments
        self.t5_device = self.devices[2]
        self.clip_device = self.devices[2]
        self.vae_device = self.devices[3]
    
    def _distribute_blocks_properly(self):
        """Properly distribute blocks with all parameters"""
        for device_idx, block_indices in enumerate(self.device_blocks):
            if not block_indices:
                continue
                
            device = self.devices[device_idx]
            logging.info(f"Moving blocks {block_indices[0]}-{block_indices[-1]} to {device}")
            
            for idx in block_indices:
                # Move entire block
                self.blocks[idx] = self.blocks[idx].to(device)
                
                # Ensure ALL parameters and buffers are on correct device
                for name, param in self.blocks[idx].named_parameters():
                    if param.device != device:
                        param.data = param.data.to(device)
                        logging.debug(f"Moved parameter {name} to {device}")
                
                for name, buffer in self.blocks[idx].named_buffers():
                    if buffer.device != device:
                        buffer.data = buffer.data.to(device)
                        logging.debug(f"Moved buffer {name} to {device}")
                
                # Special handling for modulation parameters that cause the error
                if hasattr(self.blocks[idx], 'modulation'):
                    if self.blocks[idx].modulation.device != device:
                        self.blocks[idx].modulation = self.blocks[idx].modulation.to(device)
                        logging.debug(f"Moved modulation parameter to {device}")
    
    def forward(self, x, t, context, clip_fea, seq_len, y):
        """Fixed forward pass using pre-initialized components"""
        
        device0 = self.devices[0]
        
        # 1. Move inputs to first device with BLOCKING transfers
        if isinstance(x, list):
            x = [xi.to(device0, non_blocking=False) for xi in x]  # BLOCKING
        else:
            x = [x.to(device0, non_blocking=False)]
        
        # 2. Process y if exists (same as your code)
        if y is not None:
            y = [yi.to(device0, non_blocking=False) for yi in y]
            x_new = []
            for i, (u, v) in enumerate(zip(x, y)):
                if u.dim() == 4:
                    u = u.unsqueeze(0)
                if v.dim() == 4:
                    v = v.unsqueeze(0)
                
                combined = torch.cat([u, v], dim=1)
                batch_size, channels, frames, height, width = combined.shape
                zero_channels = torch.zeros(batch_size, 4, frames, height, width, 
                                          dtype=combined.dtype, device=combined.device)
                final_input = torch.cat([combined, zero_channels], dim=1)
                x_new.append(final_input)
                del u, v, combined, zero_channels
            x = x_new
            del x_new
        else:
            x_new = []
            for u in x:
                if u.dim() == 4:
                    u = u.unsqueeze(0)
                batch_size, channels, frames, height, width = u.shape
                zero_channels = torch.zeros(batch_size, 36 - channels, frames, height, width,
                                          dtype=u.dtype, device=u.device)
                padded_input = torch.cat([u, zero_channels], dim=1)
                x_new.append(padded_input)
                del u, zero_channels
            x = x_new
            del x_new
        
        torch.cuda.synchronize(device0)  # Synchronize after input processing
        
        # 3. Use pre-initialized patch embedding (not create new one!)
        x = [self.patch_embedding(u.to(torch.bfloat16)) for u in x]
        
        # 4. Grid sizes and flattening
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        
        # 5. Sequence processing
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x
        ])
        
        # 6. Time embedding using pre-initialized layers
        t = t.to(device0, non_blocking=False)
        with torch.cuda.amp.autocast(dtype=torch.float32):
            e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        # with torch.cuda.amp.autocast(dtype=torch.float32):
        #     e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
        #     e_proj = self.time_projection(e)
        #     # Reshape for block processing (6 components for each block)
        #     e0 = e_proj.unflatten(1, (6, self.dim))
        #     # For head, we need just the basic time embedding (not projected)
        #     e_head = e  # Use the basic time embedding for head
        
        # 7. Context embedding using pre-initialized layer
        text_len = 512
        context = [c.to(device0, non_blocking=False) for c in context]
        
        context_padded = torch.stack([
            torch.cat([u, u.new_zeros(text_len - u.size(0), u.size(1))]) for u in context
        ])
        context_emb = self.text_embedding(context_padded)
        
        # 8. CLIP features using pre-initialized layer
        if clip_fea is not None:
            clip_fea = clip_fea.to(device0, non_blocking=False)
            context_clip = self.img_emb(clip_fea)
            context_emb = torch.concat([context_clip, context_emb], dim=1)
            del context_clip
        
        # 9. Use pre-initialized freqs
        freqs = self.freqs.to(device0)
        
        # 10. Prepare kwargs
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=freqs,
            context=context_emb,
            context_lens=None,
        )
        
        current_device = device0
        
        # 11. Process blocks with proper synchronization
        for device_idx, block_indices in enumerate(self.device_blocks):
            if not block_indices:
                continue
                
            device = self.devices[device_idx]
            logging.info(f"[Forward] Processing blocks on device {device} ({len(block_indices)} blocks)")
            
            # Move tensors with BLOCKING transfers and proper sync
            if device != current_device:
                # Synchronize source device
                torch.cuda.synchronize(current_device)
                
                # Move with blocking transfers
                x = x.to(device, non_blocking=False)
                
                # Move kwargs
                for k, v in kwargs.items():
                    if torch.is_tensor(v) and v.device != device:
                        kwargs[k] = v.to(device, non_blocking=False)
                
                # Synchronize target device
                torch.cuda.synchronize(device)
                current_device = device
            
            # Process blocks
            for block_idx in block_indices:
                logging.debug(f"Processing block {block_idx}")
                
                # Validate block is on correct device
                block_device = next(self.blocks[block_idx].parameters()).device
                if block_device != device:
                    logging.error(f"Block {block_idx} parameters on wrong device: {block_device} != {device}")
                    raise RuntimeError(f"Block {block_idx} device mismatch")
                
                x = self.blocks[block_idx](x, **kwargs)
            
            logging.info(f"[Forward] Completed device {device}, x shape: {x.shape}")
        
        # final_device = x.device
        # head_device = next(self.head.parameters()).device
        # if head_device != final_device:
        #     self.head = self.head.to(final_device)
        
        # # Move e_head to final device (use basic time embedding, not the 6-component version)
        # e_head = e_head.to(final_device, non_blocking=False)
        
        # x = self.head(x, e_head)
        print(f"Head input shapes: x={x.shape}, e_head={e.shape}")
        print(f"Head modulation shape: {self.head.modulation.shape}")
        e = e.to(x.device, non_blocking=False)
        
        x = self.head(x, e)
        
        # 13. Unpatchify
        def unpatchify(x, grid_sizes):
            c = 16
            out = []
            for u, v in zip(x, grid_sizes.tolist()):
                u = u[:math.prod(v)].view(*v, *self.patch_size, c)
                u = torch.einsum('fhwpqrc->cfphqwr', u)
                u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
                out.append(u)
            return out
        
        x = unpatchify(x, grid_sizes)
        return x
class FramePackingSampler:
    """Frame packing sampler for efficient video generation"""
    
    def __init__(self, model, vae, latent_window_size=1, vae_stride=(4, 8, 8)):
        self.model = model
        self.vae = vae
        self.latent_window_size = latent_window_size
        self.vae_stride = vae_stride
        
    def prepare_frame_packing_indices(self, latent_padding_size, latent_window_size):
        """Prepare indices for frame packing"""
        # Total size: 1 (start) + padding + window + 1 (end) + 2 (2x) + 16 (4x)
        total_size = 1 + latent_padding_size + latent_window_size + 1 + 2 + 16
        indices = torch.arange(0, total_size).unsqueeze(0)
        
        # Split indices according to frame packing strategy
        clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, \
        clean_latent_2x_indices, clean_latent_4x_indices = indices.split(
            [1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1
        )
        
        clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)
        
        return {
            'clean_indices': clean_latent_indices,
            'blank_indices': blank_indices,
            'latent_indices': latent_indices,
            'clean_2x_indices': clean_latent_2x_indices,
            'clean_4x_indices': clean_latent_4x_indices,
            'total_indices': indices
        }
    
    def pack_latents(self, start_latent, history_latents, latent_padding_size):
        """Pack latents according to the frame packing strategy"""
        if start_latent.dim() == 4:
            # Add batch dimension if missing: (C, F, H, W) -> (1, C, F, H, W)
            start_latent = start_latent.unsqueeze(0)
        
        if history_latents.dim() == 4:
            # Add batch dimension if missing: (C, F, H, W) -> (1, C, F, H, W)
            history_latents = history_latents.unsqueeze(0)
        
        # Split history latents into components
        clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
        
        # Ensure start_latent has same batch and channel dimensions as history
        clean_latents_pre = start_latent.to(history_latents.device).to(history_latents.dtype)
        
        # Verify dimensions before concatenation
        logging.info(f"clean_latents_pre shape: {clean_latents_pre.shape}")
        logging.info(f"clean_latents_post shape: {clean_latents_post.shape}")
        
        # Combine start and post latents
        clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
        
        # Create blank latents for padding
        if latent_padding_size > 0:
            blank_latents = torch.zeros(
                1, 16, latent_padding_size, 
                history_latents.shape[3], history_latents.shape[4],
                dtype=history_latents.dtype,
                device=history_latents.device
            )
        else:
            blank_latents = None
        
        # Create noise for the main window
        noise_latents = torch.randn(
            1, 16, self.latent_window_size,
            history_latents.shape[3], history_latents.shape[4],
            dtype=history_latents.dtype,
            device=history_latents.device
        )
        
        # Pack all components
        packed_components = [clean_latents_pre]
        if blank_latents is not None:
            packed_components.append(blank_latents)
        packed_components.extend([
            noise_latents,
            clean_latents_post,
            clean_latents_2x,
            clean_latents_4x
        ])
        
        packed_latents = torch.cat(packed_components, dim=2)
        
        return packed_latents, {
            'clean_latents': clean_latents,
            'clean_latents_2x': clean_latents_2x,
            'clean_latents_4x': clean_latents_4x,
            'noise_latents': noise_latents
        }
    
    def unpack_latents(self, generated_latents, indices, latent_padding_size):
        """Unpack generated latents back to individual components"""
        # Extract the main generated window (skip padding)
        start_idx = 1 + latent_padding_size
        end_idx = start_idx + self.latent_window_size
        
        main_latents = generated_latents[:, :, start_idx:end_idx, :, :]
        
        # Extract clean latents for history update
        clean_post = generated_latents[:, :, end_idx:end_idx+1, :, :]
        clean_2x = generated_latents[:, :, end_idx+1:end_idx+3, :, :]
        clean_4x = generated_latents[:, :, end_idx+3:end_idx+19, :, :]
        
        return main_latents, {
            'clean_post': clean_post,
            'clean_2x': clean_2x,
            'clean_4x': clean_4x
        }


class EnhancedGPUResourceManager:
    """Enhanced GPU memory manager with better resource tracking"""
    
    def __init__(self):
        self.locks = {i: threading.Lock() for i in range(torch.cuda.device_count())}
        self.available_memory = {}
        self.reserved_memory = {}  # Track memory reserved for specific operations
        self.update_memory_status()
    
    def update_memory_status(self):
        """Update available memory for each GPU"""
        for i in range(torch.cuda.device_count()):
            with self.locks[i]:
                torch.cuda.synchronize(i)
                free, total = torch.cuda.mem_get_info(i)
                self.available_memory[i] = free
                # Reserve some memory for safety
                self.reserved_memory[i] = total * 0.1  # 10% buffer
    
    def get_best_gpu(self, required_memory=None):
        """Get GPU with most available memory"""
        self.update_memory_status()
        
        if required_memory:
            # Find GPU with enough memory
            suitable_gpus = [
                (gpu_id, mem) for gpu_id, mem in self.available_memory.items()
                if mem - self.reserved_memory[gpu_id] >= required_memory
            ]
            if suitable_gpus:
                return max(suitable_gpus, key=lambda x: x[1])[0]
        
        # Return GPU with most memory
        return max(self.available_memory.items(), key=lambda x: x[1])[0]
    
    def release_gpu_resources(self, device_id):
        """Force release GPU resources"""
        with self.locks[device_id]:
            torch.cuda.synchronize(device_id)
            torch.cuda.empty_cache()
            gc.collect()
    
    def log_memory_status(self, prefix=""):
        """Log current memory status of all GPUs"""
        logging.info(f"{prefix} GPU Memory Status:")
        for i in range(torch.cuda.device_count()):
            total_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            used_mem = (torch.cuda.memory_allocated(i) + torch.cuda.memory_reserved(i)) / 1e9
            free_mem = self.available_memory.get(i, 0) / 1e9
            logging.info(f"  GPU {i}: {used_mem:.2f}/{total_mem:.2f} GB used, {free_mem:.2f} GB free")


class WanI2V:
    """Enhanced WanI2V with improved multi-GPU support"""
    
    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        init_on_cpu=True,
        multi_gpu=False,
        model_parallel=True,
        parallel_strategy='auto',  # New parameter for parallel strategy
    ):
        # Auto-detect multi-GPU if not specified
        if multi_gpu is None:
            multi_gpu = torch.cuda.device_count() > 1
        
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.use_usp = use_usp
        self.t5_cpu = t5_cpu
        self.multi_gpu = multi_gpu
        self.model_parallel = model_parallel and multi_gpu
        self.parallel_strategy = parallel_strategy
        
        # Initialize enhanced GPU resource manager
        self.gpu_manager = EnhancedGPUResourceManager() if multi_gpu else None
        
        self.num_train_timesteps = config.num_train_timesteps
        # self.param_dtype = config.param_dtype
        self.param_dtype=torch.bfloat16 
        # Log initial setup
        logging.info(f"Initializing WanI2V with {torch.cuda.device_count()} GPUs available")
        logging.info(f"Model parallel: {self.model_parallel}, Strategy: {parallel_strategy}")
        
        # Initialize components
        self._initialize_components(checkpoint_dir, device_id, t5_fsdp, dit_fsdp, init_on_cpu)
        
        self.sample_neg_prompt = config.sample_neg_prompt
    
    def _initialize_components(self, checkpoint_dir, device_id, t5_fsdp, dit_fsdp, init_on_cpu):
        """Initialize all model components with proper device placement"""
        shard_fn = partial(shard_model, device_id=device_id)
        
        # Initialize T5 text encoder - always on CPU initially to save GPU memory
        print('--------------loading encoder to gpu------------')
        self.text_encoder = T5EncoderModel(
            text_len=self.config.text_len,
            dtype=self.config.t5_dtype,
            device=torch.device('cpu'),  # Always start on CPU
            checkpoint_path=os.path.join(checkpoint_dir, self.config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, self.config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )
        
        # Initialize VAE
        self.vae_stride = self.config.vae_stride
        self.patch_size = self.config.patch_size
        
        # For model parallel, VAE will be assigned to specific GPU later
        vae_device = self.device
        if self.model_parallel and self.parallel_strategy == 'auto':
            # VAE will be on last GPU for most strategies
            num_gpus = torch.cuda.device_count()
            if num_gpus >= 2:
                vae_device = torch.device(f"cuda:{num_gpus-1}")
        
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, self.config.vae_checkpoint),
            device=vae_device
        )
        
        # Initialize CLIP - will be moved to appropriate GPU later if using model parallel
        clip_device = self.device
        if self.model_parallel and self.parallel_strategy == 'auto':
            num_gpus = torch.cuda.device_count()
            if num_gpus == 4:
                clip_device = torch.device("cuda:2")  # GPU 2 for CLIP in 4-GPU setup
        
        self.clip = CLIPModel(
            dtype=self.config.clip_dtype,
            device=clip_device,
            checkpoint_path=os.path.join(checkpoint_dir, self.config.clip_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, self.config.clip_tokenizer)
        )
        
        # Initialize main model
        logging.info(f"Creating WanModel from {checkpoint_dir}")
        
        # Apply model parallelism if enabled
        if self.model_parallel:
            devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
            logging.info(f"Enabling model parallelism across {len(devices)} GPUs with strategy: {self.parallel_strategy}")
            
            # Load model with CPU first to avoid OOM
            with torch.device('cpu'):
                self.model = WanModel.from_pretrained(checkpoint_dir)
                self.model.eval().requires_grad_(False)
                # Convert to appropriate dtype on CPU
                self.model = self.model.to(dtype=self.param_dtype)
            
            # Now wrap with model parallelism which will distribute blocks to GPUs
            self.model = ModelParallelWrapper(
                self.model, 
                devices, 
                vae=self.vae,
                strategy=self.parallel_strategy
            )
            
            # Update VAE device if set by wrapper
            if hasattr(self.model, 'vae_device'):
                # Check if VAE has a device attribute and update it
                if hasattr(self.vae, 'device'):
                    self.vae.device = self.model.vae_device
                # If VAE has a model attribute, move that
                if hasattr(self.vae, 'model'):
                    self.vae.model = self.vae.model.to(self.model.vae_device)
                logging.info(f"VAE assigned to {self.model.vae_device}")
        else:
            # Standard model loading
            if init_on_cpu:
                with torch.device('cpu'):
                    self.model = WanModel.from_pretrained(checkpoint_dir)
                    self.model.eval().requires_grad_(False)
                    self.model = self.model.to(dtype=self.param_dtype)
                # Then move to GPU if not using FSDP
                if not dit_fsdp:
                    self.model = self.model.to(device=self.device)
            else:
                # Load directly to GPU (original behavior)
                self.model = WanModel.from_pretrained(checkpoint_dir)
                self.model.eval().requires_grad_(False)
                if dit_fsdp:
                    self.model = shard_fn(self.model)
                else:
                    self.model = self.model.to(device=self.device, dtype=self.param_dtype)
        
        # Initialize model copies for data parallelism (if not using model parallelism)
        if self.multi_gpu and not self.model_parallel:
            self.num_gpus = torch.cuda.device_count()
            logging.info(f"Data parallel mode with {self.num_gpus} GPUs")
            self.models = {}
        else:
            self.models = None

    def get_model_for_device(self, device_id):
        """Get or create model copy for specific device (data parallel mode)"""
        if self.model_parallel:
            return self.model
        
        if self.models is None:
            return self.model
        
        if device_id not in self.models:
            # Create a copy of the model for this device
            device = torch.device(f"cuda:{device_id}")
            self.models[device_id] = copy.deepcopy(self.model).to(device)
            logging.info(f"Created model copy for GPU {device_id}")
        
        return self.models[device_id]

    def generate_segment(self,
                        noise,
                        context,
                        context_null,
                        clip_context,
                        y,
                        max_seq_len,
                        shift,
                        sample_solver,
                        sampling_steps,
                        guide_scale,
                        seed_g,
                        device_id,
                        offload_model=True,
                        conditioning_frames=None):
        """Generate a single video segment with improved memory management"""
        
        device = torch.device(f"cuda:{device_id}")
        
        # Get appropriate model
        if self.model_parallel:
            model = self.model
        else:
            model = self.get_model_for_device(device_id)
        
        @contextmanager
        def noop_no_sync():
            yield
        
        no_sync = getattr(model, 'no_sync', noop_no_sync)
        
        try:
            with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
                # Initialize scheduler
                if sample_solver == 'unipc':
                    sample_scheduler = FlowUniPCMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,
                        use_dynamic_shifting=False
                    )
                    sample_scheduler.set_timesteps(sampling_steps, device=device, shift=shift)
                    timesteps = sample_scheduler.timesteps
                elif sample_solver == 'dpm++':
                    sample_scheduler = FlowDPMSolverMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,
                        use_dynamic_shifting=False
                    )
                    sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                    timesteps, _ = retrieve_timesteps(
                        sample_scheduler, device=device, sigmas=sampling_sigmas
                    )
                else:
                    raise NotImplementedError(f"Unsupported solver: {sample_solver}")
                
                # Apply conditioning if provided
                if conditioning_frames is not None:
                    conditioning_frames = conditioning_frames.to(device)
                    overlap_frames = conditioning_frames.shape[1]
                    blend_weights = torch.linspace(0.7, 0.3, overlap_frames, device=device).view(1, -1, 1, 1)
                    noise[:, :overlap_frames] = (
                        blend_weights * conditioning_frames + 
                        (1 - blend_weights) * noise[:, :overlap_frames]
                    )
                
                # Prepare inputs
                latent = noise.to(device)
                clip_context = clip_context.to(device)
                y = [y.to(device)] if not isinstance(y, list) else [yi.to(device) for yi in y]
                context = [t.to(device) for t in context]
                context_null = [t.to(device) for t in context_null]
                
                arg_c = {
                    'context': context,
                    'clip_fea': clip_context,
                    'seq_len': max_seq_len,
                    'y': y
                }
                arg_null = {
                    'context': context_null,
                    'clip_fea': clip_context,
                    'seq_len': max_seq_len,
                    'y': y
                }
                test_timesteps = timesteps[:2]
                
                # Diffusion loop with periodic cleanup
                for step_idx, t in enumerate(test_timesteps):
                    latent_model_input = [latent]
                    timestep = torch.stack([t]).to(device)
                    with torch.no_grad():
                    # Generate predictions
                        noise_pred_cond = model(latent_model_input, t=timestep, **arg_c)
                        if isinstance(noise_pred_cond, list):
                            noise_pred_cond = noise_pred_cond[0]
                        
                        noise_pred_uncond = model(latent_model_input, t=timestep, **arg_null)
                        if isinstance(noise_pred_uncond, list):
                            noise_pred_uncond = noise_pred_uncond[0]
                        
                        noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)
                        
                        # Step scheduler
                        temp_x0 = sample_scheduler.step(
                            noise_pred.unsqueeze(0),
                            t,
                            latent.unsqueeze(0),
                            return_dict=False,
                            generator=seed_g
                        )[0]
                        latent = temp_x0.squeeze(0)
                    
                    # Clean up intermediate tensors
                    del latent_model_input, timestep, noise_pred_cond, noise_pred_uncond, noise_pred, temp_x0
                    
                    # Periodic memory cleanup
                    if step_idx % 5 == 0:
                        torch.cuda.synchronize(device)
                        torch.cuda.empty_cache()
                
                return latent
                
        except Exception as e:
            logging.error(f"Error in generate_segment on device {device_id}: {str(e)}")
            raise
        finally:
            # Always clean up
            if self.gpu_manager and offload_model:
                self.gpu_manager.release_gpu_resources(device_id)

    def _generate_sequential(self, num_segments, segment_frames, lat_h, lat_w,
                           h, w, context, context_null, clip_context,
                           max_seq_len, shift, sample_solver, sampling_steps,
                           guide_scale, seed_g, img, offload_model, overlap_frames):
        """Sequential generation for single GPU or when parallel is disabled"""
        all_latents = []
        
        for seg_idx in range(num_segments):
            logging.info(f"Generating segment {seg_idx + 1}/{num_segments}")
            
            # Prepare mask
            msk = torch.ones(1, segment_frames, lat_h, lat_w, device=self.device)
            if seg_idx == 0:
                msk[:, 1:] = 0
            else:
                msk[:, :] = 0
            
            msk = torch.concat([
                torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), 
                msk[:, 1:]
            ], dim=1)
            msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
            msk = msk.transpose(1, 2)[0]
            
            # Prepare y tensor
            if seg_idx == 0:
                y_frames = torch.concat([
                    torch.nn.functional.interpolate(
                        img[None].cpu(), size=(h, w), mode='bicubic'
                    ).transpose(0, 1),
                    torch.zeros(3, segment_frames - 1, h, w)
                ], dim=1).to(self.device)
            else:
                y_frames = torch.zeros(3, segment_frames, h, w, device=torch.device("cuda:3"))
            
            y_frames = y_frames.to('cuda:3')

            y = self.vae.encode([y_frames])[0]
            msk= msk.to('cuda:3')

            y = torch.concat([msk, y])
            
            # Generate noise
            segment_seed_g = torch.Generator(device=self.device)
            segment_seed_g.manual_seed(seed_g.initial_seed() + seg_idx)
            
            noise = torch.randn(
                16, (segment_frames - 1) // 4 + 1,
                lat_h, lat_w,
                dtype=torch.float16,
                generator=segment_seed_g,
                device=self.device
            )
            
            # Get conditioning from previous segment
            conditioning_frames = None
            if seg_idx > 0 and overlap_frames > 0:
                conditioning_frames = all_latents[-1][:, -overlap_frames // 4:]
            
            # Generate segment
            latent = self.generate_segment(
                noise=noise,
                context=context,
                context_null=context_null,
                clip_context=clip_context,
                y=y,
                max_seq_len=4096,
                shift=shift,
                sample_solver=sample_solver,
                sampling_steps=sampling_steps,
                guide_scale=guide_scale,
                seed_g=segment_seed_g,
                device_id=self.device.index,
                offload_model=offload_model,
                conditioning_frames=conditioning_frames
            )
            
            # Store result
            if seg_idx == 0:
                all_latents.append(latent)
            else:
                all_latents.append(latent[:, overlap_frames // 4:])
            
            # Cleanup
            del msk, y_frames, y, noise
            torch.cuda.empty_cache()
        
        return all_latents

    def _generate_parallel_with_resource_management(self, num_segments, segment_frames, lat_h, lat_w,
                                                   h, w, context, context_null, clip_context,
                                                   max_seq_len, shift, sample_solver, sampling_steps,
                                                   guide_scale, seed_g, img, offload_model, overlap_frames):
        """Enhanced parallel generation with better resource management"""
        
        num_gpus = torch.cuda.device_count()
        logging.info(f"Parallel generation across {num_gpus} GPUs")
        
        if self.gpu_manager:
            self.gpu_manager.log_memory_status("Before generation")
        
        # Pre-generate segment data with memory-efficient approach
        segment_data = []
        
        for seg_idx in range(num_segments):
            # Find best GPU for this segment
            if self.gpu_manager:
                # Estimate required memory (rough estimate)
                required_memory = 4 * 1024 * 1024 * 1024  # 4GB estimate per segment
                device_id = self.gpu_manager.get_best_gpu(required_memory)
            else:
                device_id = seg_idx % num_gpus
            
            device = torch.device(f"cuda:{device_id}")
            
            # Prepare segment data
            with torch.cuda.device(device):
                # Create mask
                msk = torch.ones(1, segment_frames, lat_h, lat_w, device=device)
                if seg_idx == 0:
                    msk[:, 1:] = 0
                else:
                    msk[:, :] = 0
                
                msk = torch.concat([
                    torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), 
                    msk[:, 1:]
                ], dim=1)
                msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
                msk = msk.transpose(1, 2)[0]
                
                # Prepare y tensor efficiently
                if seg_idx == 0:
                    # Process on CPU first to save GPU memory
                    y_frames = torch.concat([
                        torch.nn.functional.interpolate(
                            img[None].cpu(), size=(h, w), mode='bicubic'
                        ).transpose(0, 1),
                        torch.zeros(3, segment_frames - 1, h, w)
                    ], dim=1)
                    y_frames = y_frames.to(device)
                else:
                    y_frames = torch.zeros(3, segment_frames, h, w, device=device)
                
                # Encode with VAE (might be on different device)
                vae_device = getattr(self.vae, 'device', self.device)
                if vae_device != device:
                    y_frames = y_frames.to(vae_device)
                    y = self.vae.encode([y_frames])[0]
                    y = y.to(device)
                else:
                    y = self.vae.encode([y_frames])[0]
                
                y = torch.concat([msk.to(device), y])
                
                # Generate noise
                segment_seed_g = torch.Generator(device=device)
                segment_seed_g.manual_seed(seed_g.initial_seed() + seg_idx)
                
                noise = torch.randn(
                    16, (segment_frames - 1) // 4 + 1,
                    lat_h, lat_w,
                    dtype=torch.float16,
                    generator=segment_seed_g,
                    device=device
                )
                
                segment_data.append({
                    'seg_idx': seg_idx,
                    'device_id': device_id,
                    'noise': noise,
                    'y': y,
                    'seed_g': segment_seed_g
                })
                
                # Clean up temporary tensors
                del msk, y_frames
                torch.cuda.empty_cache()
        
        # Process segments with controlled parallelism
        all_latents = [None] * num_segments
        
        # Determine max parallel segments based on available memory
        if self.model_parallel:
            # With model parallelism, we can be more aggressive
            max_parallel = min(num_gpus, 4)
        else:
            # Without model parallelism, be conservative
            max_parallel = min(num_gpus, 2)
        
        logging.info(f"Processing segments with max {max_parallel} parallel generations")
        
        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            # Process in batches to avoid OOM
            for batch_start in range(0, num_segments, max_parallel):
                batch_end = min(batch_start + max_parallel, num_segments)
                batch_data = segment_data[batch_start:batch_end]
                
                futures = []
                for data in batch_data:
                    # Get conditioning from previous segment if available
                    conditioning_frames = None
                    if data['seg_idx'] > 0 and overlap_frames > 0 and all_latents[data['seg_idx'] - 1] is not None:
                        prev_latent = all_latents[data['seg_idx'] - 1]
                        conditioning_frames = prev_latent[:, -overlap_frames // 4:].to(data['noise'].device)
                    
                    future = executor.submit(
                        self.generate_segment,
                        noise=data['noise'],
                        context=context,
                        context_null=context_null,
                        clip_context=clip_context,
                        y=data['y'],
                        max_seq_len=max_seq_len,
                        shift=shift,
                        sample_solver=sample_solver,
                        sampling_steps=sampling_steps,
                        guide_scale=guide_scale,
                        seed_g=data['seed_g'],
                        device_id=data['device_id'],
                        offload_model=offload_model,
                        conditioning_frames=conditioning_frames
                    )
                    futures.append((data['seg_idx'], future))
                
                # Collect batch results
                for seg_idx, future in futures:
                    try:
                        latent = future.result()
                        
                        # Move to main device for storage
                        latent = latent.to(self.device)
                        
                        if seg_idx == 0:
                            all_latents[seg_idx] = latent
                        else:
                            # Remove overlap frames
                            all_latents[seg_idx] = latent[:, overlap_frames // 4:]
                        
                        logging.info(f"Completed segment {seg_idx + 1}/{num_segments}")
                        
                    except Exception as e:
                        logging.error(f"Error processing segment {seg_idx}: {str(e)}")
                        raise
                
                # Clean up after each batch
                torch.cuda.synchronize()
                if self.gpu_manager:
                    for i in range(num_gpus):
                        self.gpu_manager.release_gpu_resources(i)
                
                # Log memory status after batch
                if self.gpu_manager:
                    self.gpu_manager.log_memory_status(f"After batch {batch_start//max_parallel + 1}")
        
        # Final cleanup
        gc.collect()
        torch.cuda.empty_cache()
        
        return all_latents
    
    def generate_segment_with_frame_packing(self,
                                       packed_latents,
                                       mask,
                                       context,
                                       context_null,
                                       clip_context,
                                       latent_window_size,
                                       mask_start,
                                       mask_end,
                                       lat_h,
                                       lat_w,
                                       shift,
                                       sample_solver,
                                       sampling_steps,
                                       guide_scale,
                                       seed_g,
                                       device_id,
                                       offload_model=True,
                                       progress_callback=None):
        """Generate a segment using frame packing technique - FIXED VERSION"""
        
        device = torch.device(f"cuda:{device_id}")
        
        # Get appropriate model
        if self.model_parallel:
            model = self.model
        else:
            model = self.get_model_for_device(device_id)
        
        try:
            with amp.autocast(dtype=self.param_dtype), torch.no_grad():
                # FIXED: Use the EXACT same scheduler setup as the working generate_segment
                print(f"Setting up scheduler: {sample_solver}, steps: {sampling_steps}")
                
                if sample_solver == 'unipc':
                    sample_scheduler = FlowUniPCMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,  # Use 1 like in working version
                        use_dynamic_shifting=False
                    )
                    sample_scheduler.set_timesteps(sampling_steps, device=device, shift=shift)
                    timesteps = sample_scheduler.timesteps
                    
                elif sample_solver == 'dpm++':
                    sample_scheduler = FlowDPMSolverMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,  # Use 1 like in working version
                        use_dynamic_shifting=False
                    )
                    # Use the EXACT same approach as working version
                    sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                    timesteps, _ = retrieve_timesteps(
                        sample_scheduler, device=device, sigmas=sampling_sigmas
                    )
                    
                else:
                    raise NotImplementedError(f"Unsupported solver: {sample_solver}")
                
                print(f"Scheduler setup complete. Timesteps: {len(timesteps)}")
                print(f"First few timesteps: {timesteps[:5] if len(timesteps) > 5 else timesteps}")
                
                # Validate timesteps
                if len(timesteps) != sampling_steps:
                    print(f"WARNING: Expected {sampling_steps} timesteps, got {len(timesteps)}")
                    if len(timesteps) == 0:
                        raise ValueError("Scheduler generated no timesteps")
                
                # Move inputs to device
                packed_latents = packed_latents.to(device, dtype=torch.float16)
                mask = mask.to(device, dtype=torch.float16)
                context = [t.to(device, dtype=torch.float16) for t in context]
                context_null = [t.to(device, dtype=torch.float16) for t in context_null]
                clip_context = clip_context.to(device, dtype=torch.float16)
                
                # y for conditioning
                y = packed_latents
                
                # Extract just the noise window for denoising
                current_latent = packed_latents[:, :, mask_start:mask_end, :, :].clone()
                
                # CRITICAL FIX: Ensure current_latent always has 5 dimensions
                print(f"Before fix - current_latent shape: {current_latent.shape}")
                print(f"packed_latents shape: {packed_latents.shape}")
                print(f"Extracting frames {mask_start}:{mask_end}")
                
                if current_latent.dim() == 4:
                    # This happens when mask_start:mask_end gives only 1 frame
                    # Add the batch dimension back
                    current_latent = current_latent.unsqueeze(0)
                    print(f"Added batch dimension: {current_latent.shape}")
                elif current_latent.dim() == 3:
                    # This happens if the frame dimension collapses
                    # Reshape to [B, C, T, H, W] format
                    current_latent = current_latent.unsqueeze(0).unsqueeze(2)
                    print(f"Reshaped to 5D: {current_latent.shape}")
                
                # Ensure minimum temporal dimension
                if current_latent.shape[2] == 0:
                    # If no frames extracted, create a single frame
                    current_latent = packed_latents[:, :, mask_start:mask_start+1, :, :].clone()
                    print(f"Created single frame: {current_latent.shape}")
                
                print(f"Final current_latent shape: {current_latent.shape}")
                
                # Calculate max sequence length for the full packed input
                max_seq_len = packed_latents.shape[2] * lat_h * lat_w // (self.patch_size[1] * self.patch_size[2])
                self.sp_size = getattr(self, 'sp_size', 64)
                max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size
                
                print(f"Max sequence length: {max_seq_len}")
                
                # Main sampling loop
                for step_idx, t in enumerate(timesteps):
                    print(f"\n=== Step {step_idx + 1}/{len(timesteps)}, timestep: {t} ===")
                    
                    if progress_callback:
                        progress_callback({'step': step_idx + 1, 'total_steps': len(timesteps)})
                    
                    # Reconstruct full packed input with current latent
                    packed_input = packed_latents.clone()
                    packed_input[:, :, mask_start:mask_end, :, :] = current_latent
                    
                    print(f"Step {step_idx}: current_latent shape: {current_latent.shape}")
                    print(f"Step {step_idx}: packed_input shape: {packed_input.shape}")
                    
                    # Ensure timestep is properly formatted
                    if isinstance(t, torch.Tensor):
                        timestep = t.unsqueeze(0) if t.dim() == 0 else t
                    else:
                        timestep = torch.tensor([t], device=device, dtype=torch.float32)
                    
                    # Model arguments
                    arg_c = {
                        'context': context,
                        'clip_fea': clip_context,
                        'seq_len': max_seq_len,
                        'y': [y]
                    }
                    arg_null = {
                        'context': context_null,
                        'clip_fea': clip_context,
                        'seq_len': max_seq_len,
                        'y': [y]
                    }
                    
                    # SIMPLIFIED: Use the same model call pattern as working version
                    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                        # Prepare model input in the same format as working version
                        latent_model_input = [packed_input]
                        timestep = torch.stack([t]).to(device)  # Same format as working version
                        
                        # Process conditional - same pattern as working version
                        noise_pred_cond = model(latent_model_input, t=timestep, **arg_c)
                        if isinstance(noise_pred_cond, list):
                            noise_pred_cond = noise_pred_cond[0]
                        
                        print(f"Raw model output shape: {noise_pred_cond.shape}")
                        
                        # Process the output to get the right window
                        noise_pred_cond = self._extract_window_simple(noise_pred_cond, mask_start, mask_end, current_latent)
                        
                        # Process unconditional - same pattern
                        noise_pred_uncond = model(latent_model_input, t=timestep, **arg_null)
                        if isinstance(noise_pred_uncond, list):
                            noise_pred_uncond = noise_pred_uncond[0]
                        
                        noise_pred_uncond = self._extract_window_simple(noise_pred_uncond, mask_start, mask_end, current_latent)
                        
                        # Guidance - same as working version
                        noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)
                        
                        # Clean up
                        del noise_pred_cond, noise_pred_uncond, latent_model_input
                    
                    # Ensure current_latent and noise_pred have compatible shapes
                    if current_latent.shape != noise_pred.shape:
                        print(f"Adjusting shapes: current_latent={current_latent.shape}, noise_pred={noise_pred.shape}")
                        current_latent = self._match_tensor_shapes(current_latent, noise_pred)
                    
                    print(f"Before scheduler: current_latent={current_latent.shape}, noise_pred={noise_pred.shape}")
                    print(f"Timestep value: {t}, type: {type(t)}")
                    
                    # FIXED: Handle scheduler dimensions correctly
                    try:
                        print(f"Before scheduler: current_latent={current_latent.shape}, noise_pred={noise_pred.shape}")
                        
                        # Check if we need to add dimensions for scheduler
                        if current_latent.dim() == 5 and noise_pred.dim() == 5:
                            # Both are 5D, scheduler expects this format
                            temp_x0 = sample_scheduler.step(
                                noise_pred,  # Don't add extra dimension
                                t,
                                current_latent,  # Don't add extra dimension
                                return_dict=False,
                                generator=seed_g
                            )[0]
                            current_latent = temp_x0  # Don't squeeze
                            
                        elif current_latent.dim() == 4 and noise_pred.dim() == 4:
                            # Both are 4D, need to add batch dimension
                            temp_x0 = sample_scheduler.step(
                                noise_pred.unsqueeze(0),
                                t,
                                current_latent.unsqueeze(0),
                                return_dict=False,
                                generator=seed_g
                            )[0]
                            current_latent = temp_x0.squeeze(0)
                            
                        else:
                            # Mixed dimensions, standardize to 5D
                            if noise_pred.dim() == 4:
                                noise_pred = noise_pred.unsqueeze(0)
                            if current_latent.dim() == 4:
                                current_latent = current_latent.unsqueeze(0)
                                
                            temp_x0 = sample_scheduler.step(
                                noise_pred,
                                t,
                                current_latent,
                                return_dict=False,
                                generator=seed_g
                            )[0]
                            current_latent = temp_x0
                        
                        print(f"After scheduler: current_latent={current_latent.shape}")
                        
                        # Ensure current_latent stays 5D
                        if current_latent.dim() == 4:
                            current_latent = current_latent.unsqueeze(0)
                            print(f"Fixed to 5D: {current_latent.shape}")
                        
                    except Exception as scheduler_error:
                        print(f"Scheduler error: {scheduler_error}")
                        print(f"Timestep: {t}, type: {type(t)}")
                        print(f"noise_pred shape: {noise_pred.shape}")
                        print(f"current_latent shape: {current_latent.shape}")
                        raise
                    
                    del noise_pred
                    
                    # Memory cleanup
                    if step_idx % 5 == 0:
                        torch.cuda.empty_cache()
                    print("=== TESTING: Breaking after first timestep ===")
                    break
                
                print(f"Final latent shape: {current_latent.shape}")
                return current_latent
                
        except Exception as e:
            logging.error(f"Error in frame packing generation: {str(e)}")
            raise
        finally:
            if offload_model and self.gpu_manager:
                self.gpu_manager.release_gpu_resources(device_id)

    def _process_model_output(self, model_output, target_latent, mask_start, mask_end, output_type):
        """Process model output to match target shape"""
        print(f"Processing {output_type} output: {model_output.shape} -> target: {target_latent.shape}")
        
        # Handle 4D output (missing batch dimension)
        if model_output.dim() == 4:
            model_output = model_output.unsqueeze(0)  # Add batch dimension
        
        # If output has more frames than needed, we need to handle this carefully
        if model_output.dim() == 5:
            b, c, t, h, w = model_output.shape
            target_b, target_c, target_t, target_h, target_w = target_latent.shape
            
            print(f"Model output frames: {t}, target frames: {target_t}")
            
            # Fix spatial dimensions first
            if h != target_h or w != target_w:
                print(f"Resizing from ({h}, {w}) to ({target_h}, {target_w})")
                model_output = model_output.view(b * c * t, 1, h, w)
                model_output = torch.nn.functional.interpolate(
                    model_output, 
                    size=(target_h, target_w), 
                    mode='bilinear', 
                    align_corners=False
                )
                model_output = model_output.view(b, c, t, target_h, target_w)
            
            # Handle temporal dimension
            if t > target_t:
                # Model predicted more frames - extract the relevant window
                window_size = mask_end - mask_start
                if t >= mask_end:
                    # Extract the specific frames we need
                    model_output = model_output[:, :, mask_start:mask_end, :, :]
                    print(f"Extracted frames {mask_start}:{mask_end} from {t} total frames")
                else:
                    # Take the last few frames if we don't have enough
                    model_output = model_output[:, :, -window_size:, :, :]
                    print(f"Took last {window_size} frames from {t} total frames")
            
            elif t < target_t:
                # Model predicted fewer frames - repeat or pad
                repeat_factor = target_t // t
                remainder = target_t % t
                
                if repeat_factor > 0:
                    repeated = model_output.repeat(1, 1, repeat_factor, 1, 1)
                    if remainder > 0:
                        extra = model_output[:, :, :remainder, :, :]
                        model_output = torch.cat([repeated, extra], dim=2)
                    else:
                        model_output = repeated
                    print(f"Repeated {t} frames to get {model_output.shape[2]} frames")
        
        print(f"Final {output_type} shape: {model_output.shape}")
        return model_output

    def _extract_window_simple(self, model_output, mask_start, mask_end, target_latent):
        """Simple window extraction without complex processing"""
        
        # Add batch dimension if missing
        if model_output.dim() == 4:
            model_output = model_output.unsqueeze(0)
        
        print(f"Extracting window {mask_start}:{mask_end} from output shape {model_output.shape}")
        
        # If output has the right number of dimensions, try direct extraction
        if model_output.dim() == 5:
            b, c, t, h, w = model_output.shape
            
            # Handle target_latent dimensions properly
            if target_latent.dim() == 4:
                # Add batch dimension to target_latent
                target_latent = target_latent.unsqueeze(0)
                print(f"Added batch dimension to target_latent: {target_latent.shape}")
            
            if target_latent.dim() == 5:
                target_b, target_c, target_t, target_h, target_w = target_latent.shape
            else:
                raise ValueError(f"target_latent has unexpected dimensions: {target_latent.shape}")
            
            # Fix spatial dimensions if needed
            if h != target_h or w != target_w:
                model_output = torch.nn.functional.interpolate(
                    model_output.view(b * c * t, 1, h, w),
                    size=(target_h, target_w),
                    mode='bilinear',
                    align_corners=False
                ).view(b, c, t, target_h, target_w)
            
            # Extract window - be more flexible
            if t >= mask_end:
                # We have enough frames
                extracted = model_output[:, :, mask_start:mask_end, :, :]
            elif t > mask_start:
                # Partial overlap
                available = min(t - mask_start, mask_end - mask_start)
                extracted = model_output[:, :, mask_start:mask_start + available, :, :]
                
                # Pad if needed
                if available < (mask_end - mask_start):
                    needed = (mask_end - mask_start) - available
                    padding = extracted[:, :, -1:, :, :].repeat(1, 1, needed, 1, 1)
                    extracted = torch.cat([extracted, padding], dim=2)
            else:
                # No overlap, just take the last frames and repeat
                needed_frames = mask_end - mask_start
                extracted = model_output[:, :, -1:, :, :].repeat(1, 1, needed_frames, 1, 1)
            
            print(f"Extracted shape: {extracted.shape}")
            return extracted
        
        else:
            raise ValueError(f"Unexpected model output shape: {model_output.shape}")

    def generate(self,
        input_prompt,
        img,
        max_area=int(480 * 832 * 0.7),
       
                                        total_frames=4,
                                        latent_window_size=2,
                                        frame_num=None,  # For compatibility
                                        total_duration_seconds=None,
                                        fps=30,
                                        shift=3.0,
                                        sample_solver='dpm++',
                                        sampling_steps=20,
                                        guide_scale=3.5,
                                        n_prompt="",
                                        seed=-1,
                                        offload_model=False,
                                        parallel_mode='sequential'):
        """Main generation with frame packing - optimized version"""
        
        # Calculate total frames if duration specified
        if total_duration_seconds is not None:
            total_frames = int(fps * total_duration_seconds)
        
        # Setup
        device = self.device
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=device)
        seed_g.manual_seed(seed)
        
        # Process image
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(device)
        h, w = img.shape[1:]
        
        # Calculate latent dimensions
        lat_h = h // self.vae_stride[1]
        lat_w = w // self.vae_stride[2]
        
        # Encode prompts
        logging.info("Encoding prompts...")
        self.text_encoder.model.to("cuda:3")
        context = self.text_encoder([input_prompt],device='cuda:3')
        context_null = self.text_encoder([n_prompt or self.sample_neg_prompt],device='cuda:3')
        self.text_encoder.model.to("cpu")
        torch.cuda.empty_cache()
        print('--------------offloading encoder to gpu------------')
        # Encode CLIP
        img = img.to(self.clip.device)
        clip_context = self.clip.visual([img[:, None, :, :]])
        
        self.clip.model.to('cpu')
        torch.cuda.empty_cache()
        # Encode initial frame
        img_resized = torch.nn.functional.interpolate(
            img[None], size=(h, w), mode='bicubic'
        ).transpose(0, 1)
        img_resized= img_resized.to(self.vae.device)
        start_latent = self.vae.encode([img_resized])[0]
        
        # Initialize frame packing
        sampler = FramePackingSampler(self.model, self.vae, latent_window_size, self.vae_stride)
        
        # Calculate sections
        # num_frames_per_window = latent_window_size * 4 - 3
        # total_latent_sections = math.ceil((total_frames - num_frames_per_window) / ((latent_window_size - 1) * 4)) + 1
        frames_per_section = 4 # Each latent frame represents 4 video frames
        total_latent_sections = math.ceil(total_frames / frames_per_section)
        
        logging.info(f"Generating {total_frames} frames in {total_latent_sections} sections")
        
        # Initialize history
        history_latents = torch.zeros(
            size=(1, 16, 1 + 2 + 16, lat_h, lat_w),
            dtype=torch.float32,
            device='cpu'  # Keep on CPU to save GPU memory
        )
        
        # Padding sequence
        if total_latent_sections > 4:
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]
        else:
            latent_paddings = list(reversed(range(total_latent_sections)))
        
        all_generated_latents = []
        total_generated_frames = 0
        
        # Process sections
        for section_idx, latent_padding in enumerate(latent_paddings):
            is_last_section = latent_padding == 0
            latent_padding_size = latent_padding * latent_window_size
            
            logging.info(f'Section {section_idx + 1}/{len(latent_paddings)}: padding={latent_padding_size}')
            
            # Pack latents
            if section_idx == 0:
                current_start = start_latent
            else:
                # Use last frame from previous generation
                current_start = all_generated_latents[-1][:, :, -1:, :, :]
            
            packed_latents, components = sampler.pack_latents(
                current_start,
                history_latents.to(device),
                latent_padding_size
            )
            
            # Create mask
            mask = torch.zeros_like(packed_latents)
            mask_start = 1 + latent_padding_size
            mask_end = mask_start + latent_window_size
            mask[:, :, mask_start:mask_end, :, :] = 1
            
            # Generate segment
            generated = self.generate_segment_with_frame_packing(
                packed_latents=packed_latents,
                mask=mask,
                context=context,
                context_null=context_null,
                clip_context=clip_context,
                latent_window_size=latent_window_size,
                mask_start=mask_start,
                mask_end=mask_end,
                lat_h=lat_h,
                lat_w=lat_w,
                shift=shift,
                sample_solver=sample_solver,
                sampling_steps=sampling_steps,
                guide_scale=guide_scale,
                seed_g=seed_g,
                device_id=device.index,
                offload_model=offload_model,
                progress_callback=lambda info: logging.info(
                    f"Section {section_idx+1}/{len(latent_paddings)}, "
                    f"Step {info['step']}/{info['total_steps']}"
                )
            )
            print(f"Generated shape: {generated.shape}")
            if generated.dim() != 5:
                print(f"Fixing dimensions from {generated.shape}")
                # Remove extra dimensions
                while generated.dim() > 5:
                    generated = generated.squeeze()
                # Add missing dimensions  
                while generated.dim() < 5:
                    generated = generated.unsqueeze(0)
                print(f"Fixed to: {generated.shape}")
            # Store generated latents
            all_generated_latents.append(generated)
            
            # Update history from the generated output
            full_generated = packed_latents.clone()
            full_generated[:, :, mask_start:mask_end, :, :] = generated
            
            _, unpacked = sampler.unpack_latents(full_generated, None, latent_padding_size)
            
            # Update history_latents on CPU
            history_latents[:, :, 0:1, :, :] = generated[:, :, -1:, :, :].cpu()
            if 'clean_2x' in unpacked:
                history_latents[:, :, 1:3, :, :] = unpacked['clean_2x'].cpu()
            if 'clean_4x' in unpacked:
                history_latents[:, :, 3:19, :, :] = unpacked['clean_4x'].cpu()
            
            total_generated_frames += generated.shape[2] * 4 - 3
            
            # Cleanup
            del packed_latents, mask, generated
            torch.cuda.empty_cache()
        
        # Concatenate all latents
        logging.info("Concatenating all generated latents...")
        final_latents = torch.cat(all_generated_latents, dim=2)
        print(f"Concatenated latents...{final_latents.shape}" )
        
        # Trim to exact frame count
        target_latent_frames = (total_frames - 1) // 4 + 1
        if final_latents.shape[2] > target_latent_frames:
            final_latents = final_latents[:, :, :target_latent_frames]
            
        logging.info("Clearing GPU memory before VAE decoding...")
        for i in range(4):  # Your 4 GPUs
            with torch.cuda.device(f"cuda:{i}"):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        # Wait for all operations to complete  
        torch.cuda.synchronize()
        # Decode video
        logging.info("Decoding video...")
       # Move VAE to appropriate device if needed
        vae_device = getattr(self.vae, 'device', 'cuda:3')
        final_latents = final_latents.to(vae_device)

        # Ensure VAE and its model are on the right device
        if hasattr(self.vae, 'model'):
            self.vae.model = self.vae.model.to(vae_device)
            # Also ensure the scale tensors are on the right device
            if hasattr(self.vae, 'scale') and isinstance(self.vae.scale, list):
                self.vae.scale = [s.to(vae_device) if torch.is_tensor(s) else s for s in self.vae.scale]

        # CRITICAL FIX: The WanVAE.decode() method expects a LIST of tensors, not a single tensor
        # and it does NOT take a scale parameter - it uses self.scale internally
        with torch.cuda.device(vae_device):
            # Convert single tensor to list format expected by decode method
            if final_latents.shape[0] == 1:
                final_latents = final_latents.squeeze(0)
                print(f"Squeezed for VAE: {final_latents.shape}")
            videos = self.vae.decode([final_latents])

        # Trim to exact frame count
        if videos and len(videos) > 0 and videos[0].shape[1] > total_frames:
            videos = [v[:, :total_frames] for v in videos]

        return videos[0] if videos else None

    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, 'models') and self.models:
            for model in self.models.values():
                del model
        if hasattr(self, 'gpu_manager'):
            for i in range(torch.cuda.device_count()):
                try:
                    self.gpu_manager.release_gpu_resources(i)
                except:
                    pass