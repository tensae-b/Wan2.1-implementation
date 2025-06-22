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
    """Enhanced wrapper to distribute model layers across multiple GPUs with recommended strategy"""
    
    def __init__(self, model, devices, vae=None, strategy='auto'):
        super().__init__()
        self.devices = devices
        self.num_devices = len(devices)
        self.vae = vae
        self.strategy = strategy
        
        # Store model config and attributes
        self.config = model.config if hasattr(model, 'config') else None
        
        # Get all blocks
        if hasattr(model, 'blocks'):
            self.blocks = model.blocks
            self.num_blocks = len(self.blocks)
        else:
            raise ValueError("Model must have 'blocks' attribute")
        
        # Apply recommended distribution strategy based on GPU count
        if strategy == 'auto':
            if self.num_devices == 2:
                self._setup_2gpu_strategy(model)
            elif self.num_devices == 3:
                self._setup_3gpu_strategy(model)
            elif self.num_devices == 4:
                self._setup_4gpu_strategy(model)
            else:
                self._setup_default_strategy(model)
        else:
            self._setup_default_strategy(model)
        
        logging.info(f"Model parallel setup complete with {self.num_devices} GPUs")
        self._log_gpu_distribution()
    
    def _setup_2gpu_strategy(self, model):
        """2 GPU: GPU0 (prompt encoder, patch embed, 1st half blocks), GPU1 (2nd half blocks, VAE, frame gen)"""
        mid_point = self.num_blocks // 2
        
        # GPU 0: First half of processing
        self.device_blocks = [
            list(range(0, mid_point)),      # GPU 0
            list(range(mid_point, self.num_blocks))  # GPU 1
        ]
        
        # Distribute model components
        self._distribute_blocks()
        
        # Move other components
        for name, module in model.named_children():
            if name == 'blocks':
                continue
            elif name in ['x_embedder', 'patch_embed', 't_embedder']:
                # Initial processing on GPU 0
                setattr(self, name, module.to(self.devices[0]))
            else:
                # Final processing and VAE on GPU 1
                setattr(self, name, module.to(self.devices[1]))
        
        # VAE on GPU 1 for decoding
        if self.vae:
            self.vae_device = self.devices[1]
    
    def _setup_3gpu_strategy(self, model):
        """3 GPU: GPU0 (T5, patch embed, 1/3 blocks), GPU1 (middle blocks), GPU2 (final blocks, VAE, frame)"""
        third = self.num_blocks // 3
        
        self.device_blocks = [
            list(range(0, third)),                    # GPU 0
            list(range(third, 2 * third)),           # GPU 1
            list(range(2 * third, self.num_blocks))  # GPU 2
        ]
        
        self._distribute_blocks()
        
        # Distribute other components
        for name, module in model.named_children():
            if name == 'blocks':
                continue
            elif name in ['x_embedder', 'patch_embed', 't_embedder']:
                # Initial processing on GPU 0
                setattr(self, name, module.to(self.devices[0]))
            else:
                # Final processing on GPU 2
                setattr(self, name, module.to(self.devices[2]))
        
        # VAE on GPU 2
        if self.vae:
            self.vae_device = self.devices[2]
    
    def _setup_4gpu_strategy(self, model):
        """4 GPU: GPU0-1 (all blocks), GPU2 (T5, CLIP), GPU3 (VAE, frame decode)"""
        # Concentrate blocks on first 2 GPUs as requested
        half_blocks = self.num_blocks // 2
        
        self.device_blocks = [
            list(range(0, half_blocks)),                   # GPU 0: First half of blocks
            list(range(half_blocks, self.num_blocks)),     # GPU 1: Second half of blocks
            [],                                            # GPU 2: Reserved for encoders
            []                                             # GPU 3: Reserved for VAE/decoding
        ]
        
        self._distribute_blocks()
        
        # Distribute other components
        for name, module in model.named_children():
            if name == 'blocks':
                continue
            elif name in ['x_embedder', 'patch_embed', 't_embedder']:
                # Initial processing on GPU 0
                setattr(self, name, module.to(self.devices[0]))
            else:
                # Final processing on GPU 1
                setattr(self, name, module.to(self.devices[1]))
        
        # Set device assignments for other components
        self.t5_device = self.devices[2]    # GPU 2 for T5
        self.clip_device = self.devices[2]  # GPU 2 for CLIP
        self.vae_device = self.devices[3]   # GPU 3 for VAE
    
    def _setup_default_strategy(self, model):
        """Default strategy: evenly distribute blocks"""
        blocks_per_device = self.num_blocks // self.num_devices
        remainder = self.num_blocks % self.num_devices
        
        self.device_blocks = []
        start_idx = 0
        
        for i in range(self.num_devices):
            end_idx = start_idx + blocks_per_device
            if i < remainder:
                end_idx += 1
            
            device_block_indices = list(range(start_idx, end_idx))
            self.device_blocks.append(device_block_indices)
            start_idx = end_idx
        
        self._distribute_blocks()
        
        # Move other components to first device by default
        for name, module in model.named_children():
            if name != 'blocks':
                setattr(self, name, module.to(self.devices[0]))
        
        if self.vae:
            self.vae_device = self.devices[-1]  # Last GPU for VAE
    
    def _distribute_blocks(self):
        """Move blocks to their assigned devices"""
        for device_idx, block_indices in enumerate(self.device_blocks):
            if block_indices:  # Only process if there are blocks assigned
                for idx in block_indices:
                    self.blocks[idx] = self.blocks[idx].to(self.devices[device_idx])
    
    def _log_gpu_distribution(self):
        """Log how blocks are distributed across GPUs"""
        for i, block_indices in enumerate(self.device_blocks):
            if block_indices:
                logging.info(f"GPU {self.devices[i]}: Blocks {block_indices[0]}-{block_indices[-1]} ({len(block_indices)} blocks)")
            else:
                logging.info(f"GPU {self.devices[i]}: Reserved for other operations")
        
        if hasattr(self, 't5_device'):
            logging.info(f"T5 encoder on: {self.t5_device}")
        if hasattr(self, 'clip_device'):
            logging.info(f"CLIP encoder on: {self.clip_device}")
        if hasattr(self, 'vae_device'):
            logging.info(f"VAE decode on: {self.vae_device}")
    
    def forward(self, x, t, context, clip_fea, seq_len, y):
        """Forward pass with model parallelism"""
        # Initial processing on first device
#         current_device = self.devices[0]
        
#         # Move inputs to first device
#         if isinstance(x, list):
#             x = [xi.to(current_device) for xi in x]
#             x_tensor = x[0]
#         else:
#             x_tensor = x.to(current_device)
        
#         t = t.to(current_device)
        
#         # Process through embedding layers if they exist
#         if hasattr(self, 'x_embedder'):
#             x_tensor = self.x_embedder(x_tensor)
        
#         if hasattr(self, 't_embedder'):
#             t_emb = self.t_embedder(t)
#         else:
#             t_emb = t
            
#         batch_size = x_tensor.shape[0]
#         grid_sizes = torch.zeros(batch_size, 3, device=x_tensor.device, dtype=torch.int64)
#         C = x_tensor.shape[-1]
#         num_heads = 8 # if accessible here
#         freqs = torch.zeros(1024, C // num_heads // 2, device=x_tensor.device, dtype=torch.float32)
  
#         context_lens = torch.full(
#     (batch_size,),
#     fill_value=context[0].shape[1],
#     device=x_tensor.device,
#     dtype=torch.int64
# )
#         if hasattr(self, 'x_embedder'):
#                 x_tensor = self.x_embedder(x_tensor) 
        device0 = self.devices[0]
    
        # Move everything to first device initially
        if isinstance(x, list):
            x = [xi.to(device0) for xi in x]
        else:
            x = [x.to(device0)]
        if y is not None:
            y = [yi.to(device0) for yi in y]
        t = t.to(device0)
        context = [c.to(device0) for c in context]
        clip_fea = clip_fea.to(device0) if clip_fea is not None else None
        
        # 2. Concatenate x and y if y exists
        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # 3. Patch embedding
        patch_size= (1,2,2)
   
        in_dim=36
        dim= 5120
        freq_dim= 256
        text_dim =4096
        patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size).to(dtype=torch.bfloat16, device=device0)
        
        time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)).to(device0)
        time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6)).to(device0)
        text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)).to(device0)
        
        d = dim // 40
        out_dim= 16
        eps= 1e-06
        self.head = Head(dim, out_dim, patch_size, eps)
        freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1).to(device0)
        
        img_emb = MLPProj(1280, dim).to(device0)
        x = [patch_embedding(u.unsqueeze(0).to(torch.bfloat16)) for u in x]  # List of (1, C, H, W)
        
        # 4. Compute grid sizes and flatten
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])  # (batch, 2)
        x = [u.flatten(2).transpose(1, 2) for u in x]  # List of (1, seq_len, dim)
        
        # 5. Sequence lengths & pad sequences to seq_len
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x
        ])

        # 6. Time embedding
        with torch.cuda.amp.autocast(dtype=torch.float32):
            
            e = time_embedding(sinusoidal_embedding_1d(freq_dim, t).float())
            e0 = time_projection(e).unflatten(1, (6, dim))
        text_len= 512
        # 7. Context embedding with padding
        context_padded = torch.stack([
            torch.cat([u, u.new_zeros(text_len - u.size(0), u.size(1))]) for u in context
        ])
        context_emb = text_embedding(context_padded)

        # 8. Add clip features if available
        if clip_fea is not None:
            context_clip = img_emb(clip_fea)
            context_emb = torch.concat([context_clip, context_emb], dim=1)

        # 9. Prepare keyword args for blocks
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=freqs.to(device0),
            context=context_emb,
            context_lens=None,
        )
        current_device = self.devices[0]
        # Process through blocks distributed across devices
        for device_idx, block_indices in enumerate(self.device_blocks):
            device = self.devices[device_idx]
            logging.info(f"[Forward] Processing blocks on device {device} ({len(block_indices)} blocks)")

            
            # Move tensors to current device if needed
            if device != current_device:
                x = x.to(device)
                logging.info(f"[Forward] Moved x to device {device}, shape: {x.shape}")
                for k, v in kwargs.items():
                    if torch.is_tensor(v) and v.device != device:
                        kwargs[k] = v.to(device)
                        logging.info(f"[Forward] Moved {k} to device {device}, shape: {kwargs[k].shape}")
                current_device = device
            #     x_tensor = x_tensor.to(device)
            #     t_emb = t_emb.to(device)
               
            #     context = [c.to(device) for c in context]
            #     clip_fea = clip_fea.to(device)
            #     y = [yi.to(device) for yi in y] if isinstance(y, list) else y.to(device)
            #     current_device = device
            # e = t_emb if hasattr(self, 't_embedder') else t
            # e = e.float() 
            # model_dim = getattr(self, 'dim', 5120)
            # e = torch.zeros(x_tensor.size(0), 6, model_dim, device=x_tensor.device)
            
            # Process blocks on current device
            for block_idx in block_indices:
                x = self.blocks[block_idx](x, **kwargs)
                logging.info(f"[Forward] After block {block_idx}, x shape: {x.shape}")
                # Call block with appropriate arguments
                # if hasattr(self.blocks[block_idx], 'forward'):
                #     # Adjust call based on your block's expected inputs
                #     x_tensor = self.blocks[block_idx](
                #         x_tensor, 
                #         e,
                #         context=context,
                #         grid_sizes=grid_sizes,
                #         # clip_fea=clip_fea,
                #         seq_lens=seq_len,
                #         freqs= freqs,
                #         context_lens= context_lens,
                #         # y=y
                #     )
        # final_layer = nn.Linear(dim, in_dim).to(dtype=torch.bfloat16, device=current_device)
    
        # Apply final layer to get correct number of channels
        # x = final_layer(x)  # Convert from dim (5120) to in_dim (36) channels
        # logging.info(f"[Forward] After final layer, x shape: {x.shape}")
        e = e.to(x.device)
        x = self.head(x, e)
       
        
        def unpatchify( x, grid_sizes):
            """
            Reconstruct video tensors from patch embeddings.

            Args:
                x (List[Tensor]):
                    List of patchified features, each with shape [L, C_out * prod(patch_size)]
                grid_sizes (Tensor):
                    Original spatial-temporal grid dimensions before patching,
                        shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

            Returns:
                List[Tensor]:
                    Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
            """

            c = 16
            out = []
            for u, v in zip(x, grid_sizes.tolist()):
                u = u[:math.prod(v)].view(*v, *patch_size, c)
                u = torch.einsum('fhwpqrc->cfphqwr', u)
                u = u.reshape(c, *[i * j for i, j in zip(v, patch_size)])
                out.append(u)
            return out
            
            return x

        x = unpatchify(x, grid_sizes)
        return x
        # logging.info(f"[Forward] After head, shape: {x.shape}")
        
        # Final processing
        # if hasattr(self, 'final_layer'):
        #     x_tensor = self.final_layer(x_tensor)
        
        # return [x_tensor] if isinstance(x, list) else x_tensor


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
        self.param_dtype = config.param_dtype
        
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
                test_timesteps = timesteps[:3]
                
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
                max_seq_len=max_seq_len,
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

    def generate(self,
                input_prompt,
                img,
                max_area=640* 360,
                frame_num=16,
                total_duration_seconds=5,
                fps=8,
                overlap_frames=2,
                shift=3.0,
                sample_solver='dpm++',
                sampling_steps=6,
                guide_scale=1.0,
                n_prompt="",
                seed=-1,
                offload_model=False,
                parallel_mode='independent'):
        """Main generation function with enhanced multi-GPU support"""
        
        # Log initial state
        if self.gpu_manager:
            self.gpu_manager.log_memory_status("Initial")
        
        # Prepare image
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)
        
        # Calculate dimensions and segments
        total_frames = fps * total_duration_seconds
        segment_frames = frame_num
        effective_frames_per_segment = segment_frames - overlap_frames
        num_segments = math.ceil((total_frames - overlap_frames) / effective_frames_per_segment)
        
        logging.info(f"Generating {total_duration_seconds}s video at {fps} fps")
        logging.info(f"Total frames: {total_frames}, Segments: {num_segments}")
        logging.info(f"Parallel strategy: {self.parallel_strategy}")
        
        # Setup dimensions
        h, w = img.shape[1:]
        aspect_ratio = h / w
        
        # Calculate latent dimensions
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1]
        )
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2]
        )
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]
        
        # Calculate max sequence length
        max_seq_len = ((segment_frames - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2]
        )
        self.sp_size = 16
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size
        
        # Setup seed
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        
        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        
        # Encode text with proper device management
        logging.info("Encoding text prompts...")
        if not self.t5_cpu:
            # Determine T5 device
            if self.model_parallel and hasattr(self.model, 't5_device'):
                t5_device = self.model.t5_device
            elif self.model_parallel and hasattr(self.model, 'devices'):
                t5_device = self.model.devices[0]
            else:
                t5_device = self.device
            
            # Only move T5 if it's not already on the target device
            current_device = next(self.text_encoder.model.parameters()).device
            if current_device != t5_device:
                logging.info(f"Moving T5 from {current_device} to {t5_device}")
                self.text_encoder.model.to(t5_device)
            
            context = self.text_encoder([input_prompt], t5_device)
            context_null = self.text_encoder([n_prompt], t5_device)
            
            # Move to main device if different
            if t5_device != self.device:
                context = [t.to(self.device) for t in context]
                context_null = [t.to(self.device) for t in context_null]
            
            if offload_model:
                logging.info("Offloading T5 back to CPU")
                self.text_encoder.model.cpu()
                torch.cuda.empty_cache()
        else:
            # T5 on CPU
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]
        
        # Encode CLIP features
        logging.info("Encoding CLIP features...")
        
        # Determine CLIP device
        clip_device = self.device
        if self.model_parallel and hasattr(self.model, 'clip_device'):
            clip_device = self.model.clip_device
        
        # Check if CLIP needs to be moved
        if hasattr(self.clip, 'model'):
            current_clip_device = next(self.clip.model.parameters()).device
            if current_clip_device != clip_device:
                logging.info(f"Moving CLIP from {current_clip_device} to {clip_device}")
                self.clip.model.to(clip_device)
        
        # Encode on the appropriate device
        img_for_clip = img.to(clip_device) if img.device != clip_device else img
        clip_context = self.clip.visual([img_for_clip[:, None, :, :]])
        
        # Move back to main device if needed
        if clip_device != self.device:
            clip_context = clip_context.to(self.device)
        
        if offload_model and hasattr(self.clip, 'model'):
            self.clip.model.cpu()
            torch.cuda.empty_cache()
        
        # Log memory before generation
        if self.gpu_manager:
            self.gpu_manager.log_memory_status("After encoding")
        
        # Generate segments
        logging.info(f"Starting generation with {num_segments} segments...")
        
        if self.multi_gpu and parallel_mode == 'independent' and num_segments > 1:
            all_latents = self._generate_parallel_with_resource_management(
                num_segments, segment_frames, lat_h, lat_w, h, w,
                context, context_null, clip_context, max_seq_len,
                shift, sample_solver, sampling_steps, guide_scale,
                seed_g, img, offload_model, overlap_frames
            )
        else:
            all_latents = self._generate_sequential(
                num_segments, segment_frames, lat_h, lat_w, h, w,
                context, context_null, clip_context, max_seq_len,
                shift, sample_solver, sampling_steps, guide_scale,
                seed_g, img, offload_model, overlap_frames
            )
        
        # Concatenate all latents
        logging.info("Concatenating segments...")
        final_latent = torch.cat(all_latents, dim=1)
        
        # Trim to exact frame count
        target_latent_frames = (total_frames - 1) // 4 + 1
        if final_latent.shape[1] > target_latent_frames:
            final_latent = final_latent[:, :target_latent_frames]
        
        # Decode with proper device management
        videos = None
        if self.rank == 0:
            logging.info("Decoding video...")
            
            # Clear memory before decoding
            torch.cuda.empty_cache()
            
            # Determine VAE device
            vae_device = self.device
            if self.model_parallel and hasattr(self.model, 'vae_device'):
                vae_device = self.model.vae_device
                logging.info(f"Decoding on {vae_device}")
            
            # Move latent to VAE device if needed
            if vae_device != self.device:
                final_latent = final_latent.to(vae_device)
            
            # Ensure VAE is on the correct device
            if hasattr(self.vae, 'device') and self.vae.device != vae_device:
                self.vae.device = vae_device
            if hasattr(self.vae, 'model'):
                self.vae.model = self.vae.model.to(vae_device)
            
            # Decode
            try:
                videos = self.vae.decode([final_latent])
                videos = [v[:, :total_frames] for v in videos]
                
                # Move result back to main device if needed
                if vae_device != self.device:
                    videos = [v.to(self.device) for v in videos]
                    
            except torch.cuda.OutOfMemoryError:
                logging.warning("OOM during decode, trying chunked decode...")
                # Try chunked decoding
                chunk_size = 8  # Decode 8 frames at a time
                decoded_chunks = []
                
                for i in range(0, final_latent.shape[1], chunk_size):
                    chunk = final_latent[:, i:i+chunk_size]
                    decoded = self.vae.decode([chunk])[0]
                    decoded_chunks.append(decoded.cpu())  # Move to CPU to save GPU memory
                    torch.cuda.empty_cache()
                
                # Concatenate on CPU then move to GPU
                videos = [torch.cat(decoded_chunks, dim=1).to(self.device)]
                videos = [v[:, :total_frames] for v in videos]
        
        # Final cleanup
        if offload_model:
            gc.collect()
            torch.cuda.empty_cache()
        
        # Synchronize if distributed
        if dist.is_initialized():
            dist.barrier()
        
        # Log final memory status
        if self.gpu_manager and self.rank == 0:
            self.gpu_manager.log_memory_status("Final")
        
        return videos[0] if videos and self.rank == 0 else None

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