import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial
from wan.utils.utils import cache_image, cache_video, str2bool
from torchvision.transforms.functional import to_pil_image
import imageio
from PIL import Image

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .modules.clip import CLIPModel
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .modules.model import sinusoidal_embedding_1d
from .utils.vace_processor import VaceVideoProcessor
from .modules.vace_model import VaceWanModel
import torchvision
import imageio
import torch
import os.path as osp
import numpy as np
from torchvision.transforms.functional import to_pil_image


class WanFramepack:
    
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
        quantized_model_dir=None,
    ):
        
        # Determine if using quantized model
        self.use_quantized = quantized_model_dir is not None
        
        block_num = 3
        
        # Modified block_distributed_forward to use GPUs 1,2,3
        def block_distributed_forward(self, x, t=None, context=None, seq_len=None, clip_fea=None, y=None, **other_kwargs):
            num_gpus = 3  # Using 3 GPUs
            gpu_devices = [0, 1, 2]  # Skip GPU 0
            total_blocks = len(self.blocks)
            
            blocks_to_process = min(block_num, total_blocks)
            # Divide blocks evenly among 3 GPUs
            blocks_per_gpu = blocks_to_process // num_gpus
            remainder = blocks_to_process % num_gpus
            
            # Start processing on GPU 1
            device = self.patch_embedding.weight.device
            torch.cuda.empty_cache()
            
            if self.freqs.device != device:
                self.freqs = self.freqs.to(device)

            if self.model_type != 'vace' and y is not None:
                x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

            # Use float16 for quantized models
            compute_dtype = torch.float16 

            # Process embeddings
            with amp.autocast(dtype=compute_dtype):
                x_embedded = []
                for i, x_chunk in enumerate(x):
                    x_emb = self.patch_embedding(x_chunk.unsqueeze(0))
                    x_embedded.append(x_emb)
                    if len(x) > 1:
                        del x_chunk
                        torch.cuda.empty_cache()
           
                x = x_embedded
                grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
                x = [u.flatten(2).transpose(1, 2) for u in x]
                seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
                print(seq_lens.max(),'seq len')
                assert seq_lens.max() <= seq_len
                x = torch.cat([
                    torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
                    for u in x
                ])

            del x_embedded
            torch.cuda.empty_cache()

            # Time embeddings
            with amp.autocast(dtype=torch.float32):
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, t).float())
                e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            
            torch.cuda.empty_cache()

            # Context processing
            with amp.autocast(dtype=compute_dtype):
                context_lens = None
                context_processed = self.text_embedding(
                    torch.stack([
                        torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                        for u in context
                    ]))

                if self.model_type != 'vace' and clip_fea is not None:
                    context_clip = self.img_emb(clip_fea)
                    context_processed = torch.concat([context_clip, context_processed], dim=1)
                    del context_clip

            torch.cuda.empty_cache()

            block_kwargs = dict(
                e=e0,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                freqs=self.freqs,
                context=context_processed,
                context_lens=context_lens)

            # Process blocks on GPUs 1, 2, 3
            for idx, gpu_id in enumerate(gpu_devices):
                target_device = torch.device(f'cuda:{gpu_id}')
                
                # Move data to target GPU
                if isinstance(x, list):
                    x = [tensor.to(target_device) for tensor in x]
                else:
                    x = x.to(target_device)
                
                local_kwargs = {}
                for key, value in block_kwargs.items():
                    if isinstance(value, torch.Tensor):
                        local_kwargs[key] = value.to(target_device)
                    elif isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
                        local_kwargs[key] = [v.to(target_device) for v in value]
                    else:
                        local_kwargs[key] = value
                
                # Calculate block range for this GPU
                start_block = idx * blocks_per_gpu + min(idx, remainder)
                end_block = start_block + blocks_per_gpu + (1 if idx < remainder else 0)
                
                print(f"[block_distributed_forward] Processing blocks {start_block} to {end_block-1} on cuda:{gpu_id}")
                
                # Process blocks assigned to this GPU
                for block_idx in range(start_block, end_block):
                    try:
                        with amp.autocast(dtype=compute_dtype):
                            x = self.blocks[block_idx](x, **local_kwargs)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        gc.collect()
                        print(f"OOM on block {block_idx}, retrying after cleanup...")
                        with amp.autocast(dtype=compute_dtype):
                            x = self.blocks[block_idx](x, **local_kwargs)
                    
                    # Clear cache periodically
                    if block_idx % 5 == 0:
                        torch.cuda.empty_cache()
                
                del local_kwargs
            
            # Final processing on GPU 3
            final_device = torch.device('cuda:2')
            if isinstance(x, list):
                x = [tensor.to(final_device) for tensor in x]
            else:
                x = x.to(final_device)
            
            if hasattr(self, 'head'):
                e0_final = e.to(final_device)
                with amp.autocast(dtype=compute_dtype):
                    x = self.head(x, e0_final)
                del e0_final
            
            if hasattr(self, 'unpatchify'):
                grid_sizes_final = grid_sizes.to(final_device)
                x = self.unpatchify(x, grid_sizes_final)
                result = [u.float() for u in x] if isinstance(x, list) else [x.float()]
                del grid_sizes_final
                return result
            
            return x

        # Initialize on GPU 1 instead of GPU 0
        self.device = torch.device(f"cuda:0")
        self.config = config
        self.rank = rank
        self.use_usp = use_usp
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = torch.float16 if self.use_quantized else config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        
        # Load encoders
        print(f"Loading T5 encoder from {checkpoint_dir}")
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=torch.float16 if self.use_quantized else config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        
        # VAE on GPU 3
        print(f"Loading VAE from {checkpoint_dir}")
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=torch.device('cuda:2'))

        # CLIP on GPU 1
        print(f"Loading CLIP from {checkpoint_dir}")
        self.clip = CLIPModel(
            dtype=torch.float16 if self.use_quantized else config.clip_dtype,
            device=torch.device('cuda:0'),
            checkpoint_path=os.path.join(checkpoint_dir, config.clip_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.clip_tokenizer))

        # Load model (quantized or regular)
        if self.use_quantized:
            logging.info(f"Loading quantized VaceWanModel directly from {quantized_model_dir}")
            self.model = self._load_quantized_model_direct(quantized_model_dir)
        self._print_model_info()
        self.model.eval().requires_grad_(False)
        
        self.sp_size = 1

        # Distribute model blocks evenly across GPUs 1, 2, 3
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            if device_id == 0:  # Only setup distribution once
                num_gpus = 3
                gpu_devices = [0,1, 2]
                total_blocks = len(self.model.blocks)
                blocks_to_process = min(block_num, total_blocks)
                
                # Calculate even distribution
                blocks_per_gpu = blocks_to_process // num_gpus
                remainder = blocks_to_process % num_gpus
                
                print(f"Distributing {blocks_to_process} blocks across GPUs {gpu_devices}")
                
                # Distribute blocks
                for idx, gpu_id in enumerate(gpu_devices):
                    start_block = idx * blocks_per_gpu + min(idx, remainder)
                    end_block = start_block + blocks_per_gpu + (1 if idx < remainder else 0)
                    
                    target_device = torch.device(f'cuda:{gpu_id}')
                    for block_idx in range(start_block, end_block):
                        self.model.blocks[block_idx].to(target_device)
                    
                    print(f"GPU {gpu_id}: blocks {start_block}-{end_block-1} ({end_block-start_block} blocks)")
                
                # Replace forward method
                self.model.forward = types.MethodType(block_distributed_forward, self.model)
                
                # Move model components
                # Embeddings and initial layers on GPU 1
                if hasattr(self.model, 'patch_embedding'):
                    self.model.patch_embedding.to(torch.device('cuda:0'))
                if hasattr(self.model, 'time_embedding'):
                    self.model.time_embedding.to(torch.device('cuda:0'))
                if hasattr(self.model, 'time_projection'):
                    self.model.time_projection.to(torch.device('cuda:0'))
                if hasattr(self.model, 'text_embedding'):
                    self.model.text_embedding.to(torch.device('cuda:0'))
                if hasattr(self.model, 'img_emb'):
                    self.model.img_emb.to(torch.device('cuda:0'))
                
                # Head on GPU 3 (with VAE)
                if hasattr(self.model, 'head'):
                    self.model.head.to(torch.device('cuda:2'))
                    
            else:
                if not init_on_cpu:
                    self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt
    
    def _print_model_info(self):
        """Print information about the loaded model"""
        total_params = sum(p.numel() for p in self.model.parameters())
        
        # Check parameter dtypes
        dtypes = {}
        for name, param in self.model.named_parameters():
            dtype = str(param.dtype)
            if dtype not in dtypes:
                dtypes[dtype] = 0
            dtypes[dtype] += param.numel()
        
        print(f"Model info:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Parameter distribution:")
        for dtype, count in dtypes.items():
            print(f"    {dtype}: {count:,} ({count/total_params*100:.1f}%)")
   
        
    def offload_model_to_cpu(self):
        """Offload entire model to CPU to free GPU memory"""
        print("Offloading model to CPU...")
        
        # Move all model components to CPU
        if hasattr(self.model, 'patch_embedding'):
            self.model.patch_embedding.cpu()
        if hasattr(self.model, 'time_embedding'):
            self.model.time_embedding.cpu()
        if hasattr(self.model, 'time_projection'):
            self.model.time_projection.cpu()
        if hasattr(self.model, 'text_embedding'):
            self.model.text_embedding.cpu()
        if hasattr(self.model, 'img_emb'):
            self.model.img_emb.cpu()
        if hasattr(self.model, 'head'):
            self.model.head.cpu()
        if hasattr(self.model, 'unpatchify'):
            self.model.head.to('cpu')
        
        # Move all transformer blocks to CPU
        for block in self.model.blocks:
            block.cpu()
        
        # Move other model components
        if hasattr(self.model, 'freqs'):
            self.model.freqs = self.model.freqs.cpu()
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        print("Model offloaded to CPU")
    
    def select_context_frames(self, all_generated_frames, window_size=12):
        """Select frames strategically like FramePack"""
        if len(all_generated_frames) <= window_size:
            return all_generated_frames
        
        total_frames = len(all_generated_frames)
        selected_indices = []
        print('total frames', total_frames)
        
        # 40% most recent frames
        recent_count = int(window_size * 0.4)
        selected_indices.extend(range(total_frames - recent_count, total_frames))
        
        # 40% exponentially spaced historical frames
        historical_count = int(window_size * 0.4)
        for i in range(historical_count):
            t = i / max(1, historical_count - 1)
            idx = int((total_frames - recent_count) * (1 - t**2))
            selected_indices.append(idx)
        
        # 20% key frames at regular intervals
        remaining = window_size - len(set(selected_indices))
        if remaining > 0 and total_frames > 0:
            interval = max(1, total_frames // remaining)
            for i in range(remaining):
                idx = min(i * interval, total_frames - 1)
                selected_indices.append(idx)
        
        # Remove duplicates and sort
        selected_indices = sorted(set(selected_indices))[:window_size]
        
        # Extract selected frames
        selected_frames = [all_generated_frames[i] for i in selected_indices]
        print('selected_frames', selected_frames[0].shape)
        return selected_frames

    def soft_blend_frames(self, new_frames, existing_frames, overlap_size=3):
        """Blend overlapping frames instead of hard cut"""
        if not existing_frames or overlap_size == 0:
            return new_frames
        
        device = new_frames[0].device if isinstance(new_frames[0], torch.Tensor) else 'cuda:0'
        
        # Convert to tensors if needed
        if not isinstance(new_frames[0], torch.Tensor):
            new_frames = [torch.tensor(f, device=device) for f in new_frames]
        if not isinstance(existing_frames[0], torch.Tensor):
            existing_frames = [torch.tensor(f, device=device) for f in existing_frames]
        
        # Create blending weights
        blend_weights = torch.linspace(0, 1, overlap_size, device=device)
        
        # Get overlap regions
        overlap_existing = existing_frames[-overlap_size:]
        overlap_new = new_frames[:overlap_size]
        
        # Blend
        blended = []
        for i in range(overlap_size):
            weight = blend_weights[i]
            blended_frame = overlap_existing[i] * (1 - weight) + overlap_new[i] * weight
            blended.append(blended_frame)
        
        # Return: existing[:-overlap] + blended + new[overlap:]
        result = existing_frames[:-overlap_size] + blended + new_frames[overlap_size:]
        
        return result

    def initialize_noise_with_momentum(self, all_frames, shape, strength=0.3, device='cuda:0'):
        """Initialize noise with temporal momentum"""
        fresh_noise = torch.randn(shape, dtype=torch.float32, device=device)
        
        if len(all_frames) >= 3:
            # Get last 3 frames as tensors
            recent_frames = []
            for f in all_frames[-3:]:
                if isinstance(f, torch.Tensor):
                    recent_frames.append(f.to(device))
                else:
                    recent_frames.append(torch.tensor(f, device=device))
            
            recent_frames = torch.cat(recent_frames, dim=1)
            
            # Estimate motion across multiple frames
            motion_1 = recent_frames[:, -1] - recent_frames[:, -2]
            motion_2 = recent_frames[:, -2] - recent_frames[:, -3]
            
            # Average motion with decay
            avg_motion = (motion_1 + 0.5 * motion_2) / 1.5
            
            # Apply motion bias to noise (gradually decreasing)
            for i in range(min(9, shape[1])):
                alpha = (9 - i) / 9 * strength
                fresh_noise[:, i] += avg_motion.mean(dim=[1, 2], keepdim=True) * alpha
        
        return fresh_noise

    def get_adaptive_params(self, padding_idx, total_sections):
        """Adjust parameters based on position in backward generation"""
        # More overlap for sections generated first (video end)
        if padding_idx < total_sections // 3:
            overlap = 6  # More overlap at the "end"
            cfg_scale = 6.5
        elif padding_idx < 2 * total_sections // 3:
            overlap = 3  # Medium overlap
            cfg_scale = 6.5 * 0.9  # Slightly reduce
        else:
            overlap = 2  # Less overlap at the "beginning"
            cfg_scale = 6.5 * 0.95
        
        return overlap, cfg_scale
    def __init__(
        self,
        config,
        checkpoint_dir,
        quantized_model_dir=None,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.
        Now supports loading quantized model directly.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.use_quantized = quantized_model_dir is not None

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        # Load model - either quantized directly or base model
        if self.use_quantized:
            logging.info(f"Loading quantized VaceWanModel directly from {quantized_model_dir}")
            self.model = self._load_quantized_model_direct(quantized_model_dir)
        else:
            logging.info(f"Creating VaceWanModel from {checkpoint_dir}")
            self.model = VaceWanModel.from_pretrained(checkpoint_dir)
        
        self.model.eval().requires_grad_(False)

        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (
                usp_attn_forward,
                usp_dit_forward,
                usp_dit_forward_vace,
            )
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            for block in self.model.vace_blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.model.forward_vace = types.MethodType(usp_dit_forward_vace,
                                                       self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt

        self.vid_proc = VaceVideoProcessor(
            downsample=tuple(
                [x * y for x, y in zip(config.vae_stride, self.patch_size)]),
            min_area=720 * 1280,
            max_area=720 * 1280,
            min_fps=config.sample_fps,
            max_fps=config.sample_fps,
            zero_start=True,
            seq_len=75600,
            keep_last=True)

    def _load_quantized_model_direct(self, quantized_model_dir):
        print("Load quantized model directly from directory")
        
        try:
            # Method 1: Try loading directly if config.json exists
            config_path = os.path.join(quantized_model_dir, "config.json")
            if os.path.exists(config_path):
                print("Loading quantized model directly from config...")
                model = VaceWanModel.from_pretrained(
                    quantized_model_dir,
                    torch_dtype=torch.float16,
                    
                    ignore_mismatched_sizes=True  # This helps with naming issues
                )
                print("✓ Direct loading successful")
                return model
            
            # Method 2: Create model architecture and load weights manually
            print("Config not found, creating model architecture and loading weights...")
            
            # Create model with default config first
            model = VaceWanModel(self.config)  # Use your config
            
            # Load quantized weights
            from safetensors.torch import load_file
            
            safetensor_files = [f for f in os.listdir(quantized_model_dir) 
                              if f.endswith('.safetensors')]
            
            if not safetensor_files:
                raise ValueError(f"No .safetensors file found in {quantized_model_dir}")
            
            safetensor_path = os.path.join(quantized_model_dir, safetensor_files[0])
            print(f"Loading weights from {safetensor_path}")
            
            # Load and map weights
            quantized_state_dict = load_file(safetensor_path, device='cpu')
            mapped_weights = self._map_quantized_weights(quantized_state_dict)
            
            # Load weights into model
            missing_keys, unexpected_keys = model.load_state_dict(mapped_weights, strict=False)
            
            print(f"Weight loading results:")
            print(f"  Loaded: {len(mapped_weights)} weights")
            print(f"  Missing: {len(missing_keys)} keys")
            print(f"  Unexpected: {len(unexpected_keys)} keys")
            
            # Cleanup
            del quantized_state_dict
            del mapped_weights
            
            print("✓ Manual weight loading successful")
            return model
            
        except Exception as e:
            print(f"✗ Error loading quantized model: {e}")
            print("This might be due to architecture differences or missing files")
            raise
    def _map_quantized_weights(self, quantized_state_dict):
        """Map quantized weights to expected model architecture"""
        
        mapped_weights = {}
        
        for key, tensor in quantized_state_dict.items():
            # Skip metadata
            if key == 'model_type.VACE_14B':
                continue
            
            # Map weight names - remove 'vace_' prefix
            new_key = key
            if key.startswith('vace_'):
                new_key = key[5:]  # Remove 'vace_' prefix
            
            mapped_weights[new_key] = tensor
        
        return mapped_weights

    def _create_model_from_scratch(self, quantized_model_dir):
        """Create model from scratch using quantized weights"""
        
        # This is a fallback method if the above doesn't work
        print("Creating model from scratch...")
        
        from safetensors.torch import load_file
        
        # Load quantized weights to analyze structure
        safetensor_files = [f for f in os.listdir(quantized_model_dir) 
                          if f.endswith('.safetensors')]
        
        if not safetensor_files:
            raise ValueError(f"No .safetensors file found in {quantized_model_dir}")
        
        safetensor_path = os.path.join(quantized_model_dir, safetensor_files[0])
        quantized_state_dict = load_file(safetensor_path, device='cpu')
        
        # Analyze the structure to understand the model
        print("Analyzing quantized model structure...")
        
        # Count layers
        vace_block_count = 0
        block_count = 0
        
        for key in quantized_state_dict.keys():
            if key.startswith('vace_blocks.'):
                layer_num = int(key.split('.')[1])
                vace_block_count = max(vace_block_count, layer_num + 1)
            elif key.startswith('blocks.'):
                layer_num = int(key.split('.')[1])
                block_count = max(block_count, layer_num + 1)
        
        print(f"Found {vace_block_count} vace_blocks and {block_count} regular blocks")
        
        # Update config based on discovered structure
        if hasattr(self.config, 'num_layers'):
            self.config.num_layers = max(vace_block_count, block_count)
        
        # Create model with updated config
        model = VaceWanModel(self.config)
        
        # Load weights
        mapped_weights = self._map_quantized_weights(quantized_state_dict)
        missing_keys, unexpected_keys = model.load_state_dict(mapped_weights, strict=False)
        
        print(f"Model created from scratch:")
        print(f"  Loaded: {len(mapped_weights)} weights")
        print(f"  Missing: {len(missing_keys)} keys")
        
        del quantized_state_dict
        del mapped_weights
        
        return model