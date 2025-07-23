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

import torchvision
import imageio
import torch
import os.path as osp
import numpy as np
from torchvision.transforms.functional import to_pil_image

class WanI2V:
    
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
            device=torch.device('cuda:1'),
            checkpoint_path=os.path.join(checkpoint_dir, config.clip_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.clip_tokenizer))

        # Load model (quantized or regular)
        if self.use_quantized:
            print(f"Loading INT8 quantized model from {quantized_model_dir}")
            logging.info(f"Creating quantized WanModel from {quantized_model_dir}")
            
            if os.path.exists(os.path.join(quantized_model_dir, "config.json")):
                self.model = WanModel.from_pretrained(
                    quantized_model_dir,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
            else:
                print("Loading model architecture from base checkpoint...")
                self.model = WanModel.from_pretrained(
                    checkpoint_dir,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
                
                print("Loading quantized weights from safetensors...")
                from safetensors.torch import load_file
                
                safetensor_files = [f for f in os.listdir(quantized_model_dir) if f.endswith('.safetensors')]
                if not safetensor_files:
                    raise ValueError(f"No .safetensors file found in {quantized_model_dir}")
                
                safetensor_path = os.path.join(quantized_model_dir, safetensor_files[0])
                print(f"Loading from {safetensor_path}")
                
                quantized_state_dict = load_file(safetensor_path, device='cpu')
                self.model.load_state_dict(quantized_state_dict, strict=True)
                del quantized_state_dict
            
            print("Quantized model loaded successfully")
            self._print_model_info()
            
        else:
            print(f"Loading regular model from {checkpoint_dir}")
            logging.info(f"Creating WanModel from {checkpoint_dir}")
            self.model = WanModel.from_pretrained(checkpoint_dir)
        
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

    def generate(self,
                 input_prompt,
                 img,
                 max_area=120 * 208,
                 frame_num=5,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=25,
                 guide_scale=6.5,
                 n_prompt="blurry, unclear",
                 seed=-1,
                 offload_model=True,
                 total_frames=29):
        
        # Image preprocessing
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

        # Calculate dimensions
        F = frame_num
        h, w = img.shape[1:]
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]

        max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        # Seed setup
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        # Create mask
        msk = torch.ones(1, 81, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        print('Text encoding...')
        # Text encoding
        text_device = torch.device('cuda:0')
        
        if not self.t5_cpu:
            self.text_encoder.model.to(text_device)
            context = self.text_encoder([input_prompt], text_device)
            context_null = self.text_encoder([n_prompt], text_device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cuda:0'))
            context_null = self.text_encoder([n_prompt], torch.device('cuda:0'))
            context = [t.to(text_device) for t in context]
            context_null = [t.to(text_device) for t in context_null]
        
        self.text_encoder.model.to('cpu')
        print('Text encoding done')
        torch.cuda.empty_cache()
        gc.collect()

        # CLIP encoding
        torch.cuda.synchronize()
        self.clip.model.to('cuda:2')
        img = img.to('cuda:2')
        clip_context = self.clip.visual([img[:, None, :, :]])
        self.clip.model.cpu()
        torch.cuda.empty_cache()
        gc.collect()

        # VAE encoding for start frame
        print('VAE encoding...')
        y = self.vae.encode([
            torch.concat([
                torch.nn.functional.interpolate(
                    img[None], size=(h, w), mode='bicubic').transpose(0, 1),
                torch.zeros(3, F - 1, h, w, device='cuda:2')
            ], dim=1).to('cuda:2')
        ])[0]
        
        msk = msk.to('cuda:2')
        ys = torch.concat([msk, y])
        img = img.to("cpu")
        torch.cuda.empty_cache()
        gc.collect()

        # FramePack setup
        latent_window_size = 21
        generation_window_size = 9
        context_window_size = 12
        
        # Calculate sections with FramePack's inverted padding schedule
        total_sections = math.ceil((total_frames-20) / generation_window_size)+1
        
        # FramePack's inverted padding schedule for backward generation
        if total_sections <= 4:
            latent_paddings = list(reversed(range(total_sections)))
        else:
            latent_paddings = [3] + [2] * (total_sections - 3) + [1, 0]
        
        print(f"Generating {total_frames} frames in {total_sections} sections (backward order)")
        print(f"Padding schedule: {latent_paddings}")
        
        # Store the start latent
        start_latent = y
        
        # Storage for generated content
        all_generated_frames = []
        all_generated_sections = []  # Store sections for final reversal
        context_frame_decoded = []
        history_latents = []  # Memory-efficient history
        max_history = 100
        
        # Quality tracking
        quality_tracker = {
            'mean': start_latent.mean().item(),
            'std': start_latent.std().item(),
            'initialized': True
        }
        
        # No sync context
        @contextmanager
        def noop_no_sync():
            yield
        no_sync = getattr(self.model, 'no_sync', noop_no_sync)
        
        # Random generator
        rng = torch.Generator(device=self.device)
        if hasattr(seed_g, 'initial_seed'):
            rng.manual_seed(seed_g.initial_seed())
        else:
            rng.manual_seed(42)

        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
            for padding_idx, latent_padding in enumerate(latent_paddings):
                is_first_backward = (padding_idx == 0)
                is_last_backward = (latent_padding == 0)
                
                print(f'\nBackward section {padding_idx + 1}/{total_sections} (padding={latent_padding})')
                
                # Get adaptive parameters
                overlap_frames, cfg_scale = self.get_adaptive_params(padding_idx, total_sections)
                sampling_steps=40
                
                # Initialize scheduler for this section
                if sample_solver == 'unipc':
                    sample_scheduler = FlowUniPCMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=shift,
                        use_dynamic_shifting=False)
                    sample_scheduler.set_timesteps(
                        sampling_steps, device=self.device, shift=shift)
                    timesteps = sample_scheduler.timesteps
                elif sample_solver == 'dpm++':
                    sample_scheduler = FlowDPMSolverMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=shift,
                        use_dynamic_shifting=False)
                    sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                    timesteps, _ = retrieve_timesteps(
                        sample_scheduler,
                        device=self.device,
                        sigmas=sampling_sigmas)
                else:
                    raise NotImplementedError("Unsupported solver.")
                
                if is_first_backward:
                    # First section in backward generation (end of video)
                    frame_offset = (total_sections - 1) * generation_window_size
                    section_noise = torch.randn(
                        16, 21, lat_h, lat_w,
                        dtype=torch.float32,
                        generator=rng,
                        device=self.device
                    )
                    latent_sequence = start_latent
                    msk_section = msk
                else:
                    # Calculate frame offset for backward generation
                    frame_offset = latent_padding * generation_window_size
                    
                    # Select context frames intelligently
                    if len(context_frame_decoded) > 0 :
                        # Use the last decoded frame as context
                        img_context = TF.to_tensor(context_frame_decoded[-1]).sub_(0.5).div_(0.5).to(self.device)
                        h_context, w_context = img_context.shape[1:]
                        img_context = img_context.to('cuda:2')
                        
                        context_y = self.vae.encode([
                            torch.concat([
                                torch.nn.functional.interpolate(
                                    img_context[None], size=(h_context, w_context), mode='bicubic').transpose(0, 1),
                                torch.zeros(3, 80, h_context, w_context, device='cuda:2')
                            ], dim=1).to('cuda:2')
                        ])[0]
                        
                        latent_sequence = context_y
                        print('context_y',context_y.shape)
                    else:
                        # Fallback to using generated frames
                        selected_frames = self.select_context_frames(all_generated_frames, context_window_size)
                        if selected_frames:
                            latent_sequence = torch.cat(selected_frames, dim=1).to('cuda:2')
                        else:
                            latent_sequence = start_latent
                            
                        print('latent_sequence',latent_sequence.shape)
                        current_len = latent_sequence.shape[1]
                        if current_len < 21:
                            pad_shape = list(latent_sequence.shape)
                            pad_shape[1] = 21 - current_len  # How many to pad
                            padding = torch.zeros(pad_shape, device=latent_sequence.device, dtype=latent_sequence.dtype)
                            latent_sequence = torch.cat([latent_sequence, padding], dim=1)
                            
                        N = 21  # total number of latent tokens (or set dynamically)
                        H, W = lat_h, lat_w
                        num_total_tokens = 21 * 4  # = 84
                        context_tokens = current_len * 4  # each timestep has 4 tokens
                        msk_new = torch.zeros(1, num_total_tokens, lat_h, lat_w, device=self.device)
                        msk_new[:, :context_tokens] = 1  # Mark context tokens
                        msk_new = msk_new.view(1, num_total_tokens // 4, 4, lat_h, lat_w)
                        msk_new = msk_new.transpose(1, 2)[0]
                        msk_section=msk_new
                        print('new msk shape:', msk_new.shape) 
                    
                    # Initialize noise with momentum
                    section_noise = self.initialize_noise_with_momentum(
                        all_generated_frames,
                        (16, 21, lat_h, lat_w),
                        strength=0.3,
                        device=self.device
                    )
                    
                    # Create mask
                    msk_section = msk.to('cuda:2')
                
                # Prepare latent for denoising
                latent = section_noise
                
                print('msk',msk_section.shape)
                print('latent',latent_sequence.shape)
                y_section = torch.cat([msk_section, latent_sequence], dim=0)
                
                print(f'Frame offset: {frame_offset}, Overlap: {overlap_frames}')
                
                # Prepare conditioning arguments
                arg_c = {
                    'context': [context[0]],
                    'clip_fea': clip_context,
                    'seq_len': max_seq_len,
                    'y': [y_section],
                    'frame_offset': frame_offset
                }

                arg_null = {
                    'context': context_null,
                    'clip_fea': clip_context,
                    'seq_len': max_seq_len,
                    'y': [y_section],
                    'frame_offset': frame_offset
                }
                
                if offload_model:
                    torch.cuda.empty_cache()
                
                # Denoising loop
                for step_idx, t in enumerate(tqdm(timesteps, desc=f"Section {padding_idx + 1}")):
                    if step_idx % 5 == 0:
                        torch.cuda.empty_cache()
                    
                    # Denoise with adjusted cfg_scale
                    latent = self._denoise_step_consistent(
                        latent, t, arg_c, arg_null, cfg_scale,
                        sample_scheduler, seed_g, step_idx, len(timesteps)
                    )
                    
                    gc.collect()
                    break
                
                # Extract generated frames
                if is_first_backward:
                    generated_frames = latent[:, 1:, :, :]  # Skip first frame
                else:
                    generated_frames = latent[:, -generation_window_size:, :, :]
                
                # Update quality tracker
                quality_tracker['mean'] = generated_frames.mean().item() * 0.1 + quality_tracker['mean'] * 0.9
                quality_tracker['std'] = generated_frames.std().item() * 0.1 + quality_tracker['std'] * 0.9
                
                # Store frames with quality preservation
                section_frames = []
                for frame_idx in range(generated_frames.shape[1]):
                    frame = generated_frames[:, frame_idx:frame_idx+1, :, :]
                    frame = self._preserve_frame_quality(frame, quality_tracker)
                    section_frames.append(frame)
                
                # Apply soft blending if not first section
                if not is_first_backward and overlap_frames > 0 and all_generated_frames:
                    section_frames = self.soft_blend_frames(
                        section_frames,
                        all_generated_frames[-overlap_frames:],
                        overlap_frames
                    )
                
                # Update storage
                all_generated_frames.extend(section_frames)
                all_generated_sections.append(section_frames)
                
                # Memory management
                history_latents.append(generated_frames.cpu())
                if len(history_latents) > max_history:
                    history_latents.pop(0)
                
                # Cleanup
                del latent, section_noise
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()
                
                print(f"Section {padding_idx + 1} complete. Generated {len(section_frames)} new frames")
                print(f"Total frames generated so far: {len(all_generated_frames)}")
                
                # Periodic saving (only every 5 sections or last)
                if  True:
                    # Decode recent frames
                    recent_frames = all_generated_frames[-min(50, len(all_generated_frames)):]
                    recent_tensor = torch.cat(recent_frames, dim=1).to('cuda:2')
                    
                    videos = self.vae.decode([recent_tensor])
                    videos = videos[0]
                   
                    # Cache video and get context frame
                    filename = f"checkpoint_backward_{padding_idx}.mp4"
                    imagename = f"context_frame_{padding_idx}_"
                    cache_path, last_frame_img = self.cache_video_and_get_last_frame(
                        videos, save_file=filename, fps=12, save_image=imagename
                    )
                    context_frames= self.context_frame_picker(videos)
                    # Store context frame for next section
                    context_frame_decoded = [last_frame_img]
                    
                    del videos, recent_tensor
                    torch.cuda.empty_cache()

        # Final assembly - REVERSE the backward generation
        print("\nAssembling final video (reversing backward generation)...")
        final_latent = self._combine_generated_frames_backward(
            all_generated_sections, start_latent, total_frames, device='cuda:2'
        )
        
        print(f'Final latent shape: {final_latent.shape}')
        
        # Decode the final video
        if self.rank == 0:
            if offload_model:
                self.offload_model_to_cpu()
            videos = self.vae.decode([final_latent])
            print(f'Final video shape: {videos[0].shape}')
            return videos[0]
        
        return None

    def _denoise_step_consistent(self, latent, t, arg_c, arg_null, guide_scale, scheduler, seed_g, step_idx, total_steps):
        """Denoise step with momentum for stability"""
        device = 'cuda:0'
        
        # Momentum for quantized models
        if hasattr(self, '_prev_noise_pred') and self.use_quantized:
            momentum = 0.3
        else:
            momentum = 0.0
            self._prev_noise_pred = None
        
        # Move to processing device
        latent_input = [latent.to(device)]
        timestep = torch.tensor([t]).to(device)
        
        # Conditional prediction
        arg_c_gpu = self._move_args_to_device(arg_c, device)
        noise_pred_cond = self.model(latent_input, t=timestep, **arg_c_gpu)[0]
        noise_pred_cond = noise_pred_cond.to('cpu')
        
        del latent_input, arg_c_gpu
        torch.cuda.empty_cache()
        
        # Unconditional prediction
        latent_input = [latent.to(device)]
        timestep = torch.tensor([t]).to(device)
        
        arg_null_gpu = self._move_args_to_device(arg_null, device)
        noise_pred_uncond = self.model(latent_input, t=timestep, **arg_null_gpu)[0]
        noise_pred_uncond = noise_pred_uncond.to('cpu')
        
        del latent_input, timestep, arg_null_gpu
        torch.cuda.empty_cache()
        
        # Apply classifier-free guidance
        noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)
        
        # Apply momentum if needed
        if self.use_quantized and self._prev_noise_pred is not None:
            noise_pred = momentum * self._prev_noise_pred + (1 - momentum) * noise_pred
        
        self._prev_noise_pred = noise_pred.clone()
        
        # Scheduler step
        latent = latent.to('cpu')
        latent_next = scheduler.step(
            noise_pred.unsqueeze(0),
            t,
            latent.unsqueeze(0),
            return_dict=False,
            generator=seed_g
        )[0].squeeze(0)
        
        return latent_next

    def _combine_generated_frames_backward(self, all_sections, start_latent, total_frames, device):
        """Combine sections generated in backward order into final forward video"""
        # Prepare start frame
        if start_latent.dim() == 5:
            start_frame = start_latent[:, 0, 0:1, :, :].to(device)
        elif start_latent.dim() == 4:
            start_frame = start_latent[:, 0:1, :, :].to(device)
        else:
            start_frame = start_latent.unsqueeze(1).to(device)
        
        # IMPORTANT: Reverse the sections since we generated backward
        all_sections_reversed = list(reversed(all_sections))
        
        # Flatten all frames from reversed sections
        all_frames_forward = [start_frame]
        for section in all_sections_reversed:
            for frame in section:
                if frame.dim() == 3:
                    all_frames_forward.append(frame.unsqueeze(1).to(device))
                else:
                    all_frames_forward.append(frame.to(device))
        
        # Concatenate up to target frame count
        final_latent = torch.cat(all_frames_forward[:total_frames], dim=1)
        
        # Ensure exact frame count
        if final_latent.shape[1] > total_frames:
            final_latent = final_latent[:, :total_frames, :, :]
        elif final_latent.shape[1] < total_frames:
            # Pad with zeros if needed
            padding = torch.zeros(
                16, total_frames - final_latent.shape[1],
                final_latent.shape[2], final_latent.shape[3],
                device=device
            )
            final_latent = torch.cat([final_latent, padding], dim=1)
        
        print(f"Final latent shape after reversal: {final_latent.shape}")
        return final_latent

    def _move_args_to_device(self, args, device):
        """Helper to move arguments to device"""
        args_gpu = {}
        for key, value in args.items():
            if isinstance(value, torch.Tensor):
                args_gpu[key] = value.to(device)
            elif isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
                args_gpu[key] = [v.to(device) for v in value]
            else:
                args_gpu[key] = value
        return args_gpu

    def _preserve_frame_quality(self, frame, quality_tracker):
        """Final quality check for individual frames"""
        # Check for extreme values
        if (frame > 4.0).any() or (frame < -4.0).any():
            frame = torch.clamp(frame, -3.5, 3.5)
        
        # Check distribution
        frame_std = frame.std()
        if frame_std < 0.1 or frame_std > 3.0:
            frame = (frame - frame.mean()) / (frame_std + 1e-8)
            frame = frame * quality_tracker['std'] + quality_tracker['mean']
        
        return frame

    def cache_video_and_get_last_frame(self, tensor, save_file=None, fps=24, save_image=None):
        """Cache video tensor to file and return the last frame as numpy array"""
        video_tensor = tensor.cpu().detach()
        
        # Normalize from [-1, 1] to [0, 1]
        video_tensor = (video_tensor + 1.0) / 2.0
        video_tensor = torch.clamp(video_tensor, 0.0, 1.0)
        
        # Convert to NumPy: [T, H, W, C]
        video_np = video_tensor.permute(1, 2, 3, 0).numpy()
        print('video_np', video_np.shape)
        video_np_uint8 = (video_np * 255).astype(np.uint8)
        
        # Save MP4
        imageio.mimsave(save_file, video_np_uint8, fps=fps)
        
        # Extract last frame
        last_frame_img = video_np_uint8[-1]
        
        if save_image:
            Image.fromarray(last_frame_img).save(f"{save_image}.png")
        
        return save_file, last_frame_img

    def offload_model_to_cpu(self):
        """Offload model to CPU to save GPU memory"""
        self.model.cpu()
        torch.cuda.empty_cache()
        
    def context_frame_picker(self, videos):
        print('videos', videos.shape)