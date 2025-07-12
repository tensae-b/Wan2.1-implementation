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
        
        block_num = 40
        
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

    def generate(self,
                 input_prompt,
                 img,
                 max_area=120 * 208,  # Reduced from 480 * 832
                 frame_num=5,  # Reduced from 4  
                 shift=4.0,  # Reduced from 5.0 for smaller resolution
                 sample_solver='unipc',
                 sampling_steps=20,  # Reduced from 40
                 guide_scale=6.5,
                 n_prompt="blurry, unclear",
                 seed=-1,
                 offload_model=True):
       
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

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

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        

        msk = torch.ones(1, 81, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ],
                           dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        print('text encoding')
        # preprocess
        text_encoder_device = torch.device('cuda:2')
        text_device= torch.device('cuda:0')
        
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
        print('text encoding done')
        torch.cuda.empty_cache()
        gc.collect()
       
        torch.cuda.synchronize()
        self.clip.model.to('cuda:2')
        img = img.to('cuda:2')
        clip_context = self.clip.visual([img[:, None, :, :]])
        self.clip.model.cpu()
        torch.cuda.empty_cache()
        gc.collect()

        torch.cuda.synchronize()
        
       
            
        y = self.vae.encode([
             torch.concat([
                 torch.nn.functional.interpolate(
          img[None], size=(h, w), mode='bicubic').transpose(
              0, 1),
                 torch.zeros(3, F - 1, h, w, device='cuda:2')
             ],
              dim=1).to('cuda:2')
         ])[0]
        
        msk = msk.to(text_encoder_device)
        ys = torch.concat([msk, y])
        print('msk concat', msk.shape)
        print('ys concat', ys.shape)
        img=img.to("cpu")
        torch.cuda.empty_cache()
        gc.collect()

        torch.cuda.synchronize()
        
        
     
        @contextmanager
        def noop_no_sync():
            yield
        no_sync = getattr(self.model, 'no_sync', noop_no_sync)
        sampling_steps = 25
        sample_solver = 'dpm++'
        # Calculate total frames and sections
        latent_window_size = 21
        total_frames = 42
        
        
        # Fixed context window size (e.g., 12 frames of context + 9 frames to generate)
        # This matches the official code's multi-scale context: 1 + 2 + 16 ≈ 12 effective context frames
        context_window_size = 12
        generation_window_size = 21
        total_sections = math.ceil(total_frames / generation_window_size)
        # Ensure it adds up to latent_window_size
        
        
        print(f"Generating {total_frames} frames in {total_sections} sections")
        print(f"Using fixed context window: {context_window_size} frames, generating: {generation_window_size} frames per section")
        
        # Store the start latent
        start_latent = y  # Shape: (16, lat_h, lat_w)
        
        
        all_generated_latents = []
        all_generated_frames = []  # Flattened list of all frames for easy access
        
        device = self.device
        dtype = self.param_dtype
        
        # Random generator
        rng = torch.Generator(device=device)
        if hasattr(seed_g, 'initial_seed'):
            rng.manual_seed(seed_g.initial_seed())
        else:
            rng.manual_seed(42)
        
        
                

                # Initialize scheduler
        quality_tracker = {
            'mean': 0.0,
            'std': 1.0,
            'initialized': False
        }

                
        with amp.autocast(dtype=dtype), torch.no_grad(), no_sync():
            for section_idx in range(total_sections):
                print(f'\nSection {section_idx + 1}/{total_sections}')
                
                # Generate fresh noise for generation window only
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
               
                    
                if section_idx == 0:
                    frame_offset = 0
                    section_noise = torch.randn(
                    16, 21, lat_h, lat_w,
                    dtype=torch.float32,
                    generator=rng,
                    device=device
                )
                    print(section_noise.shape,'section_noise section1, shape')
                    latent_sequence = start_latent
                    print(latent_sequence.shape,'latent_sequence section 1, shape')
                    
                    # Create mask: 1 for context, 0 for generation
                    msk = torch.ones(1, 81, lat_h, lat_w, device=self.device)
                    msk[:, 1:] = 0
                    msk = torch.concat([
                        torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
                    ],
                                    dim=1)
                    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
                    msk_section = msk.transpose(1, 2)[0]
                    print(msk_section.shape,'msk_section, shape')
                    # Initial latent is pure noise for generation window
                    latent = section_noise
                    msk_section=msk_section.to('cuda:2')
                    latent_sequence=latent_sequence.to('cuda:2')
                    quality_tracker['mean'] = start_latent.mean().item()
                    quality_tracker['std'] = start_latent.std().item()
                    quality_tracker['initialized'] = True
                    
                else:
                    frame_offset = 20 + (section_idx - 1) * 9 # Correct offset calculation
    
                    # Key optimization: Sliding window with overlap
                    # We generate 9 new frames but use 3 overlap frames from previous
                    overlap_frames = 3
                    new_frames_to_generate = 9
                    context_window_size = 12
                    
                    # Get context using the optimized strategy
                    context_frames = self._get_optimized_context_window(
                        all_generated_frames, 
                        12, 
                        overlap_frames,
                        lat_h, 
                        lat_w
                    )
                    
                    # Generate noise only for new frames
                    section_noise = torch.randn(
                        16, 21, lat_h, lat_w,
                        dtype=torch.float32,
                        generator=rng,
                        device=device
                    )
                    
                    # Apply temporal coherence to initial noise
                     # CRITICAL FIX 2: Apply noise correlation properly
                    if len(all_generated_frames) >= 3:
                        # Use last 3 frames for motion continuity
                        last_frames = torch.cat(all_generated_frames[-3:], dim=1)
                        motion_delta = last_frames[:, -1] - last_frames[:, -2]
                        motion_delta=motion_delta.to('cuda:0')
                        
                        # Apply motion to noise for continuity
                        for i in range(3):
                            alpha = (3 - i) / 3  # Decreasing influence
                            section_noise[:, -9 + i] += motion_delta * alpha * 0.15
                    
                    # Create latent sequence efficiently
                    zeros_for_generation = torch.zeros(
                        16, new_frames_to_generate, lat_h, lat_w, 
                        device='cuda:2'  # Create directly on target device
                    )
                    
                    # Move context frames directly to target device
                    context_frames = context_frames.to('cuda:2')
                    
                    # Concatenate on target device (avoiding extra memory copy)
                    latent_sequence = torch.cat([context_frames, zeros_for_generation], dim=1)
                    
                    # Create mask more efficiently
                    msk_section = self._create_generation_mask(
                        context_window_size, 
                        new_frames_to_generate, 
                        lat_h, 
                        lat_w, 
                        device='cuda:2'
                    )
                    
                    # Use only generation portion of noise
                    latent = section_noise

                # Shared post-processing for both branches
                y_section = torch.cat([msk_section, latent_sequence], dim=0)
                
                
                print(f'y_section shape: {y_section.shape} (consistent across all sections)')
                print(f'latent shape: {latent.shape} (only generation window)')
                
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
                adaptive_guide_scale = guide_scale * (1.0 - section_idx * 0.1)  # Slightly reduce for later sections
                adaptive_guide_scale = max(adaptive_guide_scale, guide_scale * 0.7)
                # Denoising loop - only denoise the generation window
                for step_idx, t in enumerate(tqdm(timesteps, desc=f"Section {section_idx + 1}")):
                    if step_idx % 5 == 0:
                        torch.cuda.empty_cache()
                    
                    # Important: We need to pad latent to full window size for model input
                    # Model expects shape [16, 21, lat_h, lat_w]
                    latent_padded = latent
                    print(latent_padded.shape,'shape')
                    # Denoise
                    latent_padded = self._denoise_step_consistent(
                        latent_padded, t, arg_c, arg_null, guide_scale,
                        sample_scheduler, seed_g, step_idx, len(timesteps)
                    )
                    
                    # Extract only the generation window
                    latent = latent_padded
                    
                    
                    gc.collect()
                    
                    
                    
                if section_idx == 0:
                    # First section: skip the first frame (context)
                    generated_frames = latent[:, 1:, :, :]  # 20 new frames
                else:
                    # Other sections: only last 9 frames are new
                    generated_frames = latent[:, -9:, :, :]  # 9 new frames
                
                # Update quality tracker
                quality_tracker['mean'] = generated_frames.mean().item() * 0.1 + quality_tracker['mean'] * 0.9
                quality_tracker['std'] = generated_frames.std().item() * 0.1 + quality_tracker['std'] * 0.9
                
                # Store frames with quality check
                for frame_idx in range(generated_frames.shape[1]):
                    frame = generated_frames[:, frame_idx:frame_idx+1, :, :]
                    
                    # Final quality preservation per frame
                    frame = self._preserve_frame_quality(frame, quality_tracker)
                    
                    all_generated_frames.append(frame)
                
                all_generated_latents.append(generated_frames.cpu())
                # Store generated frames
                # for frame_idx in range(latent.shape[1]):
                #     all_generated_frames.append(latent[:, frame_idx:frame_idx+1, :, :])
                
                # all_generated_latents.append(latent.cpu())
                
                # Cleanup
                del latent, section_noise
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()
                
                print(f"Section {section_idx + 1} complete. Generated {generation_window_size} new frames")
                print(f"Total frames generated so far: {len(all_generated_frames)}")
                

        # Combine all generated frames
        final_latent = self._combine_generated_frames(
            all_generated_frames, start_latent, total_frames, device='cuda:2'
        )
        
        # Decode the final video
        if self.rank == 0:
            if offload_model:
                self.offload_model_to_cpu()
            videos = self.vae.decode([final_latent])
            return videos[0]
        
        return None

    def _get_fixed_context_window(self, all_frames, context_size, lat_h, lat_w):
        """
        Optimal context window selection for long video generation.
        Prioritizes temporal coherence and motion continuity.
        """
        device = self.device
        context_size = 12
        
        if not all_frames:
            return torch.zeros(16, context_size, lat_h, lat_w, device=device)
        
        total_available = len(all_frames)
        
        # For video generation, we want:
        # 1. Most recent frames (for continuity)
        # 2. Key frames at regular intervals (for long-term consistency)
        # 3. Motion-aware selection (frames with significant changes)
        
        def compute_motion_score(frame1, frame2):
            """Compute motion between two frames"""
            if frame1.shape != frame2.shape:
                return 0.0
            motion = torch.abs(frame2 - frame1).mean().item()
            return motion
        
        def get_frame_importance(idx, total_frames):
            """
            Calculate frame importance based on position.
            Recent frames and keyframes get higher importance.
            """
            recency_weight = (idx / total_frames) ** 2  # More recent = higher weight
            
            # Keyframe weight (every 8th frame is important)
            keyframe_weight = 1.0 if idx % 8 == 0 else 0.5
            
            return recency_weight * 0.7 + keyframe_weight * 0.3
        
        # Strategy 1: Short sequence (< 24 frames) - Use all or evenly sample
        if total_available <= 24:
            if total_available <= context_size:
                # Use all available frames
                selected_indices = list(range(total_available))
                # Pad with repeated last frame if needed
                while len(selected_indices) < context_size:
                    selected_indices.append(total_available - 1)
            else:
                # Evenly sample with bias toward recent
                step = total_available / context_size
                selected_indices = []
                for i in range(context_size):
                    # Bias toward more recent frames
                    idx = int(i * step * (1 + i / context_size * 0.3))
                    idx = min(idx, total_available - 1)
                    selected_indices.append(idx)
        
        # Strategy 2: Long sequence - Hierarchical sampling
        else:
            selected_indices = []
            
            # Phase 1: Always include most recent frames (50% of context)
            recent_count = context_size // 2  # 6 frames
            recent_start = max(0, total_available - recent_count)
            selected_indices.extend(range(recent_start, total_available))
            
            # Phase 2: Key frames at exponential intervals (25% of context)
            keyframe_count = context_size // 4  # 3 frames
            if total_available > recent_count:
                # Exponential spacing for historical frames
                remaining_frames = total_available - recent_count
                for i in range(keyframe_count):
                    # Exponential decay: more samples from recent history
                    t = (i + 1) / keyframe_count
                    idx = int(remaining_frames * (1 - t**2))
                    idx = max(0, min(idx, recent_start - 1))
                    if idx not in selected_indices:
                        selected_indices.append(idx)
            
            # Phase 3: Motion-based selection (25% of context)
            motion_count = context_size - len(selected_indices)
            if motion_count > 0 and total_available > 2:
                # Compute motion scores
                motion_scores = []
                for i in range(1, min(total_available, recent_start)):
                    if i not in selected_indices:
                        motion = compute_motion_score(all_frames[i-1], all_frames[i])
                        importance = get_frame_importance(i, total_available)
                        combined_score = motion * 0.6 + importance * 0.4
                        motion_scores.append((i, combined_score))
                
                # Select frames with highest motion scores
                motion_scores.sort(key=lambda x: x[1], reverse=True)
                for idx, _ in motion_scores[:motion_count]:
                    selected_indices.append(idx)
            
            # Ensure we have exactly context_size frames
            selected_indices = list(set(selected_indices))  # Remove duplicates
            selected_indices.sort()  # Maintain temporal order
            
            # Pad if necessary
            while len(selected_indices) < context_size:
                # Add most recent frame
                selected_indices.append(total_available - 1)
        
        # Extract selected frames
        context_frames = []
        for idx in selected_indices[:context_size]:
            frame = all_frames[idx]
            
            # Ensure correct shape
            if frame.dim() == 3:  # [16, lat_h, lat_w]
                frame = frame.unsqueeze(1)  # [16, 1, lat_h, lat_w]
            
            context_frames.append(frame)
        
        # Apply lightweight stabilization (no heavy regularization)
        context_frames = self._stabilize_context_frames(context_frames)
        
        # Stack frames
        context_tensor = torch.cat(context_frames, dim=1)
        
        print(f"Selected indices: {selected_indices[:context_size]}")
        print(f"Context shape: {context_tensor.shape}")
        
        return context_tensor.to(device)

    def _stabilize_context_frames(self, frames):
        """
        Lightweight stabilization to prevent distribution shift.
        Much gentler than the original regularization.
        """
        if not frames:
            return frames
        
        # Compute global statistics
        all_frames_cat = torch.cat(frames, dim=1)
        global_mean = all_frames_cat.mean()
        global_std = all_frames_cat.std()
        
        stabilized_frames = []
        for frame in frames:
            # Very gentle normalization only if distribution is extreme
            frame_std = frame.std()
            frame_mean = frame.mean()
            
            if frame_std < 0.1 or frame_std > 3.0:
                # Only fix extreme cases
                frame = (frame - frame_mean) / (frame_std + 1e-8)
                frame = frame * global_std + global_mean
            elif abs(frame_mean) > 2.0:
                # Only shift if mean is too far off
                frame = frame - frame_mean + global_mean
            
            stabilized_frames.append(frame)
        
        return stabilized_frames

    # Alternative: Sliding Window Approach for Very Long Videos
    def _get_sliding_window_context(self, all_frames, context_size, current_position, lat_h, lat_w):
        """
        Sliding window approach for extremely long video generation.
        Maintains local coherence while preserving long-term structure.
        """
        device = self.device
        total_frames = len(all_frames)
        
        if not all_frames:
            return torch.zeros(16, context_size, lat_h, lat_w, device=device)
        
        # Window composition:
        # - 70% recent frames (local context)
        # - 20% medium-range frames  
        # - 10% long-range anchors
        
        recent_size = int(context_size * 0.7)  # 8-9 frames
        medium_size = int(context_size * 0.2)  # 2-3 frames
        anchor_size = context_size - recent_size - medium_size  # 1-2 frames
        
        selected_indices = []
        
        # Recent window
        recent_start = max(0, current_position - recent_size)
        recent_end = current_position
        selected_indices.extend(range(recent_start, recent_end))
        
        # Medium-range sampling
        if recent_start > 0:
            medium_start = max(0, recent_start - recent_size * 3)
            medium_candidates = list(range(medium_start, recent_start))
            if len(medium_candidates) > medium_size:
                # Sample evenly
                step = len(medium_candidates) / medium_size
                for i in range(medium_size):
                    idx = medium_start + int(i * step)
                    selected_indices.append(idx)
            else:
                selected_indices.extend(medium_candidates)
        
        # Long-range anchors (beginning and key points)
        if current_position > context_size * 2:
            # Always include first frame as anchor
            selected_indices.append(0)
            
            # Add checkpoint frames at regular intervals
            checkpoint_interval = total_frames // 8
            for i in range(1, anchor_size):
                checkpoint_idx = i * checkpoint_interval
                if checkpoint_idx < medium_start:
                    selected_indices.append(checkpoint_idx)
        
        # Ensure we have the right number of frames
        selected_indices = sorted(list(set(selected_indices)))[:context_size]
        
        # Pad if necessary
        while len(selected_indices) < context_size:
            selected_indices.append(max(0, current_position - 1))
        
        # Extract and process frames
        context_frames = []
        for idx in selected_indices:
            frame = all_frames[idx]
            if frame.dim() == 3:
                frame = frame.unsqueeze(1)
            context_frames.append(frame)
        
        return torch.cat(context_frames, dim=1).to(device)

    def _denoise_step_consistent(self, latent, t, arg_c, arg_null, guide_scale, scheduler, seed_g, step_idx, total_steps):
        """Denoise step for consistent y_section approach."""
        device = 'cuda:0'
          # For quantized models, add momentum to stabilize
        if hasattr(self, '_prev_noise_pred') and self.use_quantized:
            momentum = 0.3  # Smooth out predictions
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

    def _combine_generated_frames(self, all_frames, start_latent, total_frames, device):
        """Combine all generated frames into final latent."""
        # Handle start_latent with different dimensions
        if start_latent.dim() == 5:
            # Shape: [16, 1, 21, 90, 68] - extract first frame
            start_frame = start_latent[:, 0, 0:1, :, :].to(device)  # Shape: [16, 1, 90, 68]
        elif start_latent.dim() == 4:
            # Shape: [16, 21, 90, 68] - take first frame
            start_frame = start_latent[:, 0:1, :, :].to(device)  # Shape: [16, 1, 90, 68]
        else:
            # Shape: [16, 90, 68] - add temporal dimension
            start_frame = start_latent.unsqueeze(1).to(device)  # Shape: [16, 1, 90, 68]
        
        # Stack all generated frames
        if all_frames:
            # Ensure all frames have consistent shape [16, 1, 90, 68]
            frames_to_concat = [start_frame]
            for f in all_frames[:total_frames-1]:
                if f.dim() == 3:
                    frames_to_concat.append(f.unsqueeze(1).to(device))
                else:
                    frames_to_concat.append(f.to(device))
            
            final_latent = torch.cat(frames_to_concat, dim=1)
        else:
            final_latent = start_frame
        
        # Ensure we have exactly the target number of frames
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
        
        print(f"Final latent shape: {final_latent.shape}")
        return final_latent

    def _move_args_to_device(self, args, device):
        """Helper to move arguments to device."""
        args_gpu = {}
        for key, value in args.items():
            if isinstance(value, torch.Tensor):
                args_gpu[key] = value.to(device)
            elif isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
                args_gpu[key] = [v.to(device) for v in value]
            else:
                args_gpu[key] = value
        return args_gpu
    
    def generate_correlated_noise(self, base_noise, correlation_strength=0.7):
        
        """
        Generate noise that's correlated with previous section's final frames.
        """
        # Create fresh noise
        fresh_noise = torch.randn_like(base_noise)
        
        # Blend with base noise for correlation
        correlated_noise = correlation_strength * base_noise + \
                        (1 - correlation_strength) * fresh_noise
        
        # Renormalize to maintain variance
        correlated_noise = correlated_noise / correlated_noise.std() * fresh_noise.std()
        
        return correlated_noise
    
    def _get_optimized_context_window(self, all_frames, context_size, overlap_frames, lat_h, lat_w):
        """
        Optimized context selection focusing on recent frames and motion continuity.
        """
        if not all_frames:
            return torch.zeros(16, context_size, lat_h, lat_w, device=self.device)
        
        total_available = len(all_frames)
        selected_frames = []
        
        # Strategy: 70% recent, 30% keyframes
        recent_count = int(context_size * 0.7)
        keyframe_count = context_size - recent_count
        
        # Always include the most recent frames
        recent_start = max(0, total_available - recent_count)
        for i in range(recent_start, total_available):
            frame = all_frames[i]
            if frame.dim() == 3:
                frame = frame.unsqueeze(1)
            selected_frames.append(frame)
        
        # Add keyframes from earlier in sequence
        if recent_start > 0 and keyframe_count > 0:
            # Exponential spacing for historical frames
            for i in range(keyframe_count):
                t = i / max(1, keyframe_count - 1)
                idx = int(recent_start * (1 - t**2))
                idx = max(0, min(idx, recent_start - 1))
                
                frame = all_frames[idx]
                if frame.dim() == 3:
                    frame = frame.unsqueeze(1)
                selected_frames.append(frame)
        
        # Ensure we have exactly context_size frames
        selected_frames = selected_frames[:context_size]
        while len(selected_frames) < context_size:
            # Pad with last frame
            selected_frames.append(selected_frames[-1].clone())
        
        # Stack efficiently
        return torch.cat(selected_frames, dim=1)

    def _create_generation_mask(self, context_frames, gen_frames, lat_h, lat_w, device):
        """Create generation mask directly on target device."""
        total_frames = context_frames + gen_frames
        
        # Create mask directly on target device
        msk = torch.ones(1, total_frames * 4, lat_h, lat_w, device=device)
        msk[:, context_frames * 4:] = 0
        
        # Reshape in place
        msk = msk.view(1, total_frames, 4, lat_h, lat_w)
        return msk.transpose(1, 2)[0]

    def _estimate_motion_direction(self, recent_frames, device):
        """Estimate motion direction from recent frames for coherent noise init."""
        if len(recent_frames) < 2:
            return torch.zeros_like(recent_frames[-1])
        
        # Simple motion estimation
        motion_accum = torch.zeros_like(recent_frames[-1])
        
        for i in range(1, len(recent_frames)):
            frame_diff = recent_frames[i] - recent_frames[i-1]
            # Weight more recent motion higher
            weight = (i / len(recent_frames)) ** 2
            motion_accum += frame_diff * weight
        
        # Normalize and return
        motion_accum = motion_accum / (len(recent_frames) - 1)
        return motion_accum.to(device)
    def _preserve_latent_quality(self, latent, quality_tracker, strength=0.2):
        """Preserve latent quality to prevent degradation."""
        current_mean = latent.mean()
        current_std = latent.std()
        
        # Only apply if distribution has drifted
        mean_diff = abs(current_mean.item() - quality_tracker['mean'])
        std_diff = abs(current_std.item() - quality_tracker['std'])
        
        if mean_diff > 0.5 or std_diff > 0.5:
            # Normalize
            normalized = (latent - current_mean) / (current_std + 1e-8)
            
            # Apply target statistics with blending
            target_std = quality_tracker['std'] * (1 - strength) + current_std.item() * strength
            target_mean = quality_tracker['mean'] * (1 - strength) + current_mean.item() * strength
            
            latent = normalized * target_std + target_mean
        
        return latent

    def _preserve_frame_quality(self, frame, quality_tracker):
        """Final quality check for individual frames."""
        # Check for extreme values
        if (frame > 4.0).any() or (frame < -4.0).any():
            frame = torch.clamp(frame, -3.5, 3.5)
        
        # Check distribution
        frame_std = frame.std()
        if frame_std < 0.1 or frame_std > 3.0:
            frame = (frame - frame.mean()) / (frame_std + 1e-8)
            frame = frame * quality_tracker['std'] + quality_tracker['mean']
        
        return frame