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
                 shift=5.0,  # Reduced from 5.0 for smaller resolution
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
        
        
        print('image shape', img.shape, 'none',img[None].shape , 'F', F)
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
        sampling_steps = 20
        sample_solver = 'dpm++'
        # Calculate total frames and sections
        latent_window_size = 21
        total_frames = 42
        
        
      
        context_window_size = 12
        generation_window_size = 9
        total_sections = math.ceil(total_frames / latent_window_size)
        # Ensure it adds up to latent_window_size
        
        
        print(f"Generating {total_frames} frames in {total_sections} sections")
        print(f"Using fixed context window: {context_window_size} frames, generating: {generation_window_size} frames per section")
        
        # Store the start latent
        start_latent = y  # Shape: (16, lat_h, lat_w)
        
        
        all_generated_latents = []
        all_generated_frames = []  # Flattened list of all frames for easy access
        context_frame_decoded=[]
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
                    frame_offset = 20 + (section_idx - 1) * 20 # Correct offset calculation
    
                   
                    overlap_frames = 3
                    new_frames_to_generate = 20
                    context_window_size = 12
               
                    img_context = TF.to_tensor(context_frame_decoded[0]).sub_(0.5).div_(0.5).to(self.device)
                    print('context frames', img_context.shape,'none', img_context[None].shape)
                    

                    h_context, w_context = img_context.shape[1:]
                    img_context=img_context.to('cuda:2')
                    context_y =self.vae.encode([
             torch.concat([
                 torch.nn.functional.interpolate(
          img_context[None], size=(h_context, w_context), mode='bicubic').transpose(
              0, 1),
                 torch.zeros(3, F - 1, h_context, w_context, device='cuda:2')
             ],
              dim=1).to('cuda:2')
         ])[0]
        
                    
                    section_noise = torch.randn(
                        16, 21, lat_h, lat_w,
                        dtype=torch.float32,
                        generator=rng,
                        device=device
                    )
                    
                  
                    if len(all_generated_frames) >= 3:
                       
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
                    
                    
                    latent_sequence=context_y
                  
                    msk = torch.ones(1, 81, lat_h, lat_w, device=self.device)
                    msk[:, 1:] = 0
                    msk = torch.concat([
                        torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
                    ],
                                    dim=1)
                    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
                    msk_section = msk.transpose(1, 2)[0]
                    msk_section = msk_section.to('cuda:2')
                    print('msk_section shape', msk_section.shape)
                    print('latent_sequence shape', latent_sequence.shape)
                    
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
                
                for step_idx, t in enumerate(tqdm(timesteps, desc=f"Section {section_idx + 1}")):
                    if step_idx % 5 == 0:
                        torch.cuda.empty_cache()
                    
                   
                    latent_padded = latent
                    print(latent_padded.shape,'shape')
                    # Denoise
                    latent_padded = self._denoise_step_consistent(
                        latent_padded, t, arg_c, arg_null, guide_scale,
                        sample_scheduler, seed_g, step_idx, len(timesteps)
                    )
                    
                    # Extract only the generation window
                    latent = latent_padded
                    
                    # break
                    gc.collect()
                    
                    
                    
                if section_idx == 0:
                    generated_frames = latent[:, 1:, :, :]  # 20 new frames
                else:
                   
                    generated_frames = latent[:, -9:, :, :]  # 9 new frames
                  
                quality_tracker['mean'] = generated_frames.mean().item() * 0.1 + quality_tracker['mean'] * 0.9
                quality_tracker['std'] = generated_frames.std().item() * 0.1 + quality_tracker['std'] * 0.9
                
                # Store frames with quality check
                for frame_idx in range(generated_frames.shape[1]):
                    frame = generated_frames[:, frame_idx:frame_idx+1, :, :]
                    
                    # Final quality preservation per frame
                    frame = self._preserve_frame_quality(frame, quality_tracker)
                    
                    all_generated_frames.append(frame)
                
                all_generated_latents.append(generated_frames.cpu())
              
                del latent, section_noise
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()
                
                
                print(f"Section {section_idx + 1} complete. Generated {generation_window_size} new frames")
                print(f"Total frames generated so far: {len(all_generated_frames)}")
               
                all_generated_framess = torch.cat(all_generated_frames, dim=1)
                all_generated_framess=all_generated_framess.to('cuda:2')
                print(f"all_generated_framess: {all_generated_framess.shape}")
                videos = self.vae.decode([all_generated_framess])
               
               
                videos = videos[0]  # If it's [1, N, C, H, W], take the first sample
                # videos = videos.unsqueeze(1)         # [T, 1, H, W]
                # videos = videos.repeat(1, 3, 1, 1) 
                print(videos.shape,'vidoes shape')
                def cache_video_and_get_last_frame(tensor, save_file=None, fps=24, save_image=None, suffix='.mp4',
                                   nrow=8, normalize=True, value_range=(-1, 1), retry=3):
                    """
                    Cache video tensor to file and return the last frame as numpy array.
                    
                    Uses the same processing logic as the working cache_video function.
                    
                    Returns:
                        tuple: (cache_file_path, last_frame_img) where last_frame_img is numpy array [H, W, C]
                    """
                    import random
                    import string
                    cache_file=''
                    video_tensor = tensor.cpu().detach()  # move to CPU if not already
                    if value_range == (-1, 1):
                        video_tensor = (video_tensor + 1.0) / 2.0
                    video_tensor = torch.clamp(video_tensor, 0.0, 1.0)  # Ensure values in [0, 1]

                    # Convert to NumPy: [T, H, W, C]
                    video_np = video_tensor.permute(1, 2, 3, 0).numpy()  # [77, 720, 544, 3]
                    video_np_uint8 = (video_np * 255).astype(np.uint8)

                    # Save MP4
                    imageio.mimsave(save_file, video_np_uint8, fps=24)
                    frames_12 = video_np_uint8[-12:] 
                    for id,frame in enumerate(frames_12):
                        context_frame_decoded.append(frame)
                        Image.fromarray(frame).save(f"{save_image}{id}.png")

                    # Extract and save last frame as PNG
                    last_frame_img = video_np_uint8[-1]  # Shape [720, 544, 3]
                    # Image.fromarray(last_frame_img).save(save_image)
                    
                    return cache_file, last_frame_img
                    
                    


                filename=f"section-video-{section_idx}.mp4"
                imagename=f"last-frame-{section_idx}-"
                cache_path, last_frame_img =cache_video_and_get_last_frame(videos, save_file=filename, fps=24, save_image=imagename)
                # context_frame_decoded.append(last_frame_img)
                

        # Combine all generated frames
        final_latent = self._combine_generated_frames(
            all_generated_frames, start_latent, total_frames, device='cuda:2'
        )
        print(final_latent.shape,'final latent shape')
        
        # Decode the final video
        if self.rank == 0:
            if offload_model:
                self.offload_model_to_cpu()
            videos = self.vae.decode([final_latent])
            print(videos[0].shape,'vidoes shape')
            return videos[0]
        
        return None

   
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
    
    
