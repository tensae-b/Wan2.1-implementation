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
    ):
        r"""
        Initializes the image-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
        """
        block_num=40
        first_block=8
        def block_distributed_forward(self, x, t=None, context=None, seq_len=None, clip_fea=None, y=None, **other_kwargs):
            """
            Distributed forward pass that moves data through GPUs sequentiallyS
            Last GPU has fewer blocks to accommodate encoders/VAE
            """
            num_gpus = 4
            total_blocks = len(self.blocks)
            
            # Calculate block distribution (same as in init)
            # last_gpu_reduction = max(1, total_blocks // 8)
            # blocks_for_first_three = total_blocks - last_gpu_reduction
            # blocks_per_first_gpu = math.ceil(blocks_for_first_three / 3)
            blocks_to_process = min(block_num, total_blocks)
               
            first_gpu_blocks = first_block
            remaining_blocks = blocks_to_process - first_gpu_blocks
            blocks_per_other_gpu = remaining_blocks // 3  # Divide among GPUs 1, 2, 3
            remainder = remaining_blocks % 3
            # Start processing on the device where patch_embedding is located
            device = self.patch_embedding.weight.device
            
            # Clear cache before processing
            torch.cuda.empty_cache()
            
            if self.freqs.device != device:
                self.freqs = self.freqs.to(device)

            if self.model_type != 'vace' and y is not None:
                x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

            # embeddings - process in smaller chunks if needed
            x_embedded = []
            for i, x_chunk in enumerate(x):
                x_emb = self.patch_embedding(x_chunk.unsqueeze(0))
                x_embedded.append(x_emb)
                # Clear intermediate if multiple chunks
                if len(x) > 1:
                    del x_chunk
                    torch.cuda.empty_cache()
           
            x = x_embedded
            # x = x.to(torch.bfloat16)
            grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
            x = [u.flatten(2).transpose(1, 2) for u in x]
            seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
            print(seq_lens.max(),'seq len')
            assert seq_lens.max() <= seq_len
            x = torch.cat([
                torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
                for u in x
            ])

            # Clear embedding intermediates
            del x_embedded
            torch.cuda.empty_cache()

            # time embeddings
            with amp.autocast(dtype=torch.float32):
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, t).float())
                e0 = self.time_projection(e).unflatten(1, (6, self.dim))
                assert e.dtype == torch.float32 and e0.dtype == torch.float32
            
            # Clear time embedding intermediates
            
            torch.cuda.empty_cache()

            # context
            context_lens = None
            context_processed = self.text_embedding(
                torch.stack([
                    torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in context
                ]))

            if self.model_type != 'vace' and clip_fea is not None:
                context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
                context_processed = torch.concat([context_clip, context_processed], dim=1)
                del context_clip

            # Clear context intermediates
            torch.cuda.empty_cache()

            # arguments - this is the properly formatted kwargs
            block_kwargs = dict(
                e=e0,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                freqs=self.freqs,
                context=context_processed,
                context_lens=context_lens)
            
            # Process blocks across GPUs with aggressive memory management
            for gpu_id in range(num_gpus):
                target_device = torch.device(f'cuda:{gpu_id}')
                
                # Clear cache before moving to new GPU
                # torch.cuda.empty_cache()
                
                # Move x to current GPU
                if isinstance(x, list):
                    x = [tensor.to(target_device) for tensor in x]
                else:
                    x = x.to(target_device)
                
                # Move only essential kwargs to current GPU to save memory
                local_kwargs = {}
                for key, value in block_kwargs.items():
                    if isinstance(value, torch.Tensor):
                        local_kwargs[key] = value.to(target_device)
                    elif isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
                        local_kwargs[key] = [v.to(target_device) for v in value]
                    else:
                        local_kwargs[key] = value
                
                # Calculate block range for current GPU
                if gpu_id == 0:
                    start_block = 0
                    end_block = min(first_gpu_blocks, blocks_to_process)
                else:
                    start_block = first_gpu_blocks + (gpu_id - 1) * blocks_per_other_gpu
                    if gpu_id <= remainder:
                        start_block += (gpu_id - 1)
                        end_block = min(start_block + blocks_per_other_gpu + 1, blocks_to_process)
                    else:
                        start_block += remainder
                        end_block = min(start_block + blocks_per_other_gpu, blocks_to_process)
                
                print(f"[block_distributed_forward] Processing blocks {start_block} to {end_block-1} on cuda:{gpu_id}")
                # x = x.bfloat16()
                # Process blocks assigned to this GPU
                for block_idx in range(start_block, end_block):
                    # print(f"[block_distributed_forward] Running block {block_idx} on cuda:{gpu_id}")
                    
                    # Process block with memory management
                    try:
                        x = self.blocks[block_idx](x, **local_kwargs)
                    except torch.cuda.OutOfMemoryError:
                        # Emergency memory cleanup and retry
                        torch.cuda.empty_cache()
                        gc.collect()
                        print(f"OOM on block {block_idx}, retrying after cleanup...")
                        x = self.blocks[block_idx](x, **local_kwargs)
                    
                    # Clear cache after each block if on GPU 0 (most memory constrained)
                    if gpu_id == 0:
                        torch.cuda.empty_cache()
                
                # Clear local kwargs after processing this GPU
                del local_kwargs
                # torch.cuda.empty_cache()
            
            # Apply head layer on last GPU
            final_device = torch.device(f'cuda:{num_gpus-1}')
            if isinstance(x, list):
                x = [tensor.to(final_device) for tensor in x]
            else:
                x = x.to(final_device)
            
            if hasattr(self, 'head'):
                # Move e0 to final device for head layer
                e0_final = e.to(final_device)
                x = self.head(x, e0_final)
                del e0_final
            
            # Unpatchify on final GPU
            if hasattr(self, 'unpatchify'):
                grid_sizes_final = grid_sizes.to(final_device)
                x = self.unpatchify(x, grid_sizes_final)
                result = [u.float() for u in x] if isinstance(x, list) else [x.float()]
                del grid_sizes_final
                return result
            
            return x

        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.use_usp = use_usp
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        print('vae uploaded')
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=torch.device('cuda:3'))  # Place VAE on GPU 3

        self.clip = CLIPModel(
            dtype=config.clip_dtype,
            device=self.device,
            checkpoint_path=os.path.join(checkpoint_dir,
                                         config.clip_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.clip_tokenizer))

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False)

        if t5_fsdp or dit_fsdp or use_usp:
            init_on_cpu = False

        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size
            from .distributed.xdit_context_parallel import (
                usp_attn_forward,
                usp_dit_forward,
            )
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()

        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            if device_id == 0:  # Only setup on rank 0 or modify condition as needed
                num_gpus = 4
                total_blocks = len(self.model.blocks)
                blocks_to_process = min(block_num, total_blocks)
                # Reserve space on last GPU for encoders/VAE by giving it fewer blocks
                # Last GPU gets ~15-20% fewer blocks to accommodate other models
                first_gpu_blocks = first_block
                remaining_blocks = blocks_to_process - first_gpu_blocks
                blocks_per_other_gpu = remaining_blocks // 3  # Divide among GPUs 1, 2, 3
                remainder = remaining_blocks % 3
                
                # Distribute blocks across GPUs
                for gpu_id in range(num_gpus):
                    if gpu_id == 0:
                        start_block = 0
                        end_block = min(first_gpu_blocks, blocks_to_process)
                    else:
                        start_block = first_gpu_blocks + (gpu_id - 1) * blocks_per_other_gpu
                        if gpu_id <= remainder:
                            start_block += (gpu_id - 1)
                            end_block = min(start_block + blocks_per_other_gpu + 1, blocks_to_process)
                        else:
                            start_block += remainder
                            end_block = min(start_block + blocks_per_other_gpu, blocks_to_process)
                    
                    target_device = torch.device(f'cuda:{gpu_id}')
                    for block_idx in range(start_block, end_block):
                        self.model.blocks[block_idx].to(target_device)
                    
                    print(f"GPU {gpu_id}: blocks {start_block}-{end_block-1} ({end_block-start_block} blocks)")
                
                # Replace forward method with distributed version
                self.model.forward = types.MethodType(block_distributed_forward, self.model)
                
                # Move model components to appropriate GPUs
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
                print('loading head')    
                if hasattr(self.model, 'head'):
                    self.model.head.to(torch.device('cuda:3'))
                # if hasattr(self.model, 'unpatchify'):
                #     self.model.unpatchify.to(torch.device('cuda:3'))
                    
                # # Move encoders and VAE to last GPU
                # self.text_encoder.model.to(torch.device('cuda:3'))
                # self.clip.model.to(torch.device('cuda:3'))
                
            else:
                # For other ranks, just put model on assigned device
                if not init_on_cpu:
                    self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt
   
        
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
                 max_area=240 * 416,  # Reduced from 480 * 832
                 frame_num=5,  # Reduced from 4  
                 shift=3.0,  # Reduced from 5.0 for smaller resolution
                 sample_solver='unipc',
                 sampling_steps=20,  # Reduced from 40
                 guide_scale=5.0,
                 n_prompt="blurry, unclear",
                 seed=-1,
                 offload_model=True):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """
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
        noise = torch.randn(
            16, (F - 1) // 4 + 1,
            lat_h,
            lat_w,
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)
        
        print('noise', noise.shape)

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
        text_encoder_device = torch.device('cuda:3')
        text_device= torch.device('cpu')
        
        if not self.t5_cpu:
            self.text_encoder.model.to(text_device)
            context = self.text_encoder([input_prompt], text_device)
            context_null = self.text_encoder([n_prompt], text_device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]
        
        self.text_encoder.model.to('cpu')
        print('text encoding done')
        torch.cuda.empty_cache()
        gc.collect()

        torch.cuda.synchronize()
        self.clip.model.to(text_encoder_device)
        img = img.to(text_encoder_device)
        clip_context = self.clip.visual([img[:, None, :, :]])
        self.clip.model.cpu()
        
        torch.cuda.empty_cache()
        gc.collect()

        torch.cuda.synchronize()
        
        if True== True:
            y = self.vae.encode([
                torch.concat([
                    torch.nn.functional.interpolate(
                        img[None], size=(h, w), mode='bicubic').transpose(
                            0, 1),
                    torch.zeros(3, F - 1, h, w, device=text_encoder_device)
                ],
                            dim=1).to(text_encoder_device)
            ])[0]
        
            msk = msk.to(text_encoder_device)
            
            img=img.to("cpu")
            torch.cuda.empty_cache()
            gc.collect()

            torch.cuda.synchronize()
            print('here ')
            latent_window_size=8
            result=self._generate_with_frame_packing(
                    noise, y, msk, context, context_null, clip_context,
                    lat_h, lat_w, F, shift, sample_solver, sampling_steps,
                    guide_scale, seed_g, max_seq_len, latent_window_size,
                    offload_model
                )
            return result
     
        else:
            y = self.vae.encode([
                torch.concat([
                    torch.nn.functional.interpolate(
                        img[None], size=(h, w), mode='bicubic').transpose(
                            0, 1),
                    torch.zeros(3, F - 1, h, w, device=text_encoder_device)
                ],
                            dim=1).to(text_encoder_device)
            ])[0]
        
            msk = msk.to(text_encoder_device)
            y = torch.concat([msk, y])
            
            
            
            @contextmanager
            def noop_no_sync():
                yield

            no_sync = getattr(self.model, 'no_sync', noop_no_sync)
            sampling_steps=40
            
            with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

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

                # sample videos
                latent = noise
                
                
                
                arg_c = {
                    'context': [context[0]],
                    'clip_fea': clip_context,
                    'seq_len': max_seq_len,
                    'y': [y],
                }

                arg_null = {
                    'context': context_null,
                    'clip_fea': clip_context,
                    'seq_len': max_seq_len,
                    'y': [y],
                }
                inital= latent
                if offload_model:
                    torch.cuda.empty_cache()
                print(f"Input latent stats:  std={latent.std().item():.4f}")

                for step_idx, t in enumerate(tqdm(timesteps)):
                    # Clear all GPU caches before each timestep
                    torch.cuda.empty_cache()
                    # Log step info
                    print(f"\n{'='*60}")
                    print(f"Step {step_idx + 1}/{len(timesteps)} - Timestep: {t.item() if hasattr(t, 'item') else t}")
                    print(f"{'='*60}")
                    latent_mean = latent.mean().item()
                    latent_std = latent.std().item()
                    latent_min = latent.min().item()
                    latent_max = latent.max().item()
                    print(f"Input latent stats: mean={latent_mean:.4f}, std={latent_std:.4f}, min={latent_min:.4f}, max={latent_max:.4f}")
                    

                    # Start processing on GPU 0 (where embeddings are)
                    latent_model_input = [latent.to(torch.device('cuda:0'))]
                    timestep = [t]
                    timestep = torch.stack(timestep).to(torch.device('cuda:0'))
                    
                    # Move arg_c to GPU 0 for initial processing (only what's needed)
                    arg_c_gpu0 = {}
                    for key, value in arg_c.items():
                        if isinstance(value, torch.Tensor):
                            arg_c_gpu0[key] = value.to(torch.device('cuda:0'))
                        elif isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
                            arg_c_gpu0[key] = [v.to(torch.device('cuda:0')) for v in value]
                        else:
                            arg_c_gpu0[key] = value

                    # Forward pass through distributed model - output will be on GPU 3
                    noise_pred_cond = self.model(
                        latent_model_input, t=timestep, **arg_c_gpu0)[0]
                    
                    # Immediately move to CPU to free GPU 3 memory
                    noise_pred_cond = noise_pred_cond.to(torch.device('cpu'))
                    cond_mean = noise_pred_cond.mean().item()
                    cond_std = noise_pred_cond.std().item()
                    print(f"Conditional noise pred: mean={cond_mean:.4f}, std={cond_std:.4f}")
                    # Clear intermediate results and GPU caches
                    del latent_model_input, arg_c_gpu0
                    torch.cuda.empty_cache()
                    
                    # Second forward pass for unconditional - fresh start
                    latent_model_input = [latent.to(torch.device('cuda:0'))]
                    timestep = torch.stack([t]).to(torch.device('cuda:0'))
                    
                    arg_null_gpu0 = {}
                    for key, value in arg_null.items():
                        if isinstance(value, torch.Tensor):
                            arg_null_gpu0[key] = value.to(torch.device('cuda:0'))
                        elif isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
                            arg_null_gpu0[key] = [v.to(torch.device('cuda:0')) for v in value]
                        else:
                            arg_null_gpu0[key] = value
                    
                    noise_pred_uncond = self.model(
                        latent_model_input, t=timestep, **arg_null_gpu0)[0]
                    
                    # Immediately move to CPU
                    noise_pred_uncond = noise_pred_uncond.to(torch.device('cpu'))
                    uncond_mean = noise_pred_uncond.mean().item()
                    uncond_std = noise_pred_uncond.std().item()
                    print(f"Unconditional noise pred: mean={uncond_mean:.4f}, std={uncond_std:.4f}")
                    # Clear all intermediate results
                    del latent_model_input, timestep, arg_null_gpu0
                    torch.cuda.empty_cache()
                    
                    # Compute guidance on CPU to save GPU memory
                    noise_pred = noise_pred_uncond + guide_scale * (
                        noise_pred_cond - noise_pred_uncond)
                    
                    guidance_diff = (noise_pred_cond - noise_pred_uncond).abs().mean().item()
                    print(f"Guidance effect magnitude: {guidance_diff:.4f}")
                    print(f"Guidance scale: {guide_scale}")
                    
                    noise_mean = noise_pred.mean().item()
                    noise_std = noise_pred.std().item()
                    print(f"Final noise pred: mean={noise_mean:.4f}, std={noise_std:.4f}")
                    # Move latent to CPU for scheduler step
                    latent = latent.to(torch.device('cpu'))

                    temp_x0 = sample_scheduler.step(
                        noise_pred.unsqueeze(0),
                        t,
                        latent.unsqueeze(0),
                        return_dict=False,
                        generator=seed_g)[0]
                    latent = temp_x0.squeeze(0)
                    new_latent_mean = latent.mean().item()
                    new_latent_std = latent.std().item()
                    new_latent_min = latent.min().item()
                    new_latent_max = latent.max().item()
                    print(f"Output latent stats: mean={new_latent_mean:.4f}, std={new_latent_std:.4f}, min={new_latent_min:.4f}, max={new_latent_max:.4f}")
                    
                    # Calculate change in latent
                    # latent_change = (latent - inital).abs().mean().item()
                    # print(f"Latent change magnitude: {latent_change:.4f}")
                    
                    # Check for NaN or extreme values
                    if torch.isnan(latent).any():
                        print("WARNING: NaN values detected in latent!")
                    if latent.abs().max() > 100:
                        print(f"WARNING: Extreme values in latent! Max abs value: {latent.abs().max().item():.4f}")

                    # Clean up immediately
                    del noise_pred_cond, noise_pred_uncond, noise_pred, temp_x0
                   
                    # Force garbage collection and cache clearing
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                if self.rank == 0:
                    if offload_model:
                        # Move final latent to GPU 3 where VAE is located
                        x0 = [latent.to(torch.device('cuda:3'))]
                        
                
                    videos = self.vae.decode(x0)

            del noise, latent
            del sample_scheduler
            if offload_model:
                gc.collect()
                torch.cuda.synchronize()
            if dist.is_initialized():
                dist.barrier()

            return videos[0] if self.rank == 0 else None
    
    
    
    def _generate_with_frame_packing(self, noise, y, msk, context, context_null, clip_context,
                                  lat_h, lat_w, F, shift, sample_solver, sampling_steps,
                                  guide_scale, seed_g, max_seq_len, latent_window_size,
                                  offload_model):
        """
        Generate long videos with consistent y_section shape across all sections.
        The model expects mask + latent sequence in y parameter.
        """
        
        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)
        sampling_steps = 40
        
        # Calculate total frames and sections
        total_frames = 30
        generation_window_size = 21
        context_window_size = 12
        
        # For subsequent sections, we generate fewer NEW frames
        new_frames_per_section = 9
        
        total_sections = 1 + math.ceil((total_frames - generation_window_size) / new_frames_per_section)
        
        print(f"Generating {total_frames} frames in {total_sections} sections")
        print(f"First section: {generation_window_size} frames")
        print(f"Subsequent sections: {context_window_size} context + {new_frames_per_section} new frames")
        
        # Store the start latent
        start_latent = y  # Shape: (16, lat_h, lat_w)
        
        all_generated_frames = []  # Store ALL generated frames
        
        device = self.device
        dtype = self.param_dtype
        processing_device = 'cuda:3'
        
        # Random generator
        rng = torch.Generator(device=device)
        if hasattr(seed_g, 'initial_seed'):
            rng.manual_seed(seed_g.initial_seed())
        else:
            rng.manual_seed(42)
        
        with amp.autocast(dtype=dtype), torch.no_grad(), no_sync():
            for section_idx in range(total_sections):
                print(f'\n=== Section {section_idx + 1}/{total_sections} ===')
                
                if section_idx == 0:
                    # First section - generate full window
                    section_noise = torch.randn(
                        16, generation_window_size, lat_h, lat_w,
                        dtype=torch.float32,
                        generator=rng,
                        device=device
                    )
                    
                    # Handle different dimensionalities of start_latent
                    if start_latent.dim() == 3:
                        # Shape: (16, H, W)
                        first_frame = start_latent.unsqueeze(1)
                    elif start_latent.dim() == 4:
                        # Shape: (16, F, H, W) - check if F > 1
                        if start_latent.shape[1] > 1:
                            # Multiple frames provided, use all of them
                            print(f"Using {start_latent.shape[1]} frames from start_latent")
                            # If we have more frames than window size, truncate
                            start_latent=start_latent.to('cuda:0')
                            if start_latent.shape[1] >= generation_window_size:
                                latent_sequence = start_latent[:, :generation_window_size, :, :]
                            else:
                                # Pad with zeros if needed
                                zeros_padding = torch.zeros(
                                    16, generation_window_size - start_latent.shape[1], lat_h, lat_w, 
                                    device=device
                                )
                                
                                latent_sequence = torch.cat([start_latent, zeros_padding], dim=1)
                        else:
                            # Single frame
                            first_frame = start_latent
                            zeros_for_generation = torch.zeros(16, generation_window_size - 1, lat_h, lat_w, device=device)
                            latent_sequence = torch.cat([first_frame, zeros_for_generation], dim=1)
                    elif start_latent.dim() == 5:
                        # Shape: (B, C, F, H, W) - extract frames
                        frames = start_latent.squeeze(0)  # Remove batch dimension
                        if frames.shape[1] >= generation_window_size:
                            latent_sequence = frames[:, :generation_window_size, :, :]
                        else:
                            zeros_padding = torch.zeros(
                                16, generation_window_size - frames.shape[1], lat_h, lat_w, 
                                device=device
                            )
                            latent_sequence = torch.cat([frames, zeros_padding], dim=1)
                    else:
                        raise ValueError(f"Unexpected start_latent dimensions: {start_latent.shape}")
                    
                    # Ensure latent_sequence has exactly generation_window_size frames
                    if latent_sequence.shape[1] != generation_window_size:
                        raise ValueError(f"Latent sequence has wrong number of frames: {latent_sequence.shape[1]} vs {generation_window_size}")
                    
                    # Create mask: 1 for first frame (conditioned), 0 for rest (generated)
                    # The mask has 4 channels per frame
                    msk = torch.ones(1, generation_window_size * 4, lat_h, lat_w, device=device)
                    msk[:, 4:] = 0  # Only first frame (4 channels) is masked
                    
                    # Reshape mask to match expected format
                    msk = msk.view(1, generation_window_size, 4, lat_h, lat_w)
                    msk_section = msk.transpose(1, 2)[0]  # Shape: (4, 21, H, W)
                    
                    # Move to processing device
                    latent = section_noise.to('cuda:0')
                    msk_section = msk_section.to('cuda:0')
                    latent_sequence = latent_sequence.to('cuda:0')
                    
                else:
                    # Subsequent sections - use context + generate new
                    
                    # Get context frames from previously generated
                    context_frames = []
                    context_frames.append(all_generated_frames[-1])
                    # for i in range(context_window_size):
                    #     idx = len(all_generated_frames) - context_window_size + i
                    #     if 0 <= idx < len(all_generated_frames):
                    #         context_frames.append(all_generated_frames[idx])
                    #     else:
                    #         # Use last available frame if we don't have enough
                    #         context_frames.append(all_generated_frames[-1] if all_generated_frames else 
                    #                             torch.zeros(16, 1, lat_h, lat_w, device=device))
                    
                    # context_tensor = torch.cat(context_frames, dim=1).to(device)
                    context_tensor = context_frames
                    try:
                        self._decode_latent_framess(context_tensor)
                    except Exception as e:
                        print(f'Error decoding frame')
                        continue
                    # Create noise for full window
                    section_noise = torch.randn(
                        16, generation_window_size, lat_h, lat_w,
                        dtype=torch.float32,
                        generator=rng,
                        device=device
                    )
                    
                    # Create latent sequence: context + zeros for new generation
                    zeros_for_generation = torch.zeros(
                        16, generation_window_size - context_window_size, lat_h, lat_w, 
                        device=device
                    )
                    latent_sequence = torch.cat([context_tensor, zeros_for_generation], dim=1)
                    
                    # Create mask: 1 for context, 0 for generation
                    # msk = torch.ones(1, generation_window_size * 4, lat_h, lat_w, device=device)
                    # msk[:, context_window_size * 4:] = 0  # Unmask generation frames
                    
                    # # Reshape mask
                    # msk = msk.view(1, generation_window_size, 4, lat_h, lat_w)
                    # msk_section = msk.transpose(1, 2)[0]  # Shape: (4, 21, H, W)
                    msk = torch.ones(1, generation_window_size * 4, lat_h, lat_w, device=device)
                    msk[:, 4:] = 0  # Only first frame (4 channels) is masked
                    
                    # Reshape mask to match expected format
                    msk = msk.view(1, generation_window_size, 4, lat_h, lat_w)
                    msk_section = msk.transpose(1, 2)[0]  # Shape: (4, 21, H, W)
                    
                    # Move to processing device
                    latent = section_noise.to(processing_device)
                    msk_section = msk_section.to(processing_device)
                    latent_sequence = latent_sequence.to(processing_device)
                
                print(f'Noise shape: {latent.shape}')
                print(f'Latent sequence shape: {latent_sequence.shape}')
                print(f'Mask shape: {msk_section.shape}')
                
                # Create y_section by concatenating mask and latent sequence
                y_section = torch.cat([msk_section, latent_sequence], dim=0)
                print(f'y_section shape: {y_section.shape}')  # Should be (20, 21, H, W)
                y_section=y_section.to('cuda:0')
                clip_context=clip_context.to('cuda:0')
                context[0]=context[0].to('cuda:0')
                context_null[0]=context_null[0].to('cuda:0')
                # Prepare arguments
                arg_c = {
                    'context': [context[0]],
                    'clip_fea': clip_context,
                    'seq_len': max_seq_len,
                    'y': [y_section],  # Mask + latent sequence
                }

                arg_null = {
                    'context': context_null,
                    'clip_fea': clip_context,
                    'seq_len': max_seq_len,
                    'y': [y_section],
                }
                
                # Initialize scheduler
                if sample_solver == 'unipc':
                    sample_scheduler = FlowUniPCMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=shift, use_dynamic_shifting=False)
                    sample_scheduler.set_timesteps(sampling_steps, device=processing_device, shift=shift)
                    timesteps = sample_scheduler.timesteps
                elif sample_solver == 'dpm++':
                    sample_scheduler = FlowDPMSolverMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=shift, use_dynamic_shifting=False)
                    sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                    timesteps, _ = retrieve_timesteps(sample_scheduler, device=processing_device, sigmas=sampling_sigmas)
                
                if offload_model:
                    torch.cuda.empty_cache()
                
                # Denoising loop
                for step_idx, t in enumerate(tqdm(timesteps, desc=f"Section {section_idx + 1}")):
                    if step_idx % 5 == 0:
                        torch.cuda.empty_cache()
                    
                    # Denoise
                    latent = self._denoise_step(
                        latent, t, arg_c, arg_null, guide_scale,
                        sample_scheduler, seed_g, step_idx, len(timesteps),
                        processing_device
                    )
                    
                    gc.collect()
                    
                
                # Store generated frames
                if section_idx == 0:
                    # Store all frames from first section
                    for frame_idx in range(generation_window_size):
                        all_generated_frames.append(latent[:, frame_idx:frame_idx+1, :, :].cpu())
                else:
                    # Store only new frames from subsequent sections
                    for frame_idx in range(context_window_size, generation_window_size):
                        all_generated_frames.append(latent[:, frame_idx:frame_idx+1, :, :].cpu())
                
                # Cleanup
                del latent, section_noise
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()
                
                print(f"Section {section_idx + 1} complete. Total frames: {len(all_generated_frames)}")

        # Combine all frames
        final_frames = all_generated_frames[:total_frames]
        final_latent = torch.cat([f.to(processing_device) for f in final_frames], dim=1)
        
        print(f"Final latent shape: {final_latent.shape}")
        
        # Decode the final video
        if self.rank == 0:
            if offload_model:
                self.offload_model_to_cpu()
            videos = self.vae.decode([final_latent])
            return videos[0]
        
        return None


    def _denoise_step(self, latent, t, arg_c, arg_null, guide_scale, scheduler, seed_g, step_idx, total_steps, device):
        """Standard denoising step."""
        
        # Move to processing device
        latent_input = [latent.to('cuda:0')]
        
        timestep = torch.tensor([t]).to('cuda:0')
        
        # Conditional prediction
        noise_pred_cond = self.model(latent_input, t=timestep, **arg_c)[0]
        
        # Unconditional prediction  
        noise_pred_uncond = self.model(latent_input, t=timestep, **arg_null)[0]
        
        # Apply classifier-free guidance
        noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)
        
        # Move back to latent device for scheduler step
        latent = latent.to(noise_pred.device)
        
        # Scheduler step
        latent_next = scheduler.step(
            noise_pred.unsqueeze(0),
            t,
            latent.unsqueeze(0),
            return_dict=False,
            generator=seed_g
        )[0].squeeze(0)
        
        return latent_next.to(device)


    def _decode_latent_frames(self, latent_frames):
        """Decode latent frames for visualization/debugging."""
        with torch.no_grad():
            # Ensure latent is on the right device for VAE
            latent_frames = latent_frames.to(self.vae.device)
            
            # Decode
            decoded = self.vae.decode([latent_frames])
            
            # You can save or visualize decoded[0] here
            print(f"Decoded frames shape: {decoded[0].shape}")
            
        return decoded[0]
    def _decode_latent_framess(self,context_frames):
        # Extract individual frames from the context tensor
        # context_frames shape should be [16, 12, lat_h, lat_w]
        context_frames.context_frames.to('cuda:3')
        num_context_frames = context_frames.shape[1]  # Should be 12
        print(f'Number of context frames: {num_context_frames}')

        import imageio
        import os

        # Create directory for context frames if it doesn't exist
        os.makedirs(f"context_frames", exist_ok=True)

        # Loop through all context frames
        for frame_idx in range(num_context_frames):
            print(f'\n--- Processing context frame {frame_idx + 1}/{num_context_frames} ---')
            
            # Extract single frame: [16, lat_h, lat_w]
            single_frame = context_frames[:, frame_idx:frame_idx+1, :, :]  # Keep the temporal dimension
            print(f'Single frame shape: {single_frame.shape}')
            single_frame.single_frame.to('cuda:3')
            try:
                # Decode using VAE
                with torch.no_grad():
                    decoded_video = self.vae.decode([single_frame])
                    print(f'Decoded video shape: {decoded_video[0].shape}')
                    
                    # Process the decoded frame
                    video = decoded_video[0]  # Get the video tensor
                    
                    # Handle different possible shapes
                    if video.dim() == 5:  # [B, C, F, H, W]
                        decoded = video.squeeze(0)  # Remove batch -> [C, F, H, W]
                        decoded = decoded.squeeze(1)  # Remove frame -> [C, H, W]
                    elif video.dim() == 4:  # [C, F, H, W] or [B, C, H, W]
                        if video.shape[1] == 1:  # [C, 1, H, W] - temporal dimension
                            decoded = video.squeeze(1)  # Remove temporal -> [C, H, W]
                        else:  # [B, C, H, W] - batch dimension
                            decoded = video.squeeze(0)  # Remove batch -> [C, H, W]
                    elif video.dim() == 3:  # [C, H, W]
                        decoded = video
                    else:
                        decoded = video
                    
                    print(f'After squeezing: {decoded.shape}')
                    
                    # Convert from [C, H, W] to [H, W, C]
                    if decoded.dim() == 3 and decoded.shape[0] in [1, 3]:  # Channel first
                        decoded = decoded.permute(1, 2, 0)
                    elif decoded.dim() == 4:  # Still has extra dimension
                        # Force remove any remaining singleton dimensions except spatial
                        while decoded.dim() > 3:
                            # Find the first dimension that's 1 and squeeze it
                            singleton_dims = [i for i, size in enumerate(decoded.shape) if size == 1]
                            if singleton_dims:
                                decoded = decoded.squeeze(singleton_dims[0])
                            else:
                                # If no singleton dims, take the first slice
                                decoded = decoded[0]
                                            
                                            # Now should be [C, H, W], convert to [H, W, C]
                            if decoded.shape[0] in [1, 3]:
                                decoded = decoded.permute(1, 2, 0)
                                        
                            print(f'After permute: {decoded.shape}')
                    
                    # Normalize and convert to uint8
                    # VAE output is typically in range [-1, 1], convert to [0, 1] first
                    if decoded.min() < 0:
                        decoded = (decoded + 1.0) / 2.0  # [-1, 1] -> [0, 1]
                    
                    # Clamp to valid range and convert to uint8
                    decoded = (decoded * 255.0).clamp(0, 255).to(torch.uint8)
                    
                    # Convert to numpy and save
                    decoded_np = decoded.cpu().numpy()
                    
                    # Ensure we have the right dimensions for saving
                    if decoded_np.ndim == 3:
                        if decoded_np.shape[-1] == 1:  # Grayscale with channel dim
                            decoded_np = decoded_np.squeeze(-1)  # Remove channel -> [H, W]
                        elif decoded_np.shape[0] == 1:  # Channel first grayscale
                            decoded_np = decoded_np.squeeze(0)  # Remove channel -> [H, W]
                        # If shape[-1] == 3, it's RGB and ready to save
                    elif decoded_np.ndim == 2:
                        # Already [H, W], good for grayscale
                        pass
                    else:
                        print(f'Unexpected decoded shape: {decoded_np.shape}')
                        continue
                    
                    # Save the frame
                    filename = f"context_frames/frame_{frame_idx:02d}.png"
                    imageio.imwrite(filename, decoded_np)
                    
                    print(f'Saved frame to: {filename}')
                    print(f'Frame stats: min={decoded_np.min()}, max={decoded_np.max()}, shape={decoded_np.shape}')
                    
            except Exception as e:
                print(f'Error processing frame {frame_idx}: {str(e)}')
                print(f'Frame stats: min={single_frame.min():.3f}, max={single_frame.max():.3f}')
                continue

        print(f'\n--- Summary ---')
        print(f'Processed {num_context_frames} context frames')
        print(f'Original context tensor shape: {context_frames.shape}')
        print(f'Frames saved to: context_frames/ directory')        