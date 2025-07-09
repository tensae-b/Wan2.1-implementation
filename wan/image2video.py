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
        
        if True== True:
            total_frames=42
            frames_per_section=21
            overlap_frames=4
            result= self._generate_with_frame_packing(
                                input_prompt,
                                img,
                                max_area,
                                total_frames,
                                shift,
                                sample_solver,
                                sampling_steps,
                                guide_scale,
                                n_prompt,
                                seed,
                                offload_model,
                                frames_per_section,
                                overlap_frames)
            return result
            
        else:
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
            
        
        
            # y = self.vae.encode([
            #     torch.concat([
            #         torch.nn.functional.interpolate(
            #             img[None], size=(h, w), mode='bicubic').transpose(
            #                 0, 1),
            #         torch.zeros(3, F - 1, h, w, device=text_encoder_device)
            #     ],
            #                 dim=1).to(text_encoder_device)
            # ])[0]
        
            # msk = msk.to(text_encoder_device)
            
            # img=img.to("cpu")
            # torch.cuda.empty_cache()
            # gc.collect()

            # torch.cuda.synchronize()
            # print('here ')
            # latent_window_size=8
            # result=self._generate_with_frame_packing(
            #         noise, y, msk, context, context_null, clip_context,
            #         lat_h, lat_w, F, shift, sample_solver, sampling_steps,
            #         guide_scale, seed_g, max_seq_len, latent_window_size,
            #         offload_model
            #     )
            # return result
     
        
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
    
    
    
    def _generate_with_frame_packing(self,
                                input_prompt,
                                img,
                                max_area,
                                total_frames,
                                shift,
                                sample_solver,
                                sampling_steps,
                                guide_scale,
                                n_prompt,
                                seed,
                                offload_model,
                                frames_per_section,
                                overlap_frames):
        """
        Generate long videos using sliding window approach with proper continuity.
        """
        
        # Initial setup
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)
        
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
        
        # Calculate latent dimensions
        total_latent_frames = (total_frames - 1) // 4 + 1
        section_latent_frames = (frames_per_section - 1) // 4 + 1
        overlap_latent_frames = overlap_frames // 4
        
        # Text and CLIP encoding (once for all sections)
        print('Encoding text and image...')
        text_encoder_device = torch.device('cuda:3')
        
        # Text encoding
        self.text_encoder.model.to(torch.device('cpu'))
        context = self.text_encoder([input_prompt], torch.device('cpu'))
        context_null = self.text_encoder([n_prompt if n_prompt else self.sample_neg_prompt], torch.device('cpu'))
        self.text_encoder.model.cpu()
        
        # CLIP encoding
        self.clip.model.to(text_encoder_device)
        img_clip = img.to(text_encoder_device)
        clip_context = self.clip.visual([img_clip[:, None, :, :]])
        self.clip.model.cpu()
        torch.cuda.empty_cache()
        
        # Initialize seed
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device='cpu')
        seed_g.manual_seed(seed)
        
        # Generate initial full-length noise
        print(f"Generating noise for {total_frames} frames ({total_latent_frames} latent frames)")
        full_noise = torch.randn(
            16, total_latent_frames,
            lat_h, lat_w,
            dtype=torch.float32,
            generator=seed_g,
            device='cpu'  # Keep on CPU to save GPU memory
        )
        
        # Initialize the full latent tensor (will be progressively denoised)
        full_latent = full_noise.clone()
        
        # Prepare initial frame
        initial_frame = torch.nn.functional.interpolate(
            img[None], size=(h, w), mode='bicubic'
        ).transpose(0, 1)
        
        print(f"\nGenerating {total_frames} frames using sliding window")
        print(f"Window size: {frames_per_section}, Overlap: {overlap_frames}")
        
        @contextmanager
        def noop_no_sync():
            yield
        
        no_sync = getattr(self.model, 'no_sync', noop_no_sync)
       
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
            # Initialize scheduler
            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=shift, use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            else:
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=shift, use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(sample_scheduler, device=self.device, sigmas=sampling_sigmas)
            full_noise_pred = torch.zeros_like(full_latent)
            full_weight_mask = torch.zeros_like(full_latent)

            # Process each timestep
            for step_idx, t in enumerate(tqdm(timesteps, desc="Denoising")):
                print(f"\n{'='*60}")
                print(f"Timestep {step_idx + 1}/{len(timesteps)}: t={t.item() if hasattr(t, 'item') else t}")
                print(f"{'='*60}")
                
                # For each timestep, process the video in sliding windows
                updated_latent = full_latent.clone()
                window_start = 0
                
                while window_start < total_latent_frames:
                    
                    window_end = min(window_start + section_latent_frames, total_latent_frames)
                    window_frames = window_end - window_start
                    
                    # Extract window from full latent
                    window_latent = full_latent[:, window_start:window_end].to(self.device)
                    
                    # Calculate actual frame positions
                    frame_start = window_start * 4
                    frame_end = min(window_end * 4 - 3, total_frames)
                    window_pixel_frames = frame_end - frame_start
                    
                    print(f"  Window: latent frames {window_start}-{window_end}, pixel frames {frame_start}-{frame_end}")
                    
                    # Create mask for this window
                    msk = torch.zeros(1, window_pixel_frames, lat_h, lat_w, device=self.device)
                    if window_start == 0:
                        msk[:, 0] = 1  # First frame is known
                    
                    # Handle VAE encoding for y
                    if window_start == 0:
                        # First window includes the initial frame
                        y_frames = torch.cat([
                            initial_frame.to(text_encoder_device),
                            torch.zeros(3, window_pixel_frames - 1, h, w, device=text_encoder_device)
                        ], dim=1)
                    else:
                        # Other windows are all zeros (to be generated)
                        y_frames = torch.zeros(3, window_pixel_frames, h, w, device=text_encoder_device)
                    
                    # Reshape mask for VAE format
                    msk_vae = msk.clone()
                    if window_pixel_frames % 4 != 0:
                        pad_frames = 4 - (window_pixel_frames % 4)
                        msk_vae = torch.cat([msk_vae, torch.zeros(1, pad_frames, lat_h, lat_w, device=self.device)], dim=1)
                        y_frames = torch.cat([y_frames, torch.zeros(3, pad_frames, h, w, device=text_encoder_device)], dim=1)
                    
                    msk_vae = msk_vae.view(1, -1, 4, lat_h, lat_w).transpose(1, 2)[0]
                    
                    # VAE encode
                    y = self.vae.encode([y_frames])[0]
                    y = torch.concat([msk_vae.to(text_encoder_device), y])
                    
                    # Calculate sequence length for this window
                    max_seq_len = window_frames * lat_h * lat_w // (self.patch_size[1] * self.patch_size[2])
                    max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size
                    
                    # Prepare arguments
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
                    
                    # Forward pass for this window
                    torch.cuda.empty_cache()
                    
                    # Conditional
                    latent_input = [window_latent.to(torch.device('cuda:0'))]
                    timestep = torch.stack([t]).to(torch.device('cuda:0'))
                    
                    arg_c_gpu = {k: (v.to(torch.device('cuda:0')) if isinstance(v, torch.Tensor) 
                                else [x.to(torch.device('cuda:0')) for x in v] if isinstance(v, list) 
                                else v) for k, v in arg_c.items()}
                    
                    noise_pred_cond = self.model(latent_input, t=timestep, **arg_c_gpu)[0].cpu()
                    
                    del latent_input, arg_c_gpu
                    torch.cuda.empty_cache()
                    
                    # Unconditional
                    latent_input = [window_latent.to(torch.device('cuda:0'))]
                    
                    arg_null_gpu = {k: (v.to(torch.device('cuda:0')) if isinstance(v, torch.Tensor) 
                                else [x.to(torch.device('cuda:0')) for x in v] if isinstance(v, list) 
                                else v) for k, v in arg_null.items()}
                    
                    noise_pred_uncond = self.model(latent_input, t=timestep, **arg_null_gpu)[0].cpu()
                    
                    del latent_input, timestep, arg_null_gpu
                    torch.cuda.empty_cache()
                    
                    # Apply guidance
                    noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)
                    
                    window_weights = torch.ones_like(window_latent)
                    if window_start > 0 and overlap_latent_frames > 0:
                        # Fade in from left overlap
                        fade_in = torch.linspace(0, 1, overlap_latent_frames, device=window_latent.device)
                        fade_in = fade_in.view(1, -1, 1, 1).expand_as(window_latent[:, :overlap_latent_frames])
                        window_weights[:, :overlap_latent_frames] *= fade_in
                    
                    if window_end < total_latent_frames and overlap_latent_frames > 0:
                        # Fade out to right overlap
                        fade_out = torch.linspace(1, 0, overlap_latent_frames, device=window_latent.device)
                        fade_out = fade_out.view(1, -1, 1, 1).expand_as(window_latent[:, -overlap_latent_frames:])
                        window_weights[:, -overlap_latent_frames:] *= fade_out
                    
                    # Accumulate weighted noise predictions
                    full_noise_pred[:, window_start:window_end] += noise_pred.cpu() * window_weights.cpu()
                    full_weight_mask[:, window_start:window_end] += window_weights.cpu()
                    
                    # Move to next window
                    window_start += section_latent_frames - overlap_latent_frames
                    
                    # Clean up
                    del window_latent, noise_pred, noise_pred_cond, noise_pred_uncond
                    torch.cuda.empty_cache()
                    
                    # Move to next window with overlap
                    window_start += section_latent_frames - overlap_latent_frames
                    
                   
                    torch.cuda.empty_cache()
                
                # Update the full latent for next timestep
                full_noise_pred = full_noise_pred / (full_weight_mask + 1e-8)
    
                # Now do a single scheduler step with the full blended noise prediction
                full_latent = sample_scheduler.step(
                    full_noise_pred.unsqueeze(0), t, full_latent.unsqueeze(0),
                    return_dict=False, generator=seed_g)[0].squeeze(0)
                
                # Optional: save intermediate results
                if step_idx % 10 == 0:
                    print(f"Latent stats at step {step_idx}: mean={full_latent.mean():.4f}, std={full_latent.std():.4f}")
                
            # Final latent is ready
            print("\nDenoising complete!")
            
            # Decode the final video
           
                        # Decode in chunks to save memory
            final_latent = full_latent.to(torch.device('cuda:3'))
            chunk_size = 25  # Adjust based on available memory
            print('final latent shape',final_latent.shape)
            video = self.vae.decode([final_latent])[0]
            
            # if final_latent.shape[1] > chunk_size:
            #     print("Decoding video in chunks...")
            #     videos = []
            #     for i in range(0, final_latent.shape[1], chunk_size):
            #         chunk = final_latent[:, i:i+chunk_size]
            #         video_chunk = self.vae.decode([chunk])[0].cpu()
            #         videos.append(video_chunk)
            #         torch.cuda.empty_cache()
                
            #     final_video = torch.cat(videos, dim=1)
            # else:
            #     final_video = self.vae.decode([final_latent])[0]
            
            # # Trim to exact frame count
            # final_video = final_video[:, :total_frames]
            
            # return final_video
            return video[0] if self.rank == 0 else None
            
           