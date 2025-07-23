# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
from easydict import EasyDict

from .shared_config import wan_shared_cfg

#------------------------ Wan I2V 14B ------------------------#

wan_frampack_vace = EasyDict(__name__='VaceWanModel')
wan_frampack_vace.update(wan_shared_cfg)
wan_frampack_vace.sample_neg_prompt = "镜头晃动，" + wan_frampack_vace.sample_neg_prompt

wan_frampack_vace.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
wan_frampack_vace.t5_tokenizer = 'google/umt5-xxl'

# clip
wan_frampack_vace.clip_model = 'clip_xlm_roberta_vit_h_14'
wan_frampack_vace.clip_dtype = torch.float16
wan_frampack_vace.clip_checkpoint = 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'
wan_frampack_vace.clip_tokenizer = 'xlm-roberta-large'

# vae
wan_frampack_vace.vae_checkpoint = 'Wan2.1_VAE.pth'
wan_frampack_vace.vae_stride = (4, 8, 8)

# transformer
wan_frampack_vace.patch_size = (1, 2, 2)
wan_frampack_vace.dim = 5120
wan_frampack_vace.ffn_dim = 13824
wan_frampack_vace.freq_dim = 256
wan_frampack_vace.num_heads = 40
wan_frampack_vace.num_layers = 40
wan_frampack_vace.window_size = (-1, -1)
wan_frampack_vace.qk_norm = True
wan_frampack_vace.cross_attn_norm = True
wan_frampack_vace.eps = 1e-6
