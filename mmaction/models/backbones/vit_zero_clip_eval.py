from collections import OrderedDict
from typing import Tuple, Union
from mmcv.cnn.bricks import DropPath
from mmengine.model.weight_init import trunc_normal_
from timm.models.layers import to_2tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import clip
from mmengine.logging import MMLogger
from einops import rearrange
from mmaction.registry import MODELS
from ..common import STDHA


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock_Zero(nn.Module):
    def __init__(self, d_model: int, n_head: int, num_frames=8, stdha_cfg=dict()):
        super().__init__()
        self.n_head = n_head
        self.num_frames = num_frames
        self.stdha_cfg = stdha_cfg

        self.attn = STDHA(embed_dim=d_model, num_heads=n_head, 
                num_frames=num_frames, shift_div=self.stdha_cfg.get('shift_div', 4),
                 divide_head=self.stdha_cfg.get('divide_head', False), shift_stride=self.stdha_cfg.get('shift_stride', 1))
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)

    def attention(self, x: torch.Tensor):
        return self.attn(x, x, x, need_weights=False, attn_mask=None)[0]

    def forward(self, x: torch.Tensor):
        # x shape [HW+1, BT, D]
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x

    
class Transformer(nn.Module):
    def __init__(self, num_frames, width: int, layers: int, heads: int, stdha_cfg=dict()):
        super().__init__()
        self.width = width
        self.layers = layers

        self.resblocks = nn.Sequential(*[ResidualAttentionBlock_Zero(width, heads, num_frames, stdha_cfg=stdha_cfg) for i in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)




@MODELS.register_module()
class ViT_Zero_CLIP_eval(nn.Module):
    ## ViT definition in CLIP image encoder
    def __init__(self, input_resolution: int, num_frames: int, patch_size: int, width: int, layers: int, heads: int, 
                stdha_cfg=dict(), freeze_all=False):
        super().__init__()
        self.input_resolution = input_resolution
        self.freeze_all = freeze_all
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.layers = layers
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.num_frames = num_frames
        self.temporal_embedding = nn.Parameter(torch.zeros(1, num_frames, width))

        self.transformer = Transformer(num_frames, width, layers, heads, stdha_cfg=stdha_cfg)
       
        self.ln_post = LayerNorm(width)

        self.freeze_all = freeze_all
 

    def init_weights(self):
        if self.freeze_all:
            ## freeze all parameters NOTE 不知道冻结参数会不会影响速度，理论上不会
            for name, param in self.named_parameters():
                param.requires_grad = False

            for name, param in self.named_parameters():
                print('{}: {}'.format(name, param.requires_grad))
            num_param = sum(p.numel() for p in self.parameters() if p.requires_grad)
            num_total_param = sum(p.numel() for p in self.parameters())
            print('Number of total parameters: {}, tunable parameters: {}'.format(num_total_param, num_param))
            assert num_param == 0
    

    def forward(self, x: torch.Tensor):
        B, C, T, H, W = x.shape
        assert T == self.num_frames, f"{T} != {self.num_frames}"
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.conv1(x)  
        x = x.reshape(x.shape[0], x.shape[1], -1) 
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)

        n = x.shape[1]
        x = rearrange(x, '(b t) n d -> (b n) t d', t=self.num_frames)
        x = x + self.temporal_embedding
        x = rearrange(x, '(b n) t d -> (b t) n d', n=n)
            
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x[:, 0] # 取cls token ND

        x = self.ln_post(x)

        x = rearrange(x, '(b t) d -> b d t',b=B,t=T)
        x = x.unsqueeze(-1).unsqueeze(-1)  # BDTHW for I3D head

        return x
