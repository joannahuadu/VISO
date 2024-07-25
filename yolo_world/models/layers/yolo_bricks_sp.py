# Copyright (c) Tencent Inc. All rights reserved.
from typing import List

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule, Linear
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.model import BaseModule
from mmyolo.registry import MODELS
from mmyolo.models.layers import CSPLayerWithTwoConv
from .yolo_bricks import MaxSigmoidCSPLayerWithTwoConv
import spconv.pytorch as spconv

@MODELS.register_module()
class MaxSigmoidAttnBlockSP(BaseModule):
    """Max Sigmoid attention block."""

    def __init__(self,
                 in_channels: int,
                 guide_channels: int,
                 embed_channels: int,
                 num_heads: int = 1,
                 with_scale: bool = False,
                 init_cfg: OptMultiConfig = None,
                 use_einsum: bool = True) -> None:
        super().__init__(init_cfg=init_cfg)
        
        assert (in_channels == embed_channels), \
            'in_channels, embed_channels and out_channels must be equal.'
        assert (embed_channels % num_heads == 0), \
            'out_channels and embed_channels should be divisible by num_heads.'
        self.num_heads = num_heads
        self.head_channels = embed_channels // num_heads
        self.use_einsum = use_einsum

        self.embed_conv = nn.Identity()
        self.guide_fc = Linear(guide_channels, embed_channels)
        self.bias = nn.Parameter(torch.zeros(num_heads))
        if with_scale:
            self.scale = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        else:
            self.scale = 1.0

    def forward(self, x: Tensor, guide: Tensor) -> Tensor:
        """Forward process."""
        B, _, H, W = x.shape

        guide = self.guide_fc(guide)
        guide = guide.reshape(B, -1, self.num_heads, self.head_channels)
        embed = self.embed_conv(x)
        embed = embed.reshape(B, self.num_heads, self.head_channels, H, W)

        if self.use_einsum:
            attn_weight = torch.einsum('bmchw,bnmc->bmhwn', embed, guide)
        else:
            batch, m, channel, height, width = embed.shape
            _, n, _, _ = guide.shape
            embed = embed.permute(0, 1, 3, 4, 2)
            embed = embed.reshape(batch, m, -1, channel)
            guide = guide.permute(0, 2, 3, 1)
            attn_weight = torch.matmul(embed, guide)
            attn_weight = attn_weight.reshape(batch, m, height, width, n)

        attn_weight = attn_weight.max(dim=-1)[0]
        attn_weight = attn_weight / (self.head_channels**0.5)
        attn_weight = attn_weight + self.bias[None, :, None, None]
        attn_weight = attn_weight.sigmoid() * self.scale
        
        # x = x.reshape(B, self.num_heads, -1, H, W)
        # x = x * attn_weight.unsqueeze(2)
        # x = x.reshape(B, -1, H, W)
        return x, attn_weight

@MODELS.register_module()
class MaxSigmoidCSPLayerWithTwoConvSPInfer(MaxSigmoidCSPLayerWithTwoConv):
    """Sigmoid-attention based CSP layer with two convolution layers."""

    def __init__(self, *args, **kwargs) -> None:
            # bn_converted: bool = False
        super().__init__(*args, **kwargs)

        self.sparse_module_list = [self.main_conv, self.blocks, self.attn_block, self.final_conv]
        # self.bn_converted = bn_converted
        
    def forward(self, x: Tensor, guide: Tensor) -> Tensor:
        """Forward process."""
        for sm in self.sparse_module_list:
            self.replace_qinfer(sm)
        x_main = self.main_conv(x)
        x_main = list(x_main.split((self.mid_channels, self.mid_channels), 1))
        x_main.extend(blocks(x_main[-1]) for blocks in self.blocks)
        x_main.append(self.attn_block(x_main[-1], guide))
        return self.final_conv(torch.cat(x_main, 1))

    def replace_qinfer(self, module) -> Tensor:
        for name, child in module.named_modules():
            if isinstance(child, ConvModule):
                weights, biases = self.get_params(child)
                act = child.activate
                setattr(module, name, self._make_spconv(weights, biases, act)) # TODO: _make_spconv -> _make_conv
            elif isinstance(child, nn.Conv2d):
                weights, biases = self.get_params(child)
                setattr(module, name, self._make_spconv(weights, biases)) # TODO: _make_spconv -> _make_conv
                # forward = getattr(getattr(module, name), 'forward')
                # sp_forward = lambda x, filters=forward: self._run_spconvs(x, filters) # TODO: _run_spconvs -> _run_convs
                # setattr(getattr(module, name), 'forward', sp_forward)
            else:
                self.replace_qinfer(child)
    
    def get_params(self, module) -> Tensor:
        # if not self.bn_converted:
        if isinstance(module, ConvModule):
            self._bn_convert(module)
        
        ws = module.weight.data
        bs = module.bias.data
        return ws, bs

    def _bn_convert(self, module):
        # assert not self.training
        # if self.bn_converted:
        #     return
        running_mean = module.norm.running_mean.data
        running_var = module.norm.running_var.data
        gamma = module.norm.weight.data
        beta = module.norm.bias.data
        bn_scale = gamma * torch.rsqrt(running_var + 1e-10)
        bn_bias  = beta - bn_scale * running_mean
        module.weight.data = module.conv.layer.weight.data * bn_scale.view(-1, 1, 1, 1)
        module.bias.data  = torch.nn.Parameter(bn_bias)
        # self.bn_converted = True
        
    def _make_spconv(self, weights, biases, act=None):
        nets = []
        in_channel  = weights.shape[1]
        out_channel = weights.shape[0]
        k_size      = weights.shape[2]
        filter = spconv.SubMConv2d(in_channel, out_channel, k_size, 1, padding=k_size//2, indice_key="asd", algo=spconv.ConvAlgo.Native).to(device=weights.device)
        filter.weight.data[:] = weights.permute(2,3,1,0).contiguous()[:] # transpose(1,2).transpose(0,1).transpose(2,3).transpose(1,2).transpose(2,3)
        filter.bias.data   = biases
        nets.append(filter)
        if not act is None:
            nets.append(act) ## TODO: Change into SiLU
        return spconv.SparseSequential(*nets)
    
    def _make_conv(self, weights, biases, act=None):
        nets = []
        in_channel  = weights.shape[0]
        out_channel = weights.shape[1]
        k_size      = weights.shape[2]
        filter = nn.Conv2d(in_channel, out_channel, k_size, 1, padding=k_size//2)
        filter.weight.data = weights
        filter.bias.data   = biases
        nets.append(filter)
        if not act is None:
            nets.append(act)
        return torch.nn.Sequential(*nets)
    
    def _run_spconvs(self, x, filters):
        y = filters(x)
        return y.dense(channels_first=False)

    def _run_convs(self, x, filters):
        return filters(x)