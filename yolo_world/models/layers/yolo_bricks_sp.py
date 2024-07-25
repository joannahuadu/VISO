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
class MaxSigmoidAttnBlockSPInfer(BaseModule):
    """Max Sigmoid attention block."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 guide_channels: int,
                 embed_channels: int,
                 kernel_size: int = 3,
                 padding: int = 1,
                 num_heads: int = 1,
                 use_depthwise: bool = False,
                 with_scale: bool = False,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN',
                                             momentum=0.03,
                                             eps=0.001),
                 init_cfg: OptMultiConfig = None,
                 use_einsum: bool = True) -> None:
        super().__init__(init_cfg=init_cfg)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        assert (out_channels % num_heads == 0 and
                embed_channels % num_heads == 0), \
            'out_channels and embed_channels should be divisible by num_heads.'
        self.num_heads = num_heads
        self.head_channels = embed_channels // num_heads
        self.use_einsum = use_einsum

        self.embed_conv = ConvModule(
            in_channels,
            embed_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None) if embed_channels != in_channels else None
        self.guide_fc = Linear(guide_channels, embed_channels)
        self.bias = nn.Parameter(torch.zeros(num_heads))
        if with_scale:
            self.scale = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        else:
            self.scale = 1.0

        self.project_conv = conv(in_channels,
                                 out_channels,
                                 kernel_size,
                                 stride=1,
                                 padding=padding,
                                 conv_cfg=conv_cfg,
                                 norm_cfg=norm_cfg,
                                 act_cfg=None)

    def forward(self, x: Tensor, guide: Tensor) -> Tensor:
        """Forward process."""
        # B, _, H, W = x.features.shape
        B = 1
        guide = self.guide_fc(guide)
        guide = guide.reshape(B, -1, self.num_heads, self.head_channels)
        embed = self.embed_conv(x) if self.embed_conv is not None else x
        embed = embed.dense(channels_first=True)
        _, _, H, W = embed.shape
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

        x = self.project_conv(x)
        x = x.replace_feature(x.features.reshape(B, self.num_heads, -1, H, W))
        x = x.replace_feature(x.features * attn_weight.unsqueeze(2))
        x = x.replace_feature(x.features.reshape(B, -1, H, W))
        return x

@MODELS.register_module()
class KnowledgeAttnBlock(BaseModule):
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

    def __init__(self, 
                guide_channels: int,
                embed_channels: int,
                num_heads: int = 1,
                with_scale: bool = False,
                conv_cfg: OptConfigType = None,
                norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
                use_einsum: bool = True,
                *args, **kwargs) -> None:
            # bn_converted: bool = False
        super().__init__(guide_channels=guide_channels,
                        embed_channels=embed_channels,
                        num_heads=num_heads,
                        with_scale=with_scale,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        use_einsum=use_einsum,
                        *args, **kwargs)
        
        self.attn_block = MaxSigmoidAttnBlockSPInfer(self.mid_channels,
                                                    self.mid_channels,
                                                    guide_channels=guide_channels,
                                                    embed_channels=embed_channels,
                                                    num_heads=num_heads,
                                                    with_scale=with_scale,
                                                    conv_cfg=conv_cfg,
                                                    norm_cfg=norm_cfg,
                                                    use_einsum=use_einsum)
        # self.sparse_module_list = [self.main_conv, self.blocks, self.attn_block, self.final_conv]
        self.sparse_module_name = ['main_conv', 'blocks', 'attn_block', 'final_conv']
        self.sparse_module_list = [getattr(self, name) for name in self.sparse_module_name]
        # self.bn_converted = bn_converted
           
    def forward(self, x: Tensor, guide: Tensor) -> Tensor:
        """Forward process."""
        for name, m in zip(self.sparse_module_name, self.sparse_module_list):
            self._replace_spinfer(name, m, self)
        x_main = self.main_conv(x)
        x_main_ = list(x_main.features.split((self.mid_channels, self.mid_channels), 1))
        x_main_.extend(blocks(x_main.replace_feature(x_main_[-1])).features for blocks in self.blocks)
        x_main_.append(self.attn_block(x_main.replace_feature(x_main_[-1]), guide).features)
        return self.final_conv(x_main.replace_feature(torch.cat(x_main_, 1)))

    def _replace_spinfer(self, name, module, parent) -> Tensor:
        if isinstance(module, ConvModule):
            weights, biases = self.get_params(module)
            if hasattr(module, 'activate'):
                act = module.activate
            else:
                act = None
            setattr(parent, name, self._make_spconv(weights, biases, act)) # TODO: _make_spconv -> _make_conv
            return
        elif isinstance(module, nn.Conv2d):
            weights, biases = self.get_params(module)
            setattr(parent, name, self._make_spconv(weights, biases)) # TODO: _make_spconv -> _make_conv
            return
        else:
            for name, child in module.named_children():
                self._replace_spinfer(name, child, module)
    
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
        setattr(module, 'weight', module.conv.weight.data * bn_scale.view(-1, 1, 1, 1))
        setattr(module, 'bias',  torch.nn.Parameter(bn_bias))
        # self.bn_converted = True
        
    def _make_spconv(self, weights, biases, act=None):
        nets = []
        in_channel  = weights.shape[1]
        out_channel = weights.shape[0]
        k_size      = weights.shape[2]
        filter = spconv.SubMConv2d(in_channel, out_channel, k_size, 1, padding=k_size//2, indice_key="asd", algo=spconv.ConvAlgo.Native).to(device=weights.device)
        filter.weight.data[:] = weights.permute(0,2,3,1).contiguous()[:] # transpose(1,2).transpose(0,1).transpose(2,3).transpose(1,2).transpose(2,3)
        filter.bias.data   = biases
        nets.append(filter)
        if not act == None:
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