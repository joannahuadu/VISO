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
from .yolo_bricks import MaxSigmoidCSPLayerWithTwoConv, RepConvMaxSigmoidCSPLayerWithTwoConv

# import time
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
                 use_einsum: bool = True,
                 is_sparse: int = 1) -> None:
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
        self.is_sparse = is_sparse

    def forward(self, x: Tensor, guide: Tensor) -> Tensor:
        """Forward process."""
        # B, _, H, W = x.features.shape
        B = 1
        guide = self.guide_fc(guide)
        guide = guide.reshape(B, -1, self.num_heads, self.head_channels)
        embed = self.embed_conv(x) if self.embed_conv is not None else x
        if self.is_sparse:
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
        if self.is_sparse:
            try:
                attn_weight = attn_weight[:,:,x.indices[:,1],x.indices[:,2]].squeeze(0).permute(1, 0)
            except Exception as e:
                attn_weight = attn_weight[:,:,x.indices[:,1].long(),x.indices[:,2].long()].squeeze(0).permute(1, 0)
            x = x.replace_feature((x.features.reshape(len(x.indices), self.num_heads, -1) * attn_weight.unsqueeze(2)).reshape(len(x.indices), -1))
        else:
            x = x.reshape(B, self.num_heads, -1, H, W)
            x = x * attn_weight.unsqueeze(2)
            x = x.reshape(B, -1, H, W)
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
                act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                use_einsum: bool = True,
                is_sparse: int = 1,
                sp_type: str = "vspconv",
                *args, **kwargs) -> None:
            # bn_converted: bool = False
        super().__init__(guide_channels=guide_channels,
                        embed_channels=embed_channels,
                        num_heads=num_heads,
                        with_scale=with_scale,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
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
                                                    use_einsum=use_einsum,
                                                    is_sparse=is_sparse)
        # self.sparse_module_list = [self.main_conv, self.blocks, self.attn_block, self.final_conv]
        self.is_sparse = is_sparse
        self.sp_type = sp_type
        self.sparse_module_name = ['main_conv', 'blocks', 'attn_block', 'final_conv']
        self.sparse_module_list = [getattr(self, name) for name in self.sparse_module_name]
        # self.bn_converted = bn_converted
           
    def forward(self, x: Tensor, guide: Tensor) -> Tensor:
        """Forward process."""
        if self.is_sparse:
            # sp_infer = SPInfer(self.sp_type)
            # for name, m in zip(self.sparse_module_name, self.sparse_module_list):
            #     sp_infer._replace_spinfer(name, m, self)
            x_main = self.main_conv(x)
            x_main_ = list(x_main.features.split((self.mid_channels, self.mid_channels), 1))
            # start_event = torch.cuda.Event(enable_timing=True)
            # end_event = torch.cuda.Event(enable_timing=True)
            # start_event.record()
            # start_event = time.time()
            x_main_.extend(blocks(x_main.replace_feature(x_main_[-1])).features for blocks in self.blocks)
            # end_event.record()
            # end_event.synchronize()
            # elapsed_time_ms = start_event.elapsed_time(end_event)
            # print(f"with sparse: {elapsed_time_ms} milliseconds")
            # print(f"with sparse: {time.time()-start_event} milliseconds")
            x_main_.append(self.attn_block(x_main.replace_feature(x_main_[-1]), guide).features)
            return self.final_conv(x_main.replace_feature(torch.cat(x_main_, 1)))
        else:
            x_main = self.main_conv(x)
            x_main = list(x_main.split((self.mid_channels, self.mid_channels), 1))
            # start_event = torch.cuda.Event(enable_timing=True)
            # end_event = torch.cuda.Event(enable_timing=True)
            # start_event.record()
            # start_event = time.time()
            x_main.extend(blocks(x_main[-1]) for blocks in self.blocks)
            # end_event.record()
            # end_event.synchronize()
            # elapsed_time_ms = start_event.elapsed_time(end_event)
            # print(f"without sparse: {elapsed_time_ms} milliseconds")
            # print(f"without sparse: {time.time()-start_event} milliseconds")
            x_main.append(self.attn_block(x_main[-1], guide))
            return self.final_conv(torch.cat(x_main, 1))   

@MODELS.register_module()
class RepConvMaxSigmoidCSPLayerWithTwoConvSPInfer(RepConvMaxSigmoidCSPLayerWithTwoConv):
    """Sigmoid-attention based CSP layer with two convolution layers."""

    def __init__(self,
                guide_channels: int,
                embed_channels: int,
                num_heads: int = 1,
                with_scale: bool = False,
                conv_cfg: OptConfigType = None,
                norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
                use_einsum: bool = True,
                is_sparse: int = 1,
                sp_type: str = "vspconv",
                *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.attn_block = RepConvMaxSigmoidAttnBlockSPInfer(
            self.mid_channels,
            self.mid_channels,
            embed_channels=embed_channels,
            guide_channels=guide_channels,
            num_heads=num_heads,
            with_scale=with_scale,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            use_einsum=use_einsum,
            is_sparse=is_sparse)
        
        self.is_sparse = is_sparse
        self.sp_type = sp_type
        self.sparse_module_name = ['main_conv', 'blocks', 'attn_block', 'final_conv']
        self.sparse_module_list = [getattr(self, name) for name in self.sparse_module_name]

    def forward(self, x: Tensor, guide: Tensor) -> Tensor:
        """Forward process."""
        if self.is_sparse:
            x_main = self.main_conv(x)
            x_main_ = list(x_main.features.split((self.mid_channels, self.mid_channels), 1))
            x_main_.extend(blocks(x_main.replace_feature(x_main_[-1])).features for blocks in self.blocks)
            x_main_.append(self.attn_block(x_main.replace_feature(x_main_[-1]), guide).features)
            return self.final_conv(x_main.replace_feature(torch.cat(x_main_, 1)))
        else:
            x_main = self.main_conv(x)
            x_main = list(x_main.split((self.mid_channels, self.mid_channels), 1))
            x_main.extend(blocks(x_main[-1]) for blocks in self.blocks)
            x_main.append(self.attn_block(x_main[-1], guide))
            return self.final_conv(torch.cat(x_main, 1))

@MODELS.register_module()
class RepConvMaxSigmoidAttnBlockSPInfer(BaseModule):
    """Max Sigmoid attention block."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 embed_channels: int,
                 guide_channels: int,
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
                 use_einsum: bool = True,
                 is_sparse: int = 1) -> None:
        super().__init__(init_cfg=init_cfg)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        assert (out_channels % num_heads == 0 and
                embed_channels % num_heads == 0), \
            'out_channels and embed_channels should be divisible by num_heads.'
        self.num_heads = num_heads
        self.head_channels = out_channels // num_heads
        self.use_einsum = use_einsum

        self.embed_conv = ConvModule(
            in_channels,
            embed_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None) if embed_channels != in_channels else None
        self.bias = nn.Parameter(torch.zeros(num_heads))
        self.num_heads = num_heads
        self.split_channels = embed_channels // num_heads
        self.guide_convs = nn.ModuleList(
            nn.Conv2d(self.split_channels, guide_channels, 1, bias=False)
            for _ in range(num_heads))
        self.project_conv = conv(in_channels,
                                 out_channels,
                                 kernel_size,
                                 stride=1,
                                 padding=padding,
                                 conv_cfg=conv_cfg,
                                 norm_cfg=norm_cfg,
                                 act_cfg=None)
        self.is_sparse = is_sparse

    def forward(self, x: Tensor, txt_feats: Tensor = None) -> Tensor:
        """Forward process."""
        # B, C, H, W = x.shape
        B = 1
        embed = self.embed_conv(x) if self.embed_conv is not None else x
        embed = list(embed.split(self.split_channels, 1))
        # Bx(MxN)xHxW (H*c=C, H: heads)
        if self.is_sparse:
            attn_weight = torch.cat(
                [conv(x).dense(channels_first=True) for conv, x in zip(self.guide_convs, embed)], dim=1)
        else:
            attn_weight = torch.cat(
                [conv(x) for conv, x in zip(self.guide_convs, embed)], dim=1)
        _, _, H, W = attn_weight.shape
        # BxMxNxHxW
        attn_weight = attn_weight.view(B, self.num_heads, -1, H, W)
        # attn_weight = torch.stack(
        #     [conv(x) for conv, x in zip(self.guide_convs, embed)])
        # BxMxNxHxW -> BxMxHxW
        attn_weight = attn_weight.max(dim=2)[0] / (self.head_channels**0.5)
        attn_weight = (attn_weight + self.bias.view(1, -1, 1, 1)).sigmoid()
        # .transpose(0, 1)
        # BxMx1xHxW
        attn_weight = attn_weight[:, :, None]
        x = self.project_conv(x)
        if self.is_sparse:
            try:
                attn_weight = attn_weight[:,:,x.indices[:,1],x.indices[:,2]].squeeze(0).permute(1, 0)
            except Exception as e:
                attn_weight = attn_weight[:,:,x.indices[:,1].long(),x.indices[:,2].long()].squeeze(0).permute(1, 0)
            x = x.replace_feature((x.features.reshape(len(x.indices), self.num_heads, -1) * attn_weight.unsqueeze(2)).reshape(len(x.indices), -1))
        else:
            # BxHxCxHxW
            x = x.view(B, self.num_heads, -1, H, W)
            x = x * attn_weight
            x = x.view(B, -1, H, W)
        return x

@MODELS.register_module()
class DownSampleConvSP(BaseModule):
    def __init__(self, 
                in_channels: int,
                out_channels: int,
                init_cfg: OptMultiConfig = None,
                norm_cfg: ConfigType = dict(type='BN',
                    momentum=0.03,
                    eps=0.001),
                act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.conv = ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        
    def forward(self, x: Tensor):
        return self.conv(x)

@MODELS.register_module()
class DownSampleConvSPInfer(BaseModule):
    def __init__(self, 
                in_channels: int,
                out_channels: int,
                init_cfg: OptMultiConfig = None,
                norm_cfg: ConfigType = dict(type='BN',
                    momentum=0.03,
                    eps=0.001),
                act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                is_sparse: int = 1,
                sp_type: str = "spconv",
                ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.conv = ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.is_sparse = is_sparse
        self.sp_type = sp_type
        self.sparse_module_name = ['conv']
        self.sparse_module_list = [getattr(self, name) for name in self.sparse_module_name]
        
    def forward(self, x: Tensor):
        # if self.is_sparse:
        #     sp_infer = SPInfer(self.sp_type)
        #     for name, m in zip(self.sparse_module_name, self.sparse_module_list):
        #         sp_infer._replace_spinfer(name, m, self)
        return self.conv(x)

@MODELS.register_module()
class TextKnowledgeAttnBlock(KnowledgeAttnBlock):
    """Max Sigmoid attention block."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

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
        
        assert self.num_heads == 1
        attn_weight = attn_weight.squeeze(1)
        attn_weight = attn_weight / (self.head_channels**0.5)
        attn_weight = attn_weight + self.bias[None, :, None, None]
        attn_weight = attn_weight.sigmoid() * self.scale

        return x, attn_weight