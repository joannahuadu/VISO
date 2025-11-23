# Copyright (c) Tencent Inc. All rights reserved.
import copy
from typing import List, Union

import torch
import torch.nn as nn
from torch import Tensor
from mmdet.utils import ConfigType, OptMultiConfig

from mmyolo.registry import MODELS
from mmyolo.models.utils import make_divisible, make_round
from mmyolo.models.necks.yolov5_pafpn import YOLOv5PAFPN
from ...utils.mask_vis import featuremap_visulize
from mmcv.cnn import ConvModule

class TopDownLayer_idx_not_equal_one(nn.Module):
    def __init__(self, block_cfg, conv_module):
        super(TopDownLayer_idx_not_equal_one, self).__init__()
        self.block = MODELS.build(block_cfg)
        self.conv = conv_module

    def forward(self, x, guide):
        x = self.block(x, guide)
        x = self.conv(x)
        return x


@MODELS.register_module()
class YOLOv5WorldPAFPN(YOLOv5PAFPN):
    """Path Aggregation Network used in YOLOv5 World
    Following YOLOv5 PAFPN, including text to image fusion
    """
    def __init__(self,
                 in_channels: List[int],
                 out_channels: Union[List[int], int],
                 guide_channels: int,
                 embed_channels: List[int],
                 num_heads: List[int],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 1,
                 freeze_all: bool = False,
                 block_cfg: ConfigType = dict(type='CSPLayer'), #yolov5是CSPLayer (without two conv)
                 norm_cfg: ConfigType = dict(type='BN',
                                             momentum=0.03,
                                             eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None,
                 mask_vis: bool = False,
                 ) -> None:
        self.guide_channels = guide_channels
        self.embed_channels = embed_channels
        self.num_heads = num_heads
        self.block_cfg = block_cfg
        self.mask_vis = mask_vis
        self.num_csp_blocks = num_csp_blocks
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            freeze_all=freeze_all,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)
    def build_top_down_layer(self, idx: int) -> nn.Module:
        """build top down layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The top down layer.
            yolov5如果
        """
        block_cfg = copy.deepcopy(self.block_cfg)
        block_cfg.update(
            dict(
                in_channels=make_divisible(self.in_channels[idx - 1] * 2, 
                                            self.widen_factor),
                out_channels=make_divisible(self.in_channels[idx - 1], self.widen_factor),
                guide_channels=self.guide_channels,
                embed_channels=make_round(self.embed_channels[idx - 1],
                                            self.widen_factor),
                num_heads = make_round(self.num_heads[idx - 1],
                                            self.widen_factor),
                num_blocks = make_round(self.num_csp_blocks,
                                            self.deepen_factor),
                add_identity=False,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))

        conv_module = ConvModule(
            make_divisible(self.in_channels[idx - 1],
                        self.widen_factor),
            make_divisible(self.in_channels[idx - 2],
                        self.widen_factor),
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )

        if idx == 1:
            return MODELS.build(block_cfg)
        else:
            return TopDownLayer_idx_not_equal_one(block_cfg, conv_module)



    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        """build bottom up layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The bottom up layer.
        """
        block_cfg = copy.deepcopy(self.block_cfg)
        block_cfg.update(
            dict(in_channels=make_divisible(self.in_channels[idx] * 2, self.widen_factor),
                 out_channels=make_divisible(self.in_channels[idx + 1], self.widen_factor),
                 guide_channels=self.guide_channels,
                 embed_channels=make_round(self.embed_channels[idx + 1],
                                           self.widen_factor),
                 num_heads=make_round(self.num_heads[idx + 1],
                                      self.widen_factor),
                 num_blocks=make_round(self.num_csp_blocks,
                                       self.deepen_factor),
                 add_identity=False,
                 norm_cfg=self.norm_cfg,
                 act_cfg=self.act_cfg))
        return MODELS.build(block_cfg)

    def forward(self, img_feats: List[Tensor], txt_feats: Tensor = None) -> tuple:
        """Forward function.
        including multi-level image features, text features: BxLxD
        """
        assert len(img_feats) == len(self.in_channels)
        # reduce layers
        reduce_outs = []
        for idx in range(len(self.in_channels)):
            reduce_outs.append(self.reduce_layers[idx](img_feats[idx]))
        if self.mask_vis:
            featuremap_visulize(reduce_outs)
        # top-down path
        inner_outs = [reduce_outs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = reduce_outs[idx - 1]
            upsample_feat = self.upsample_layers[len(self.in_channels) - 1 -
                                                 idx](feat_high)
            if self.upsample_feats_cat_first:
                top_down_layer_inputs = torch.cat([upsample_feat, feat_low], 1)
            else:
                top_down_layer_inputs = torch.cat([feat_low, upsample_feat], 1)
            inner_out = self.top_down_layers[len(self.in_channels) - 1 - idx](
                top_down_layer_inputs, txt_feats)
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_layers[idx](feat_low)
            out = self.bottom_up_layers[idx](torch.cat(
                [downsample_feat, feat_high], 1), txt_feats)
            outs.append(out)

        # out_layers
        results = []
        for idx in range(len(self.in_channels)):
            results.append(self.out_layers[idx](outs[idx]))

        return tuple(results)

# @MODELS.register_module()
# class YOLOv5WorldDualPAFPN(YOLOv5WorldPAFPN):
#     pass
