# Copyright (c) Tencent Inc. All rights reserved.
import copy
from typing import List, Union

import torch
import torch.nn as nn
from torch import Tensor
from mmdet.utils import ConfigType, OptMultiConfig

from mmyolo.registry import MODELS
from mmyolo.models.utils import make_divisible, make_round
from mmyolo.models.necks.yolov8_pafpn import YOLOv8PAFPN
from .yolo_world_pafpn import YOLOWorldPAFPN

@MODELS.register_module()
class YOLOWorldPAFPNSP(YOLOWorldPAFPN):
    """Path Aggregation Network with sparse convolution used in YOLO World
    Following YOLOv8 PAFPN, including text to image fusion, including text-guided masked image features., including forward with sparse convolution
    """
    def __init__(self,
                 reduce_embed_channels: List[int],
                 reduce_num_heads: List[int],
                 reduce_block_cfg: ConfigType = dict(type='MaxSigmoidAttnBlock'),
                 *args, **kwargs) -> None:
        self.reduce_embed_channels = reduce_embed_channels
        self.reduce_num_heads = reduce_num_heads
        self.reduce_block_cfg = reduce_block_cfg
        super().__init__(*args, **kwargs)



    def build_reduce_layer(self, idx: int) -> nn.Module:
        """build reduce layer.
        Generate text-guided masked image features.
        
        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The reduce layer.
        
        """
        reduce_block_cfg = copy.deepcopy(self.reduce_block_cfg)
        reduce_block_cfg.update(in_channels=make_divisible(
                                self.in_channels[idx], self.widen_factor),
                                out_channels=make_divisible(
                                self.in_channels[idx], self.widen_factor),
                                guide_channels=self.guide_channels,
                                embed_channels=make_round(self.reduce_embed_channels[idx],
                                                        self.widen_factor),
                                num_heads=make_round(self.reduce_num_heads[idx],
                                                    self.widen_factor),
                                norm_cfg=self.norm_cfg,
                                )
        return MODELS.build(reduce_block_cfg)

    def forward(self, img_feats: List[Tensor], txt_feats: Tensor = None) -> tuple:
        """Forward function.
        including multi-level image features, text features: BxLxD
        """
        assert len(img_feats) == len(self.in_channels)
        # reduce layers
        reduce_outs = []
        for idx in range(len(self.in_channels)):
            reduce_outs.append(self.reduce_layers[idx](img_feats[idx], txt_feats, attn=True))
            
        # top-down path
        inner_outs = [reduce_outs[-1][0]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = reduce_outs[idx - 1][0]
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

        # add reduce_outs_attn
        for idx in range(len(self.in_channels)):
            results.append(reduce_outs[idx][1])
            
        return tuple(results)
