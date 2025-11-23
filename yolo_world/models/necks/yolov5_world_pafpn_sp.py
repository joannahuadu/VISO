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
from .yolov5_world_pafpn import YOLOv5WorldPAFPN
from ...utils.mask_vis import mask_visulize, featuremap_visulize
from yolo_world.models.sputils import _make_indice_tensor, _concat
from mmcv.cnn import ConvModule

class ReduceLayer(nn.Module):
    def __init__(self, block_cfg, conv_module):
        super(ReduceLayer, self).__init__()
        self.conv = conv_module
        self.block = MODELS.build(block_cfg)


    def forward(self, x, guide):
        x = self.conv(x)
        x = self.block(x, guide)
        return x

@MODELS.register_module()
class YOLOv5WorldPAFPNSP(YOLOv5WorldPAFPN):
    def __init__(self,
                 # reduce_embed_channels: List[int] = [256, 512, 1024],
                 reduce_num_heads: List[int] = [1, 1, 1],
                 reduce_block_cfg: ConfigType = dict(type='KnowledgeAttnBlock'),
                #  downsample_block_cfg: ConfigType = dict(type='DownSampleConvSP'),
                 *args, **kwargs):
        # self.reduce_embed_channels = reduce_embed_channels
        self.reduce_num_heads = reduce_num_heads
        self.reduce_block_cfg = reduce_block_cfg
        # self.downsample_block_cfg = downsample_block_cfg
        super().__init__(*args, **kwargs)
        
    # def build_downsample_layer(self, idx: int) -> nn.Module:
    #     """build downsample layer.

    #     Args:
    #         idx (int): layer idx.

    #     Returns:
    #         nn.Module: The downsample layer.
    #     """
    #     raise NotImplementedError
    
    def build_reduce_layer(self, idx: int) -> nn.Module:
        """build reduce layer.
        yolov5的reduce_layer跟yolov8的不一样, 不是一个空的nn.Identity()

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The reduce layer.
        """
        # yolov5的reduce_layer:
        if idx == len(self.in_channels) - 1:
            layer = ConvModule(
                make_divisible(self.in_channels[idx], self.widen_factor),
                make_divisible(self.in_channels[idx - 1], self.widen_factor),
                1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            layer = nn.Identity()
        
        # 加上knowledge_attn_block
        reduce_block_cfg = copy.deepcopy(self.reduce_block_cfg)
        reduce_block_cfg.update(in_channels=make_divisible(
                                self.in_channels[idx], self.widen_factor),
                                # out_channels=make_divisible(
                                # self.in_channels[idx], self.widen_factor),
                                guide_channels=self.guide_channels,
                                # embed_channels=make_round(self.reduce_embed_channels[idx],
                                #                         self.widen_factor),
                                embed_channels=make_divisible(
                                self.in_channels[idx], self.widen_factor),
                                num_heads=make_round(self.reduce_num_heads[idx],
                                                    self.widen_factor),
                                )
        reduce_block = MODELS.build(reduce_block_cfg)
        return ReduceLayer(layer, reduce_block) # 封装成ReduceLayer
    

    def forward(self, img_feats: List[Tensor], txt_feats: Tensor = None) -> tuple:
        """Forward function.
        including multi-level image features, text features: BxLxD
        直接copy YOLOWorldPAFPNSP的forward
        """
        assert len(img_feats) == len(self.in_channels)
        # reduce layers
        reduce_outs = []
        for idx in range(len(self.in_channels)):
            reduce_outs.append(self.reduce_layers[idx](img_feats[idx], txt_feats))
            
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



@MODELS.register_module()
class YOLOv5WorldPAFPNSPInfer(YOLOv5WorldPAFPN):
    def __init__(self,
                 reduce_num_heads: List[int] = [1, 1, 1],
                 reduce_block_cfg: ConfigType = dict(type='KnowledgeAttnBlock'),
                 is_sparse_levels: List[int] = [1,1,0],
                 mask_vis: bool = False,
                 score_th: float = 0.501,
                #  downsample_block_cfg: ConfigType = dict(type='DownSampleConvSPInfer'),
                 *args, **kwargs) -> None:
        self.reduce_num_heads = reduce_num_heads
        self.reduce_block_cfg = reduce_block_cfg
        self.is_sparse_levels = is_sparse_levels
        super().__init__(*args, **kwargs)
        assert len(self.is_sparse_levels) == len(self.in_channels)
        self.score_th = score_th
        self.sp_module = ['top_down_layers', 'bottom_up_layers']
        self.mask_vis = mask_vis
        
        raise NotImplementedError
