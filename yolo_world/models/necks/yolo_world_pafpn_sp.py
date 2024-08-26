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
from ...utils.mask_vis import mask_visulize, featuremap_visulize
from yolo_world.models.sputils import _make_indice_tensor, _concat
@MODELS.register_module()
class YOLOWorldPAFPNSP(YOLOWorldPAFPN):
    """Path Aggregation Network with sparse convolution used in YOLO World
    Following YOLOv8 PAFPN, including text to image fusion, including text-guided masked image features., including forward with sparse convolution
    """
    def __init__(self,
                #  reduce_embed_channels: List[int],
                 reduce_num_heads: List[int],
                 reduce_block_cfg: ConfigType = dict(type='KnowledgeAttnBlock'),
                #  downsample_block_cfg: ConfigType = dict(type='DownSampleConvSP'),
                 *args, **kwargs) -> None:
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
    #     downsample_block_cfg = copy.deepcopy(self.downsample_block_cfg)
    #     downsample_block_cfg.update(in_channels=make_divisible(
    #                                 self.in_channels[idx], self.widen_factor),
    #                                 out_channels=make_divisible(
    #                                 self.in_channels[idx], self.widen_factor),
    #                                 norm_cfg=self.norm_cfg,
    #                                 act_cfg=self.act_cfg)
    #     return MODELS.build(downsample_block_cfg)
        

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
        return MODELS.build(reduce_block_cfg)

    def forward(self, img_feats: List[Tensor], txt_feats: Tensor = None) -> tuple:
        """Forward function.
        including multi-level image features, text features: BxLxD
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
class YOLOWorldPAFPNSPInfer(YOLOWorldPAFPN):
    """Path Aggregation Network with sparse convolution used in YOLO World
    Following YOLOv8 PAFPN, including text to image fusion, including text-guided masked image features., including forward with sparse convolution
    """
    def __init__(self,
                 reduce_num_heads: List[int],
                 reduce_block_cfg: ConfigType = dict(type='KnowledgeAttnBlock'),
                 is_split_attn: bool = False,
                 is_sparse_levels: List[int] = [1,1,0],
                 mask_vis: bool = False,
                 score_th: float = 0.501,
                #  downsample_block_cfg: ConfigType = dict(type='DownSampleConvSPInfer'),
                 *args, **kwargs) -> None:
        self.reduce_num_heads = reduce_num_heads
        self.reduce_block_cfg = reduce_block_cfg
        self.is_sparse_levels = is_sparse_levels
        # self.downsample_block_cfg = downsample_block_cfg
        super().__init__(*args, **kwargs)
        assert len(self.is_sparse_levels) == len(self.in_channels)
        self.score_th = score_th
        self.sp_module = ['top_down_layers', 'bottom_up_layers']
        self.mask_vis = mask_vis
        self.is_split_attn = is_split_attn

    def build_top_down_layer(self, idx: int) -> nn.Module:
        """build top down layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The top down layer.
        """
        self.block_cfg.update(dict(is_sparse = self.is_sparse_levels[idx - 1]))
        return super().build_top_down_layer(idx)

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        """build bottom up layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The bottom up layer.
        """
        self.block_cfg.update(dict(is_sparse = self.is_sparse_levels[idx + 1]))
        return super().build_bottom_up_layer(idx)
    
    # def build_downsample_layer(self, idx: int) -> nn.Module:
    #     """build downsample layer.

    #     Args:
    #         idx (int): layer idx.

    #     Returns:
    #         nn.Module: The downsample layer.
    #     """
    #     downsample_block_cfg = copy.deepcopy(self.downsample_block_cfg)
    #     downsample_block_cfg.update(in_channels=make_divisible(
    #                                 self.in_channels[idx], self.widen_factor),
    #                                 out_channels=make_divisible(
    #                                 self.in_channels[idx], self.widen_factor),
    #                                 norm_cfg=self.norm_cfg,
    #                                 act_cfg=self.act_cfg)
    #     return MODELS.build(downsample_block_cfg)
    
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
        return MODELS.build(reduce_block_cfg)

    def _sparse_indices(self, feature_value, attn_preds):
        N, _, qh, qw = attn_preds.shape
        assert N==1
        assert feature_value.shape[2:] == attn_preds.shape[2:], f"Shapes do not match: {feature_value.shape[2:]} != {attn_preds.shape[2:]}"
        sparse_inds = torch.where(attn_preds.view(-1) > self.score_th)[0]
        sparse_y = torch.div(sparse_inds, qw).int()
        sparse_x = torch.remainder(sparse_inds, qw).int()
        return torch.stack((torch.zeros_like(sparse_y), sparse_y, sparse_x), dim=-1)
    
    def forward(self, img_feats: List[Tensor], txt_feats: Tensor = None) -> tuple:
        """Forward function.
        including multi-level image features, text features: BxLxD
        """
        assert len(img_feats) == len(self.in_channels)
                
        # reduce layers
        reduce_outs = []
        masks_all_levels = []
        for idx in range(len(self.in_channels)):
            x, attn = self.reduce_layers[idx](img_feats[idx], txt_feats)
            if self.is_split_attn:
                masks_all_levels.append(attn.permute(0, 3, 1, 2))
                attn = attn.max(dim=-1)[0].unsqueeze(1)
            reduce_outs.append((x, attn))
        if self.mask_vis:
            if self.is_split_attn:
                mask_visulize(masks_all_levels)
            mask_visulize([attn_weight for _, attn_weight in reduce_outs])
            featuremap_visulize([feature_value for feature_value, _ in reduce_outs])
            
        base_attns = []
        for idx in range(len(self.in_channels)):
            base_attns.append(self._sparse_indices(*reduce_outs[idx]))
            
        # top-down path
        inner_outs = [reduce_outs[-1][0]]
        inner_attns = [base_attns[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            inner_attn = inner_attns[0]
            feat_low = _make_indice_tensor(
                            reduce_outs[idx - 1][0], 
                            base_attns[idx - 1])
            upsample_feat = _make_indice_tensor(
                                    self.upsample_layers[len(self.in_channels) - 1 - 
                                                         idx](feat_high), 
                                    inner_attn, 
                                    project='up')
            if self.upsample_feats_cat_first:
                top_down_layer_inputs, inner_attn = _concat(upsample_feat, feat_low, self.is_sparse_levels[idx - 1])
                # top_down_layer_inputs = torch.cat([upsample_feat, feat_low], 1)
            else:
                top_down_layer_inputs, inner_attn = _concat(feat_low, upsample_feat, self.is_sparse_levels[idx - 1])
                # top_down_layer_inputs = torch.cat([feat_low, upsample_feat], 1)
            if inner_attn.shape[0] == 0 and self.is_sparse_levels[idx - 1]==1:
                return None
            inner_out = self.top_down_layers[len(self.in_channels) - 1 - idx](
                top_down_layer_inputs, txt_feats)
            if self.is_sparse_levels[idx - 1]:
                inner_outs.insert(0, inner_out.dense(channels_first=True))
            else:
                inner_outs.insert(0, torch.tensor(inner_out))
            inner_attns.insert(0, inner_attn)
        
        # bottom-up path
        outs = [inner_outs[0]]
        out_attns = [inner_attns[0]]
        for idx in range(len(self.in_channels) - 1):
            out_attn = out_attns[-1]
            feat_low = outs[-1]
            feat_high = _make_indice_tensor(inner_outs[idx + 1], inner_attns[idx + 1])
            downsample_feat = _make_indice_tensor(self.downsample_layers[idx](feat_low), out_attn, project='down')
            bottom_up_layer_inputs, out_attn = _concat(downsample_feat, feat_high, self.is_sparse_levels[idx + 1])
            if out_attn.shape[0] == 0 and self.is_sparse_levels[idx + 1]==1:
                return None
            out = self.bottom_up_layers[idx](bottom_up_layer_inputs, txt_feats)
            if self.is_sparse_levels[idx + 1]:
                outs.append(out.dense(channels_first=True))
            else:
                outs.append(torch.tensor(out))
            out_attns.append(out_attn)
        
        # out_layers
        results = []
        for idx in range(len(self.in_channels)):
            results.append(self.out_layers[idx](outs[idx]))
        
        for idx in range(len(self.in_channels)):
            results.append(out_attns[idx])
            
        return tuple(results)
    
@MODELS.register_module()
class YOLOWorldPAFPNUTMSP(YOLOWorldPAFPNSP):
    """
    """
    def __init__(self,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, img_feats: List[Tensor], txt_feats: Tensor = None) -> tuple:
        """Forward function.
        including multi-level image features, text features: BxLxD
        """
        assert len(img_feats) == len(self.in_channels)
        # reduce layers
        reduce_outs = []
        for idx in range(len(self.in_channels)):
            reduce_outs.append(self.reduce_layers[idx](img_feats[idx], txt_feats))

        # add reduce_outs_attn
        results = []
        for idx in range(len(self.in_channels)):
            results.append(reduce_outs[idx][1])
            
        return tuple(results)