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
import spconv.pytorch as spconv

@MODELS.register_module()
class YOLOWorldPAFPNSP(YOLOWorldPAFPN):
    """Path Aggregation Network with sparse convolution used in YOLO World
    Following YOLOv8 PAFPN, including text to image fusion, including text-guided masked image features., including forward with sparse convolution
    """
    def __init__(self,
                #  reduce_embed_channels: List[int],
                 reduce_num_heads: List[int],
                 reduce_block_cfg: ConfigType = dict(type='MaxSigmoidAttnBlockSP'),
                 *args, **kwargs) -> None:
        # self.reduce_embed_channels = reduce_embed_channels
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
class YOLOWorldPAFPNSPInfer(YOLOWorldPAFPNSP):
    """Path Aggregation Network with sparse convolution used in YOLO World
    Following YOLOv8 PAFPN, including text to image fusion, including text-guided masked image features., including forward with sparse convolution
    """
    def __init__(self,
                 score_th: float = 0.501,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.score_th = score_th
    
    def _make_sparse_tensor(self, feature_value, attn_preds, project=False):
        _, fc, fh, fw = feature_value.shape
        N, _, qh, qw = attn_preds.shape
        assert N==1
        if not project:
            assert feature_value.shape[2:] == attn_preds.shape[2:], f"Shapes do not match: {feature_value.shape[2:]} != {attn_preds.shape[2:]}"
            sparse_inds = torch.where(attn_preds.view(-1) > self.score_th)[0]
            sparse_y = torch.div(sparse_inds, qw).int()
            sparse_x = torch.remainder(sparse_inds, qw).int()
        else:
            if attn_preds.ndim == feature_value.ndim:
                pidxs = torch.where(attn_preds.view(-1) > self.score_th)[0]
                y = torch.div(pidxs, qw).int()
                x = torch.remainder(pidxs, qw).int()
            else:
                assert attn_preds.ndim == feature_value.ndim -1, f"ndim do not match: {attn_preds.ndim} != {feature_value.ndim -1}"
                y = attn_preds[:, 1]
                x = attn_preds[:, 2]
            
            sparse_y, sparse_x = [], []
            for i in range(2):
                for j in range(2):
                    sparse_y.append(y * 2 + i)
                    sparse_x.append(x * 2 + j)

            sparse_y = torch.cat(sparse_y, dim=0)
            sparse_x = torch.cat(sparse_x, dim=0)
            # good_idx = (sparse_y >= 0) & (sparse_y < fh) & (sparse_x >= 0)  & (sparse_x < fw)
            # sparse_y = sparse_y[good_idx]
            # sparse_x = sparse_x[good_idx]
            # sparse_yx = torch.stack((sparse_y, sparse_x), dim=0).t()
            # sparse_yx = torch.unique(sparse_yx, sorted=False, dim=0)
            # sparse_y = sparse_yx[:, 0]
            # sparse_x = sparse_yx[:, 1]
            sparse_inds = (sparse_y * fw + sparse_x).long()
            
        sparse_features = feature_value.view(fc, -1).transpose(0, 1)[sparse_inds].view(-1, fc)
        sparse_indices  = torch.stack((torch.zeros_like(sparse_y), sparse_y, sparse_x), dim=-1)  
        sparse_tensor = spconv.SparseConvTensor(sparse_features, sparse_indices.int(), (fh, fw), 1)
        return sparse_tensor
            
            
    def forward(self, img_feats: List[Tensor], txt_feats: Tensor = None) -> tuple:
        """Forward function.
        including multi-level image features, text features: BxLxD
        """
        assert len(img_feats) == len(self.in_channels)
        # reduce layers
        reduce_outs = []
        for idx in range(len(self.in_channels)):
            reduce_outs.append(self.reduce_layers[idx](img_feats[idx], txt_feats, attn=True))
        
        sparse_reduce_outs = []
        for idx in range(len(self.in_channels)):
            x = self._make_sparse_tensor(*reduce_outs[idx])
            sparse_reduce_outs.append(x)
            
        # top-down path
        inner_outs = [reduce_outs[-1][0]]
        inner_attns = [reduce_outs[-1][1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            inner_attn = inner_attns[0]
            feat_low = sparse_reduce_outs[idx - 1]
            upsample_feat = self.upsample_layers[len(self.in_channels) - 1 -
                                                 idx](feat_high)
            upsample_feat = self._make_sparse_tensor(upsample_feat, inner_attn, project=True)
            if self.upsample_feats_cat_first:
                top_down_layer_inputs = self._concat(upsample_feat, feat_low)
                # top_down_layer_inputs = torch.cat([upsample_feat, feat_low], 1)
            else:
                top_down_layer_inputs = self._concat(feat_low, upsample_feat)
                # top_down_layer_inputs = torch.cat([feat_low, upsample_feat], 1)
            inner_out, inner_attn = self.top_down_layers[len(self.in_channels) - 1 - idx](
                top_down_layer_inputs, txt_feats)
            inner_outs.insert(0, inner_out)
            inner_attns.insert(0, inner_attn)
            
        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_layers[idx](feat_low) ## TODO: _make_spconv
            out = self.bottom_up_layers[idx](torch.cat(
                [downsample_feat, feat_high], 1), txt_feats)
            outs.append(out)

        # out_layers
        results = []
        for idx in range(len(self.in_channels)):
            results.append(self.out_layers[idx](outs[idx]))
            
        return tuple(results)


    def _concat(self, f1, f2):
        if isinstance(f1, spconv.SparseConvTensor) and isinstance(f2, spconv.SparseConvTensor):
            concat_indices, inverse_indices = torch.unique(torch.cat([f1.indices, f2.indices], 0), sorted=True, return_inverse=True, dim=0)
            num_features = f1.features.size(1) + f2.features.size(1)
            new_features =  torch.zeros(concat_indices.size(0), num_features, device=f1.features.device)
            f1_idx = inverse_indices[:f1.indices.size(0)]
            f2_idx = inverse_indices[f1.indices.size(0):]
            new_features[f1_idx, :f1.features.size(1)] = f1.features
            new_features[f2_idx, f1.features.size(1):] = f2.features
            
            return spconv.SparseConvTensor(new_features, concat_indices, f1.spatial_shape, f1.batch_size)

        else:
            return torch.cat([f1, f2], 1)