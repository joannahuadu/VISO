import torch
import torch.nn as nn
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmengine.model import BaseModule
from mmyolo.registry import MODELS
from typing import List, Optional, Tuple, Union, Sequence
from torch import Tensor
from mmdet.utils import ConfigType, OptConfigType, InstanceList, OptInstanceList
from mmdet.structures import SampleList
from mmengine.structures import InstanceData
from mmyolo.models.utils import make_divisible, make_round
from mmdet.models.utils import multi_apply
from mmcv.cnn import ConvModule

@MODELS.register_module()
class CloudCoverageHeadModule(BaseModule):
    """CloudCoverageHeadModule head module used in `Cloud Coverage`.

    Args:
        in_channels (Union[int, Sequence]): Number of channels in the input
            feature map.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        featmap_strides (Sequence[int]): Downsample factor of each feature map.
             Defaults to [8, 16, 32].
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for normalization
            layer. Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: Union[int, Sequence],
                 widen_factor: float = 1.0,
                 featmap_strides: Sequence[int] = (8, 16, 32),
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg=init_cfg)
        self.featmap_strides = featmap_strides
        self.num_levels = len(self.featmap_strides)
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_channels = in_channels

        in_channels = []
        for channel in self.in_channels:
            channel = make_divisible(channel, widen_factor)
            in_channels.append(channel)
        self.in_channels = in_channels

        self._init_layers()

    def _init_layers(self):
        self.cov_convs = nn.ModuleList()
        self.cov_pools = nn.ModuleList()
        self.cov_preds = nn.ModuleList()
        self.flattens = nn.ModuleList()
        cov_out_channels = max(
            (16, self.in_channels[0] // 4))
        for i in range(self.num_levels):
            self.cov_convs.append(
                nn.Sequential(
                    ConvModule(
                        in_channels=self.in_channels[i],
                        out_channels=cov_out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg),
                    ConvModule(
                        in_channels=cov_out_channels,
                        out_channels=cov_out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg),
                    nn.MaxPool2d(2)))
            self.cov_pools.append(
                nn.AdaptiveAvgPool2d(1))
            # self.cov_preds.append(
            #     nn.Conv2d(
            #         in_channels=cov_out_channels, 
            #         out_channels=1, 
            #         kernel_size=1))
            self.cov_preds.append(
                nn.Linear(
                    in_features=cov_out_channels, 
                    out_features=1))
            
            self.flattens.append(
                nn.Flatten())

    def forward(self, x: Tuple[Tensor]) -> Tuple[List]:
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            Tuple[List]: A tuple of multi-level coverage scores.
        """
        assert len(x) == self.num_levels
        cov_logits = multi_apply(self.forward_single, x, self.cov_convs, self.cov_pools, self.flattens,
                           self.cov_preds)
        cov_logits = [torch.stack(cov_logits_level) for cov_logits_level in zip(*cov_logits)]
        return (cov_logits,)

    def forward_single(self, x: torch.Tensor, cov_conv: nn.ModuleList, cov_pool: nn.ModuleList, flatten: nn.ModuleList,
                       cov_pred: nn.ModuleList) -> Tuple:
        """Forward feature of a single scale level."""
        cov_logit = cov_pred(flatten(cov_pool(cov_conv(x)))).sigmoid() * 100
        return cov_logit

@MODELS.register_module()
class CloudCoverageHead(BaseDenseHead):
    """Cloud Coverage Detection Head.

    Args:
        in_channels (Union[int, Sequence]): Number of channels in the input
            feature map.
        feat_channels (int): Number of feature channels for intermediate convolutions.
    """

    def __init__(self, 
                head_module: ConfigType,
                loss_pre: ConfigType = dict(
                    type='mmdet.MSELoss',
                    reduction='mean'),
                init_cfg: OptConfigType = None):
        super().__init__(init_cfg=init_cfg)
        
        self.head_module = MODELS.build(head_module)
        self.featmap_strides = self.head_module.featmap_strides
        self.num_levels = len(self.featmap_strides)
        
        self.loss_pre: nn.Module = MODELS.build(loss_pre)
             
    def forward(self, x: Tuple[Tensor]) -> Tuple[List]:
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            Tuple[List]: A tuple of multi-level classification scores, bbox
            predictions, and objectnesses.
        """
        return self.head_module(x)

    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples: SampleList) -> InstanceList:
        
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        outs = self(x)
        predictions = self.predict_by_feat(*outs,
                                           batch_img_metas=batch_img_metas)
        return predictions

    def predict_by_feat(self,
                        cov_scores: List[Tensor],
                        batch_img_metas: Optional[List[dict]] = None) -> InstanceList:
        num_imgs = len(batch_img_metas)
        score = sum(cov_scores) / self.num_levels
        results_list = []
        for sc in score:
            results = InstanceData(scores=sc)
            results_list.append(results)
        return results_list

    def loss(self, x: Tuple[Tensor], batch_data_samples: Union[list,
                                                               dict]) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`], dict): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """

        if isinstance(batch_data_samples, list):
            losses = super().loss(x, batch_data_samples)
        else:
            outs = self(x)
            # Fast version
            loss_inputs = outs + (batch_data_samples['bboxes_scores'],
                                  batch_data_samples['img_metas'])
            losses = self.loss_by_feat(*loss_inputs)

        return losses
    
    def loss_by_feat(
            self,
            cov_scores: Sequence[Tensor],
            batch_gt_instances: Sequence[InstanceData],
            batch_img_metas: Sequence[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        
        num_imgs = len(batch_img_metas)
        gt_scores = batch_gt_instances[:,:1].view(-1,1)
        device = cov_scores[0].device
        score = sum(cov_scores) / self.num_levels
        loss_pre = torch.zeros(1, device=device)
        # loss_pre += (self.loss_pre(score, gt_scores) * torch.log(gt_scores + 1)).mean()
        loss_pre += self.loss_pre(score, gt_scores)
        
        return dict(loss_pre= loss_pre * num_imgs * 0.01)