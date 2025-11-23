# Copyright (c) Tencent Inc. All rights reserved.
import math
import copy
import warnings
from typing import List, Optional, Tuple, Union, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.cnn import ConvModule, is_norm
from mmengine.config import ConfigDict
from mmengine.model import BaseModule, normal_init, constant_init
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm

from mmengine.dist import get_dist_info
from mmengine.structures import InstanceData
from mmdet.structures import SampleList
from mmdet.structures.bbox import HorizontalBoxes, distance2bbox, get_box_type
from mmdet.structures.bbox.transforms import bbox_cxcywh_to_xyxy, scale_boxes
from mmdet.utils import ConfigType, OptConfigType, InstanceList, OptInstanceList, reduce_mean
from mmdet.models.utils import (multi_apply, unpack_gt_instances,
                                filter_scores_and_topk)
from mmyolo.registry import MODELS, TASK_UTILS
from mmyolo.models.dense_heads import RTMDetRotatedSepBNHeadModule, RTMDetRotatedHead
from mmyolo.models.utils import gt_instances_preprocess
from mmcv.cnn.bricks import build_norm_layer
import cv2
from .yolo_world_head import RepBNContrastiveHead, BNContrastiveHead, ContrastiveHead, YOLOWorldHead
try:
    from mmrotate.structures.bbox import RotatedBoxes, distance2obb
    MMROTATE_AVAILABLE = True
except ImportError:
    RotatedBoxes = None
    distance2obb = None
    MMROTATE_AVAILABLE = False
from yolo_world.models.sputils import _make_sparse_tensor

@MODELS.register_module()
class RTMWorldRotatedHeadModule(RTMDetRotatedSepBNHeadModule):
    """Head Module for YOLO-World (DOTA, rotated)

    Args:
        embed_dims (int): embed dim for text feautures and image features
        use_bn_head (bool): use batch normalization head
        angle_out_dim (int): angle dimension for the angle head in rotated boxes prediction, default: 1
    """

    def __init__(self,
                 *args,
                 embed_dims: int,
                 use_bn_head: bool = False,
                 use_einsum: bool = True,
                 freeze_all: bool = False,
                 angle_out_dim: int = 1,
                 **kwargs) -> None:
        self.embed_dims = embed_dims
        self.use_bn_head = use_bn_head
        self.use_einsum = use_einsum
        self.freeze_all = freeze_all
        self.angle_out_dim = angle_out_dim
        super().__init__(*args, **kwargs)

    def init_weights(self, prior_prob=0.01):
        """Initialize the weight and bias of PPYOLOE head."""
        super().init_weights()
        for rtm_cls, cls_contrast, stride in zip(self.rtm_cls,
                                                  self.cls_contrasts,
                                                  self.featmap_strides):
            rtm_cls[-1].bias.data[:] = 0.0  # reset bias
            if hasattr(cls_contrast, 'bias'):
                nn.init.constant_(
                    cls_contrast.bias.data,
                    math.log(5 / self.num_classes / (1024 / stride)**2))
            
    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self.rtm_ang = nn.ModuleList()
        for _ in range(len(self.featmap_strides)):
            self.rtm_ang.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.num_base_priors * self.angle_out_dim,
                    self.pred_kernel_size,
                    padding=self.pred_kernel_size // 2))
    
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_contrasts = nn.ModuleList()
        self.rtm_cls = nn.ModuleList()
        self.rtm_reg = nn.ModuleList()
        for n in range(len(self.featmap_strides)):
            cls_convs = nn.ModuleList()
            reg_convs = nn.ModuleList()
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels
                cls_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                reg_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)

            self.rtm_cls.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.num_base_priors * self.embed_dims,
                    self.pred_kernel_size,
                    padding=self.pred_kernel_size // 2))
            self.rtm_reg.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.num_base_priors * 4,
                    self.pred_kernel_size,
                    padding=self.pred_kernel_size // 2))
            if self.use_bn_head:
                self.cls_contrasts.append(
                    BNContrastiveHead(self.embed_dims,
                                      self.norm_cfg,
                                      use_einsum=self.use_einsum))
            else:
                self.cls_contrasts.append(
                    ContrastiveHead(self.embed_dims,
                                    use_einsum=self.use_einsum))
        if self.share_conv:
            for n in range(len(self.featmap_strides)):
                for i in range(self.stacked_convs):
                    self.cls_convs[n][i].conv = self.cls_convs[0][i].conv
                    self.reg_convs[n][i].conv = self.reg_convs[0][i].conv

    def _freeze_all(self):
        """Freeze the model."""
        for m in self.modules():
            if isinstance(m, _BatchNorm):
                m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_all:
            self._freeze_all()

    def forward(self, img_feats: Tuple[Tensor],
                txt_feats: Tensor) -> Tuple[List]:
        """Forward features from the upstream network."""
        assert len(img_feats) == self.num_levels
        txt_feats = [txt_feats for _ in range(self.num_levels)]
        return multi_apply(self.forward_single, img_feats, txt_feats,
                           self.cls_preds, self.reg_convs, self.reg_preds, self.ang_preds, self.cls_contrasts)

    def forward_single(self, img_feat: Tensor, txt_feat: Tensor,
                       cls_pred: nn.ModuleList, reg_conv: nn.ModuleList, 
                       reg_pred: nn.ModuleList, ang_pred: nn.ModuleList,
                       cls_contrast: nn.ModuleList) -> Tuple:
        """Forward feature of a single scale level."""
        b, _, h, w = img_feat.shape
        cls_embed = cls_pred(img_feat)
        cls_logit = cls_contrast(cls_embed, txt_feat)
        bbox_dist_preds = reg_pred(reg_conv(img_feat))
        angle_dist_preds = ang_pred(reg_conv(img_feat))
        if self.reg_max > 1:
            bbox_dist_preds = bbox_dist_preds.reshape(
                [-1, 4, self.reg_max, h * w]).permute(0, 3, 1, 2)
            angle_dist_preds = angle_dist_preds.reshape(
                [-1, 1, self.reg_max, h * w]).permute(0, 3, 1, 2)
            # TODO: The get_flops script cannot handle the situation of
            #  matmul, and needs to be fixed later
            # bbox_preds = bbox_dist_preds.softmax(3).matmul(self.proj)
            bbox_preds = bbox_dist_preds.softmax(3).matmul(
                self.proj.view([-1, 1])).squeeze(-1)
            bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
            ang_preds = angle_dist_preds.softmax(3).matmul(
                self.proj.view([-1, 1])).squeeze(-1)
            ang_preds = ang_preds.transpose(1, 2).reshape(b, -1, h, w)
        else:
            bbox_preds = bbox_dist_preds
            ang_preds = angle_dist_preds
        if self.training:
            return cls_logit, bbox_preds, ang_preds, bbox_dist_preds, angle_dist_preds
        else:
            return cls_logit, bbox_preds, ang_preds
        
        
@MODELS.register_module()
class RTMWorldRotatedHead(RTMDetRotatedHead):
    """YOLO-World Rotated Head
    """

    def __init__(self, 
                head_module: ConfigType,
                angle_version: str = 'le90', 
                use_hbbox_loss: bool = False, 
                angle_coder: ConfigType = dict(type='mmrotate.PseudoAngleCoder'),
                loss_angle: OptConfigType = None,
                *args, **kwargs) -> None:
        if not MMROTATE_AVAILABLE:
            raise ImportError(
                'Please run "mim install -r requirements/mmrotate.txt" '
                'to install mmrotate first for rotated detection.')
        self.angle_version = angle_version
        self.use_hbbox_loss = use_hbbox_loss
        if self.use_hbbox_loss:
            assert loss_angle is not None, \
                ('When use hbbox loss, loss_angle needs to be specified')
        self.angle_coder = TASK_UTILS.build(angle_coder)
        self.angle_out_dim = self.angle_coder.encode_size
        if head_module.get('angle_out_dim') is not None:
            warnings.warn('angle_out_dim will be overridden by angle_coder '
                          'and does not need to be set manually')

        head_module['angle_out_dim'] = self.angle_out_dim
        super().__init__(head_module=head_module, *args, **kwargs)
        
        if loss_angle is not None:
            self.loss_angle = MODELS.build(loss_angle)
        else:
            self.loss_angle = None
        
    def loss_by_feat(
            self,
            cls_scores: Sequence[Tensor],
            bbox_preds: Sequence[Tensor],
            angle_preds: Sequence[Tensor],
            bbox_dist_preds: Sequence[Tensor],
            angle_dist_preds: Sequence[Tensor],
            batch_gt_instances: Sequence[InstanceData],
            batch_img_metas: Sequence[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (Sequence[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (Sequence[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            angle_preds (list[Tensor]): Angle prediction for each scale
                level with shape (N, num_priors?? * angle_out_dim, H, W).
            bbox_dist_preds?? (Sequence[Tensor]): Box distribution logits for
                each scale level with shape (bs, reg_max + 1, H*W, 4).
            angle_dist_preds?? (Sequence[Tensor]): ??.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
        Returns:
            dict[str, Tensor]: A dictionary of losses.
        """
        num_imgs = len(batch_img_metas)

        current_featmap_sizes = [
            cls_score.shape[2:] for cls_score in cls_scores
        ]
        # If the shape does not equal, generate new one
        if current_featmap_sizes != self.featmap_sizes_train:
            self.featmap_sizes_train = current_featmap_sizes

            mlvl_priors_with_stride = self.prior_generator.grid_priors(
                self.featmap_sizes_train,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device,
                with_stride=True)

            self.num_level_priors = [len(n) for n in mlvl_priors_with_stride]
            self.flatten_priors_train = torch.cat(mlvl_priors_with_stride,
                                                  dim=0)
            self.stride_tensor = self.flatten_priors_train[..., [2]]

        # gt info
        gt_info = gt_instances_preprocess(batch_gt_instances, num_imgs)
        gt_labels = gt_info[:, :, :1]
        gt_bboxes = gt_info[:, :, 1:]  # xywha
        pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        # pred info
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.num_classes)
            for cls_score in cls_scores
        ]
        flatten_tblrs = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_angles = [
            angle_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.angle_out_dim) 
            for angle_pred in angle_preds
        ]
        
        # # (bs, n, 4 * reg_max)
        # flatten_pred_dists = [
        #     bbox_pred_org.reshape(num_imgs, -1, self.head_module.reg_max * self.bbox_coder.encode_size)
        #     for bbox_pred_org in bbox_dist_preds
        # ]

        # flatten_dist_preds = torch.cat(flatten_pred_dists, dim=1)
        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1)
        flatten_tblrs = torch.cat(flatten_tblrs, dim=1)
        flatten_tblrs = flatten_tblrs * self.flatten_priors_train[..., -1,
                                                                  None]
        flatten_angles = torch.cat(flatten_angles, dim=1)
        flatten_decoded_angle = self.angle_coder.decode(
            flatten_angles, keepdim=True)
        
        flatten_tblra = torch.cat([flatten_tblrs, flatten_decoded_angle],
                                  dim=-1)
        flatten_rbboxes = distance2obb(
            self.flatten_priors_train[..., :2],
            flatten_tblra,
            angle_version=self.angle_version)

        if self.use_hbbox_loss:
            flatten_hbboxes = distance2bbox(self.flatten_priors_train[..., :2],
                                            flatten_tblrs)
        assigned_result = self.assigner(
            (flatten_rbboxes.detach()).type(gt_bboxes.dtype),
            flatten_cls_scores.detach().sigmoid(), self.flatten_priors_train,
            gt_labels, gt_bboxes, pad_bbox_flag)

        labels = assigned_result['assigned_labels'].reshape(-1)
        label_weights = assigned_result['assigned_labels_weights'].reshape(-1)
        bbox_targets = assigned_result['assigned_bboxes'].reshape(-1, 5)
        assign_metrics = assigned_result['assign_metrics'].reshape(-1)
        cls_preds = flatten_cls_scores.reshape(-1, self.num_classes)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)
        avg_factor = reduce_mean(assign_metrics.sum()).clamp_(min=1).item()

        loss_cls = self.loss_cls(
            cls_preds, (labels, assign_metrics),
            label_weights,
            avg_factor=avg_factor)

        pos_bbox_targets = bbox_targets[pos_inds]

        if self.use_hbbox_loss:
            bbox_preds = flatten_hbboxes.reshape(-1, 4)
            pos_bbox_targets = bbox_cxcywh_to_xyxy(pos_bbox_targets[:, :4])
        else:
            bbox_preds = flatten_rbboxes.reshape(-1, 5)
        angle_preds = flatten_angles.reshape(-1, self.angle_out_dim)

        if len(pos_inds) > 0:
            loss_bbox = self.loss_bbox(
                bbox_preds[pos_inds],
                pos_bbox_targets,
                weight=assign_metrics[pos_inds],
                avg_factor=avg_factor)
            loss_angle = angle_preds.sum() * 0
            if self.loss_angle is not None:
                pos_angle_targets = bbox_targets[pos_inds][:, 4:5]
                pos_angle_targets = self.angle_coder.encode(pos_angle_targets)
                loss_angle = self.loss_angle(
                    angle_preds[pos_inds],
                    pos_angle_targets,
                    weight=assign_metrics[pos_inds],
                    avg_factor=avg_factor)
        else:
            loss_bbox = bbox_preds.sum() * 0
            loss_angle = angle_preds.sum() * 0

        # ?? world_size, num_imgs
        if self.world_size == -1:
            _, world_size = get_dist_info()
        else:
            world_size = self.world_size
        # ?? why *num_imgs * world_size
        losses = dict()
        losses['loss_cls'] = loss_cls * num_imgs * world_size
        losses['loss_bbox'] = loss_bbox * num_imgs * world_size
        if self.loss_angle is not None:
            losses['loss_angle'] = loss_angle * num_imgs * world_size
        
        return losses
        # return dict(
        #     loss_cls=loss_cls * num_imgs * world_size,
        #     loss_bbox=loss_bbox * num_imgs * world_size,
        #     loss_dfl=loss_dfl * num_imgs * world_size)

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        angle_preds: List[Tensor],
                        objectnesses: Optional[List[Tensor]] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = True,
                        with_nms: bool = True) -> List[InstanceData]:
        """Transform a batch of output features extracted by the head into
        bbox results.
        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            angle_preds (list[Tensor]): Box angle for each scale level
                with shape (N, num_points * angle_dim, H, W)
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            batch_img_metas (list[dict], Optional): Batch image meta info.
                Defaults to None.
            cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(cls_scores) == len(bbox_preds)
        if objectnesses is None:
            with_objectnesses = False
        else:
            with_objectnesses = True
            assert len(cls_scores) == len(objectnesses)

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)

        multi_label = cfg.multi_label
        multi_label &= self.num_classes > 1
        cfg.multi_label = multi_label
        
        # Whether to decode rbox with angle.
        # different setting lead to different final results.
        # Defaults to True.
        decode_with_angle = cfg.get('decode_with_angle', True)
        
        num_imgs = len(batch_img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

        # If the shape does not change, use the previous mlvl_priors
        if featmap_sizes != self.featmap_sizes:
            self.mlvl_priors = self.prior_generator.grid_priors(
                featmap_sizes,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device)
            self.featmap_sizes = featmap_sizes
        flatten_priors = torch.cat(self.mlvl_priors)

        mlvl_strides = [
            flatten_priors.new_full(
                (featmap_size.numel() * self.num_base_priors, ), stride) for
            featmap_size, stride in zip(featmap_sizes, self.featmap_strides)
        ]
        flatten_stride = torch.cat(mlvl_strides)

        # flatten cls_scores, bbox_preds and objectness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.num_classes)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_angle_preds = [
            angle_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                   self.angle_out_dim)
            for angle_pred in angle_preds
        ]
        
        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_angle_preds = torch.cat(flatten_angle_preds, dim=1)
        flatten_angle_preds = self.angle_coder.decode(
            flatten_angle_preds, keepdim=True)
        
        if decode_with_angle:
            flatten_rbbox_preds = torch.cat(
                [flatten_bbox_preds, flatten_angle_preds], dim=-1)
            flatten_decoded_bboxes = self.bbox_coder.decode(
                flatten_priors[None], flatten_rbbox_preds, flatten_stride)
        else:
            flatten_decoded_hbboxes = self.bbox_coder.decode(
                flatten_priors[None], flatten_bbox_preds, flatten_stride)
            flatten_decoded_hbboxes = HorizontalBoxes.xyxy_to_cxcywh(
                flatten_decoded_hbboxes)
            flatten_decoded_bboxes = torch.cat(
                [flatten_decoded_hbboxes, flatten_angle_preds], dim=-1)
            
        if with_objectnesses:
            flatten_objectness = [
                objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
                for objectness in objectnesses
            ]
            flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
        else:
            flatten_objectness = [None for _ in range(num_imgs)]
        # 8400
        # print(flatten_cls_scores.shape)
        results_list = []
        for (bboxes, scores, objectness,
             img_meta) in zip(flatten_decoded_bboxes, flatten_cls_scores,
                              flatten_objectness, batch_img_metas):
            # ori_shape = img_meta['ori_shape']
            scale_factor = img_meta['scale_factor']
            if 'pad_param' in img_meta:
                pad_param = img_meta['pad_param']
            else:
                pad_param = None

            score_thr = cfg.get('score_thr', -1)
            # yolox_style does not require the following operations
            if objectness is not None and score_thr > 0 and not cfg.get(
                    'yolox_style', False):
                conf_inds = objectness > score_thr
                bboxes = bboxes[conf_inds, :]
                scores = scores[conf_inds, :]
                objectness = objectness[conf_inds]

            if objectness is not None:
                # conf = obj_conf * cls_conf
                scores *= objectness[:, None]

            if scores.shape[0] == 0:
                empty_results = InstanceData()
                empty_results.bboxes = RotatedBoxes(bboxes)
                empty_results.scores = scores[:, 0]
                empty_results.labels = scores[:, 0].int()
                results_list.append(empty_results)
                continue

            nms_pre = cfg.get('nms_pre', 100000)
            if cfg.multi_label is False:
                scores, labels = scores.max(1, keepdim=True)
                scores, _, keep_idxs, results = filter_scores_and_topk(
                    scores,
                    score_thr,
                    nms_pre,
                    results=dict(labels=labels[:, 0]))
                labels = results['labels']
            else:
                scores, labels, keep_idxs, _ = filter_scores_and_topk(
                    scores, score_thr, nms_pre)

            results = InstanceData(scores=scores,
                                   labels=labels,
                                   bboxes=RotatedBoxes(bboxes[keep_idxs]))

            if rescale:
                if pad_param is not None:
                    # results.bboxes -= results.bboxes.new_tensor([
                    #     pad_param[2], pad_param[0], pad_param[2], pad_param[0]
                    # ])
                    results.bboxes.translate_([-pad_param[2], -pad_param[0]])
                
                # results.bboxes /= results.bboxes.new_tensor(
                    # scale_factor).repeat((1, 2))
                scale_factor = [1 / s for s in img_meta['scale_factor']]
                results.bboxes = scale_boxes(results.bboxes, scale_factor)

            if cfg.get('yolox_style', False):
                # do not need max_per_img
                cfg.max_per_img = len(results)

            results = self._bbox_post_process(results=results,
                                              cfg=cfg,
                                              rescale=False,
                                              with_nms=with_nms,
                                              img_meta=img_meta)
            # results.bboxes[:, 0::2].clamp_(0, ori_shape[1])
            # results.bboxes[:, 1::2].clamp_(0, ori_shape[0])

            results_list.append(results)
        return results_list