# Copyright (c) Tencent Inc. All rights reserved.
import math
import copy
import warnings
from typing import List, Optional, Tuple, Union, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.cnn import ConvModule
from mmengine.config import ConfigDict
from mmengine.model import BaseModule, normal_init
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
from mmyolo.models.dense_heads import YOLOv8HeadModule
from mmyolo.models.utils import gt_instances_preprocess
from mmcv.cnn.bricks import build_norm_layer
import cv2
from .yolo_world_head import BNContrastiveHead, ContrastiveHead, YOLOWorldHead
try:
    from mmrotate.structures.bbox import RotatedBoxes, distance2obb
    MMROTATE_AVAILABLE = True
except ImportError:
    RotatedBoxes = None
    distance2obb = None
    MMROTATE_AVAILABLE = False
from yolo_world.models.sputils import _make_sparse_tensor, SPInfer

@MODELS.register_module()
class YOLOWorldRotatedHeadModule(YOLOv8HeadModule):
    """Head Module for YOLO-World (DOTA, rotated)

    Args:
        embed_dims (int): embed dim for text feautures and image features
        use_bn_head (bool): use batch normalization head
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
        # super().init_weights()
        # BaseModule.init_weights()
        for ang_pred, reg_pred, cls_pred, cls_contrast, stride in zip(self.ang_preds, self.reg_preds, self.cls_preds,
                                                  self.cls_contrasts,
                                                  self.featmap_strides):
            normal_init(ang_pred, std=0.01)
            normal_init(reg_pred, std=0.01)
            cls_pred[-1].bias.data[:] = 0.0  # reset bias
            if hasattr(cls_contrast, 'bias'):
                nn.init.constant_(
                    cls_contrast.bias.data,
                    math.log(5 / self.num_classes / (1024 / stride)**2))
            
    def _init_layers(self) -> None:
        """initialize conv layers in YOLOv8 head."""
        # Init decouple head
        self.cls_preds = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.cls_contrasts = nn.ModuleList()
        self.ang_preds = nn.ModuleList()
        reg_out_channels = max(
            (16, self.in_channels[0] // 4, self.reg_max * 4))
        cls_out_channels = max(self.in_channels[0], self.num_classes)

        for i in range(self.num_levels):
            self.reg_convs.append(
                nn.Sequential(
                    ConvModule(in_channels=self.in_channels[i],
                               out_channels=reg_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    ConvModule(in_channels=reg_out_channels,
                               out_channels=reg_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg)))
            self.ang_preds.append(
                nn.Conv2d(
                    in_channels=reg_out_channels,
                    out_channels=self.angle_out_dim * self.reg_max,
                    kernel_size=1))
            self.reg_preds.append(
                    nn.Conv2d(
                        in_channels=reg_out_channels,
                        out_channels=4 * self.reg_max,
                        kernel_size=1))
            self.cls_preds.append(
                nn.Sequential(
                    ConvModule(in_channels=self.in_channels[i],
                               out_channels=cls_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    ConvModule(in_channels=cls_out_channels,
                               out_channels=cls_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    nn.Conv2d(in_channels=cls_out_channels,
                              out_channels=self.embed_dims,
                              kernel_size=1)))
            if self.use_bn_head:
                self.cls_contrasts.append(
                    BNContrastiveHead(self.embed_dims,
                                      self.norm_cfg,
                                      use_einsum=self.use_einsum))
            else:
                self.cls_contrasts.append(
                    ContrastiveHead(self.embed_dims,
                                    use_einsum=self.use_einsum))

        proj = torch.arange(self.reg_max, dtype=torch.float)
        self.register_buffer('proj', proj, persistent=False)

        if self.freeze_all:
            self._freeze_all()

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
class YOLOWorldRotatedHead(YOLOWorldHead):
    """YOLO-World Head
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


@MODELS.register_module()
class YOLOWorldQBoxHead(YOLOWorldHead):
    """YOLO-World Head for 'qbox'
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    
    def loss_by_feat(
            self,
            cls_scores: Sequence[Tensor],
            bbox_preds: Sequence[Tensor],
            bbox_dist_preds: Sequence[Tensor],
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
            bbox_dist_preds (Sequence[Tensor]): Box distribution logits for
                each scale level with shape (bs, reg_max + 1, H*W, 4).
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
        gt_bboxes = gt_info[:, :, 1:]  # xyxy
        pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        # pred info
        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.num_classes)
            for cls_pred in cls_scores
        ]
        flatten_pred_bboxes = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        # (bs, n, 4 * reg_max)
        flatten_pred_dists = [
            bbox_pred_org.reshape(num_imgs, -1, self.head_module.reg_max * 4)
            for bbox_pred_org in bbox_dist_preds
        ]

        flatten_dist_preds = torch.cat(flatten_pred_dists, dim=1)
        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_pred_bboxes = torch.cat(flatten_pred_bboxes, dim=1)
        flatten_pred_bboxes = self.bbox_coder.decode(
            self.flatten_priors_train[..., :2], flatten_pred_bboxes,
            self.stride_tensor[..., 0])

        assigned_result = self.assigner(
            (flatten_pred_bboxes.detach()).type(gt_bboxes.dtype),
            flatten_cls_preds.detach().sigmoid(), self.flatten_priors_train,
            gt_labels, gt_bboxes, pad_bbox_flag)

        assigned_bboxes = assigned_result['assigned_bboxes']
        assigned_scores = assigned_result['assigned_scores']
        fg_mask_pre_prior = assigned_result['fg_mask_pre_prior']

        assigned_scores_sum = assigned_scores.sum().clamp(min=1)

        loss_cls = self.loss_cls(flatten_cls_preds, assigned_scores).sum()
        loss_cls /= assigned_scores_sum

        # rescale bbox
        assigned_bboxes /= self.stride_tensor
        flatten_pred_bboxes /= self.stride_tensor

        # select positive samples mask
        num_pos = fg_mask_pre_prior.sum()
        if num_pos > 0:
            # when num_pos > 0, assigned_scores_sum will >0, so the loss_bbox
            # will not report an error
            # iou loss
            prior_bbox_mask = fg_mask_pre_prior.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(
                flatten_pred_bboxes, prior_bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = torch.masked_select(
                assigned_bboxes, prior_bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(assigned_scores.sum(-1),
                                              fg_mask_pre_prior).unsqueeze(-1)
            loss_bbox = self.loss_bbox(
                pred_bboxes_pos, assigned_bboxes_pos,
                weight=bbox_weight) / assigned_scores_sum

            # dfl loss
            pred_dist_pos = flatten_dist_preds[fg_mask_pre_prior]
            assigned_ltrb = self.bbox_coder.encode(
                self.flatten_priors_train[..., :2] / self.stride_tensor,
                assigned_bboxes,
                max_dis=self.head_module.reg_max - 1,
                eps=0.01)
            assigned_ltrb_pos = torch.masked_select(
                assigned_ltrb, prior_bbox_mask).reshape([-1, 4])
            loss_dfl = self.loss_dfl(pred_dist_pos.reshape(
                -1, self.head_module.reg_max),
                                     assigned_ltrb_pos.reshape(-1),
                                     weight=bbox_weight.expand(-1,
                                                               4).reshape(-1),
                                     avg_factor=assigned_scores_sum)
        else:
            loss_bbox = flatten_pred_bboxes.sum() * 0
            loss_dfl = flatten_pred_bboxes.sum() * 0
        if self.world_size == -1:
            _, world_size = get_dist_info()
        else:
            world_size = self.world_size
        return dict(loss_cls=loss_cls * num_imgs * world_size,
                    loss_bbox=loss_bbox * num_imgs * world_size,
                    loss_dfl=loss_dfl * num_imgs * world_size)
        
    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
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

        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_decoded_bboxes = self.bbox_coder.decode(
            flatten_priors[None], flatten_bbox_preds, flatten_stride)

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
            ori_shape = img_meta['ori_shape']
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
                empty_results.bboxes = bboxes
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
                                   bboxes=bboxes[keep_idxs])

            if rescale:
                if pad_param is not None:
                    results.bboxes -= results.bboxes.new_tensor([
                        pad_param[2], pad_param[0], pad_param[2], pad_param[0]
                    ])
                results.bboxes /= results.bboxes.new_tensor(
                    scale_factor).repeat((1, 2))

            if cfg.get('yolox_style', False):
                # do not need max_per_img
                cfg.max_per_img = len(results)
                
            results.bboxes = get_box_type('hbox')[1](results.bboxes).convert_to('qbox').tensor
            results = self._bbox_post_process(results=results,
                                              cfg=cfg,
                                              rescale=False,
                                              with_nms=with_nms,
                                              img_meta=img_meta)
            results.bboxes[:, 0::2].clamp_(0, ori_shape[1])
            results.bboxes[:, 1::2].clamp_(0, ori_shape[0])

            results_list.append(results)
        return results_list

@MODELS.register_module()
class YOLOWorldRotatedHeadSP(YOLOWorldRotatedHead):
    """YOLO-World Head
    """

    def __init__(self, 
                # TODO add configs
                *args, **kwargs) -> None:
        # TODO add init
        super().__init__(*args, **kwargs)
        self.num_levels = self.head_module.num_levels
   
    def loss(self, img_feats: Tuple[Tensor], txt_feats: Tensor,
             batch_data_samples: Union[list, dict]) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network."""

        outs = self(img_feats[:self.num_levels], txt_feats)
        # Fast versions
        loss_inputs = outs + tuple([list(img_feats[self.num_levels:])]) +(batch_data_samples['bboxes_labels'],
                              batch_data_samples['img_metas'])
        losses = self.loss_by_feat(*loss_inputs)

        return losses
    # def get_mask_gt(self, batch_gt_instances: Sequence[InstanceData],

    def predict(self,
                img_feats: Tuple[Tensor],
                txt_feats: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network.
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        outs = self(img_feats[:self.num_levels], txt_feats)
        predictions = self.predict_by_feat(*outs,
                                           batch_img_metas=batch_img_metas,
                                           rescale=rescale)
        return predictions    
                    
    def loss_by_feat(
            self,
            cls_scores: Sequence[Tensor],
            bbox_preds: Sequence[Tensor],
            angle_preds: Sequence[Tensor],
            bbox_dist_preds: Sequence[Tensor],
            angle_dist_preds: Sequence[Tensor],
            attn_preds: Sequence[Tensor],
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
            attn_preds (list[Tensor]): Attention prediction for each scale
                level with shape (N, 1, H, W).
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
        ## TODO: add loss of attn_preds
        
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
        gt_bboxes = gt_info[:, :, 1:]  # xywha [batch, num_pred, 5]
        pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float() #num_pred中有pad的

        # pred info
        flatten_cls_scores = [ # Sequence(Tensor[batch, flatten_featmap_size, num_classes]), len(seq)=num_levels 
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.num_classes)
            for cls_score in cls_scores
        ]
        flatten_tblrs = [ # Sequence(Tensor[batch, flatten_featmap_size, 4]), len(seq)=num_levels
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_angles = [ # Sequence(Tensor[batch, flatten_featmap_size, 1]), len(seq)=num_levels
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
                                                                  None] # scale to original image size
        flatten_angles = torch.cat(flatten_angles, dim=1)
        flatten_decoded_angle = self.angle_coder.decode(
            flatten_angles, keepdim=True)
        
        flatten_tblra = torch.cat([flatten_tblrs, flatten_decoded_angle],
                                  dim=-1)
        flatten_rbboxes = distance2obb( # [batch, flatten_featmap_size, 5], obb means (left, top, right, bottom, angle)
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
                bbox_preds[pos_inds], # [num_foreground, 5]
                pos_bbox_targets, # [num_foreground, 5]
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
        
        # cal mask loss
        
        mask_gt = self.get_mask_gt(gt_bboxes, self.featmap_sizes_train, self.featmap_strides)
        loss_mask = self.cal_loss_mask(attn_preds, mask_gt)
        # loss_mask = 
        
        # ?? world_size, num_imgs
        if self.world_size == -1:
            _, world_size = get_dist_info()
        else:
            world_size = self.world_size
        # ?? why *num_imgs * world_size
        # ! 因为这个框架采用了学习率 自动缩放，在算步长的时候会 除以 (num_gpu * batch_per_gpu)， 因此每一个loss都要乘以这个值
        losses = dict()
        losses['loss_cls'] = loss_cls * num_imgs * world_size
        losses['loss_bbox'] = loss_bbox * num_imgs * world_size
        if self.loss_angle is not None:
            losses['loss_angle'] = loss_angle * num_imgs * world_size
        losses['loss_mask'] = loss_mask * num_imgs * world_size
        
        return losses
        # return dict(
        #     loss_cls=loss_cls * num_imgs * world_size,
        #     loss_bbox=loss_bbox * num_imgs * world_size,
        #     loss_dfl=loss_dfl * num_imgs * world_size)
    
    def get_mask_gt(self, gt_bboxes, featmap_sizes, featmap_strides):
        '''
        intput:
        gt_bboxes: [batch, num_pred, 5], 5 means (cx, cy, w, h, a), a \in [-pi/2, pi/2]
        featmap_sizes: Sequence[tensor[H, W]], len(seq)=num_levels
        featmap_strides: Sequence[tensor[int]], len(seq)=num_levels
        
        output:
        mask_gt: list([batch, 1, H, W]), len(seq)=num_levels
        
        将gt_bboxes转成实例分割的mask，输出的mask的大小为[H, W]，如果特征图上的点对应有物体，则mask上的点为1，否则为0
        '''
        batch_size, num_pred, _ = gt_bboxes.shape
        device = gt_bboxes.device
        num_levels = len(featmap_sizes)
        
        mask_gt = []
        
        for level, (featmap_size, stride) in enumerate(zip(featmap_sizes, featmap_strides)):
            H, W = featmap_size
            mask_level = torch.zeros((batch_size, 1, H, W), device=device, dtype=torch.uint8)
            scale_factor = torch.tensor([1/stride, 1/stride, 1/stride, 1/stride, 1], device=device)
            scaled_bboxes = gt_bboxes * scale_factor[None, None, :]
            
            for b in range(batch_size):
                for n in range(num_pred):
                    x, y, w, h, angle = scaled_bboxes[b, n].cpu().numpy()
                    
                    angle_deg = np.degrees(angle)
                    
                    rect = ((x, y), (w, h), angle_deg)
                    box = cv2.boxPoints(rect)
                    
                    box = np.intp(box)  # 改用 np.intp
                    
                    mask = np.zeros((H, W), dtype=np.uint8)
                    cv2.fillPoly(mask, [box], 1)
                    
                    mask_level[b, 0] = mask_level[b, 0] | torch.from_numpy(mask).to(device)
            mask_gt.append(mask_level)

        mask_gt = [mask.float() for mask in mask_gt]
        return mask_gt

    def cal_loss_mask(self, attn_preds, mask_gt):
        '''
        计算mask loss, 使用二元交叉熵损失(BCE loss)
        
        input:
        attn_preds: list([batch, 1, H, W]), len(list)=num_levels
        mask_gt: list([batch, 1, H, W]), len(list)=num_levels
        
        output:
        loss_mask: tensor[1]
        '''
        num_levels = len(attn_preds)
        device = attn_preds[0].device
        total_loss = torch.tensor(0., device=device)
        
        for level in range(num_levels):
            pred = attn_preds[level]
            target = mask_gt[level]
            
            assert pred.shape[2:] == target.shape[2:]
            target = target.sum(dim=1, keepdim=True).clamp(0, 1) # 这个不需要
            
            loss = F.binary_cross_entropy(pred, target, reduction='mean')
            
            total_loss += loss
        
        avg_loss = total_loss / num_levels
        return avg_loss

@MODELS.register_module()
class YOLOWorldRotatedHeadModuleSPInfer(YOLOWorldRotatedHeadModule):
    """Sparse Head Module for YOLO-World (DOTA, rotated)
    """

    def __init__(self,
                 sp_type: str = "vspconv",
                 is_sparse_levels: List[int] = [1,1,0],
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.is_sparse_levels = is_sparse_levels
        self.sp_type = sp_type
        self.sparse_module_name = ['cls_preds', 'reg_convs', 'reg_preds', 'ang_preds']
        self.sparse_module_list = [getattr(self, name) for name in self.sparse_module_name]
        
    def forward(self, img_feats: Tuple[Tensor],
                img_attns: Tuple[Tensor],
                txt_feats: Tensor) -> Tuple[List]:
        """Forward features from the upstream network."""
        assert len(img_feats) == self.num_levels
        txt_feats = [txt_feats for _ in range(self.num_levels)]
        
        # sp_infer = SPInfer(self.sp_type)
        # for idx, is_sparse in enumerate(self.is_sparse_levels):
        #     if is_sparse:
        #         for m in self.sparse_module_list:
        #             sp_infer._replace_spinfer(str(idx), m[idx], m)

        return multi_apply(self.forward_single, img_feats, img_attns, txt_feats, self.is_sparse_levels,
                           self.cls_preds, self.reg_convs, self.reg_preds, self.ang_preds, self.cls_contrasts)

    def forward_single(self, img_feat: Tensor, img_attn: Tensor, txt_feat: Tensor,
                       is_sparse: bool,
                       cls_pred: nn.ModuleList, reg_conv: nn.ModuleList, 
                       reg_pred: nn.ModuleList, ang_pred: nn.ModuleList,
                       cls_contrast: nn.ModuleList) -> Tuple:
        """Forward feature of a single scale level."""
        b, _, h, w = img_feat.shape
        img_feat = _make_sparse_tensor(img_feat, img_attn, is_sparse, ishead=True)
        if is_sparse:        
            cls_embed = cls_pred(img_feat).dense(channels_first=True)
            cls_logit = cls_contrast(cls_embed, txt_feat)
            bbox_dist_preds = reg_pred(reg_conv(img_feat)).dense(channels_first=True)
            angle_dist_preds = ang_pred(reg_conv(img_feat)).dense(channels_first=True)
        else:
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
class YOLOWorldRotatedHeadSPInfer(YOLOWorldRotatedHead):
    """YOLO-World Head
    """
    def __init__(self, 
                *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_levels = self.head_module.num_levels
        self.sp_module = ['head_module']
    
    def predict(self,
                img_feats: Tuple[Tensor],
                txt_feats: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network.
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        outs = self(img_feats[:self.num_levels], img_feats[self.num_levels:], txt_feats)
        predictions = self.predict_by_feat(*outs,
                                           batch_img_metas=batch_img_metas,
                                           rescale=rescale)
        return predictions
    
    def forward(self, img_feats: Tuple[Tensor],
                img_attns: Tuple[Tensor],
                txt_feats: Tensor) -> Tuple[List]:
        """Forward features from the upstream network."""
        return self.head_module(img_feats, img_attns, txt_feats)
    
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