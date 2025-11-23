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
from mmyolo.models.dense_heads import YOLOv8HeadModule, YOLOv8Head
from mmyolo.models.utils import gt_instances_preprocess
from mmcv.cnn.bricks import build_norm_layer
import cv2
from .rtm_world_rotated_head import RTMWorldRotatedHead, RTMWorldRotatedHeadModule
try:
    from mmrotate.structures.bbox import RotatedBoxes, distance2obb
    MMROTATE_AVAILABLE = True
except ImportError:
    RotatedBoxes = None
    distance2obb = None
    MMROTATE_AVAILABLE = False
from yolo_world.models.sputils import _make_sparse_tensor


@MODELS.register_module()
class RTMWorldRotatedHeadSP(RTMWorldRotatedHead):
    """YOLO-World Head
    """

    def __init__(self, 
                # TODO add configs
                # attn_loss_weight: List[float] = [],
                is_split_attn: bool = False,
                loss_attn: ConfigType = dict(
                     type='BCELoss',
                     reduction='mean'),
                is_skip_mask: bool = False,
                loss_attn_weight: int = 1,
                *args, **kwargs) -> None:
        # TODO add init
        super().__init__(*args, **kwargs)
        self.num_levels = self.head_module.num_levels
        self.loss_attn: nn.Module = MODELS.build(loss_attn)
        self.is_split_attn = is_split_attn
        self.is_skip_mask = is_skip_mask
        self.loss_attn_weight = loss_attn_weight
        # self.attn_loss_weight = attn_loss_weight
   
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
        mask_gt = self.get_mask_gt(gt_bboxes, gt_labels)
        loss_mask = self.cal_loss_mask(attn_preds, mask_gt)
            
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
        losses['loss_mask'] = self.loss_attn_weight * loss_mask * num_imgs * world_size
        
        return losses
        # return dict(
        #     loss_cls=loss_cls * num_imgs * world_size,
        #     loss_bbox=loss_bbox * num_imgs * world_size,
        #     loss_dfl=loss_dfl * num_imgs * world_size)
    
    def get_mask_gt(self, gt_bboxes, gt_labels):
        '''
        将gt_bboxes转成实例分割的mask，输出的mask的大小为[num_words, H, W]，如果特征图上的点对应有物体，则对应word的mask上的点为1，否则为0
        Args
            gt_bboxes: [batch, num_pred, 5], 5 means (cx, cy, w, h, a), a \in [-pi/2, pi/2]
            gt_labels: [batch, num_pred, 1], 1 means class id
            featmap_sizes: Sequence[tensor[H, W]], len(seq)=num_levels
            featmap_strides: Sequence[tensor[int]], len(seq)=num_levels
            is_split_attn: bool, 决定gt类型
            num_classes: int, 类别数
        Returns:
            mask_gt:
                if is_split_attn=False, list([batch, 1, H, W]), len(seq)=num_levels
                if is_split_attn=True, list([batch, num_words, H, W]), len(seq)=num_levels
            
        '''
        batch_size, num_pred, _ = gt_bboxes.shape
        device = gt_bboxes.device
        num_levels = len(self.featmap_sizes_train)
        
        mask_gt = []
        
        for level, (featmap_size, stride) in enumerate(zip(self.featmap_sizes_train, self.featmap_strides)):
            H, W = featmap_size
            if self.is_split_attn:
                mask_level = torch.zeros((batch_size, self.num_classes, H, W), device=device, dtype=torch.uint8)
            else:
                mask_level = torch.zeros((batch_size, 1, H, W), device=device, dtype=torch.uint8)
            
            scale_factor = torch.tensor([1/stride, 1/stride, 1/stride, 1/stride, 1], device=device)
            scaled_bboxes = gt_bboxes * scale_factor[None, None, :]
            
            for b in range(batch_size):
                for n in range(num_pred):
                    x, y, w, h, angle = scaled_bboxes[b, n].cpu().numpy()
                    if w>0 and h>0:
                        class_id = gt_labels[b, n].item()
                        class_id = int(round(class_id))
                        angle_deg = np.degrees(angle)
                        
                        rect = ((x, y), (w, h), angle_deg)
                        box = cv2.boxPoints(rect)
                        
                        box = np.intp(box)
                        
                        mask = np.zeros((H, W), dtype=np.uint8)
                        cv2.fillPoly(mask, [box], 1)
                        
                        if self.is_split_attn:
                            mask_level[b, class_id] = mask_level[b, class_id] | torch.from_numpy(mask).to(device)
                        else:
                            mask_level[b, 0] = mask_level[b, 0] | torch.from_numpy(mask).to(device)
            
            mask_gt.append(mask_level)

        mask_gt = [mask.float() for mask in mask_gt]
        return mask_gt

    def cal_loss_mask(self, attn_preds, mask_gt):
        '''
        计算mask loss, 使用二元交叉熵损失(BCE loss)
        
        input:
        attn_preds: 
            if is_split_attn=False, list([batch, 1, H, W]), len(list)=num_levels
            if is_split_attn=True, list([batch, H, W, num_words]), len(list)=num_levels
        mask_gt: 
            if is_split_attn=False, list([batch, 1, H, W]), len(list)=num_levels
            if is_split_attn=True, list([batch, num_words, H, W]), len(list)=num_levels
        is_split_attn: bool
        num_classes: int
        
        output:
        loss_mask: tensor[1]
        '''
        if self.is_split_attn:
            attn_preds = [attn_pred.permute(0, 3, 1, 2) for attn_pred in attn_preds]
        num_levels = len(attn_preds)
        device = attn_preds[0].device
        total_loss = torch.tensor(0., device=device)
        
        for level in range(num_levels):
            pred = attn_preds[level]
            target = mask_gt[level]
            
            assert pred.shape == target.shape, f"Shape mismatch at level {level}: pred {pred.shape}, target {target.shape}"
            
            if self.is_split_attn:
                # 对于每个类别分别计算损失
                for class_id in range(self.num_classes):
                    pred_class = pred[:, class_id:class_id+1, :, :]
                    target_class = target[:, class_id:class_id+1, :, :]
                    if self.is_skip_mask and torch.sum(target_class) == 0:
                        continue  # Skip this class if there are no positive samples
                    loss = self.loss_attn(pred_class, target_class)
                    total_loss += loss
            else:
                # 原来的逻辑，但移除了不必要的 target 处理
                loss = self.loss_attn(pred, target)
                total_loss += loss
        
        if self.is_split_attn:
            avg_loss = total_loss / (num_levels * self.num_classes)
        else:
            avg_loss = total_loss / num_levels
        
        return avg_loss


@MODELS.register_module()
class RTMWorldRotatedHeadModuleSPInfer(RTMWorldRotatedHeadModule):
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
        # img_feat = _make_indice_tensor(img_feat, img_attn, ishead=True)
        if is_sparse:
            img_feat = _make_sparse_tensor(img_feat, img_attn)
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
            if is_sparse:
                featmap_sizes = cls_logit.shape[2:]
                inds = (img_attn[:,1] * featmap_sizes[0] + img_attn[:,2]).long()
                return cls_logit.permute(0, 2, 3, 1).reshape(-1, self.num_classes)[inds], bbox_preds.permute(0, 2, 3, 1).reshape(-1, 4)[inds], ang_preds.permute(0, 2, 3, 1).reshape(-1, self.angle_out_dim)[inds], featmap_sizes, inds
            else:
                featmap_sizes = cls_logit.shape[2:]
                return cls_logit.permute(0, 2, 3, 1).reshape(-1, self.num_classes), bbox_preds.permute(0, 2, 3, 1).reshape(-1, 4), ang_preds.permute(0, 2, 3, 1).reshape(-1, self.angle_out_dim), featmap_sizes, None

@MODELS.register_module()
class RTMWorldRotatedHeadSPInfer(RTMWorldRotatedHead):
    """YOLO-World Head
    """
    def __init__(self, 
                 box_type: str = None,
                *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_levels = self.head_module.num_levels
        self.sp_module = ['head_module']
        self.box_type = box_type

    def forward(self, img_feats: Tuple[Tensor],
                txt_feats: Tensor) -> Tuple[List]:
        """Forward features from the upstream network."""
        return self.head_module(img_feats[:self.num_levels], img_feats[self.num_levels:], txt_feats)
    
    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        angle_preds: List[Tensor],
                        featmap_sizes: List[Tensor],
                        featmap_inds: List[Tensor],
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
        # featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

        # If the shape does not change, use the previous mlvl_priors
        if featmap_sizes != self.featmap_sizes:
            self.mlvl_priors = self.prior_generator.grid_priors(
                featmap_sizes,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device)
            self.featmap_sizes = featmap_sizes
        # flatten_priors = torch.cat(self.mlvl_priors)
        flatten_priors = torch.cat([self.mlvl_priors[i] if inds is None else self.mlvl_priors[i][inds] for i, inds in enumerate(featmap_inds)])

        # mlvl_strides = [
        #     flatten_priors.new_full(
        #         (featmap_size.numel() * self.num_base_priors, ), stride) for
        #     featmap_size, stride in zip(featmap_sizes, self.featmap_strides)
        # ]
        mlvl_strides = [
            flatten_priors.new_full(
                ((featmap_size.numel() if inds is None else len(inds)) * self.num_base_priors, ), stride) 
            for featmap_size, inds, stride in zip(featmap_sizes, featmap_inds, self.featmap_strides)
        ]
        flatten_stride = torch.cat(mlvl_strides)

        # flatten cls_scores, bbox_preds and objectness
        flatten_cls_scores = [
            cls_score.reshape(num_imgs, -1, self.num_classes)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_angle_preds = [
            angle_pred.reshape(num_imgs, -1, self.angle_out_dim)
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
            
            if self.box_type is not None:
                results.bboxes = results.bboxes.convert_to(self.box_type).tensor
            results = self._bbox_post_process(results=results,
                                              cfg=cfg,
                                              rescale=False,
                                              with_nms=with_nms,
                                              img_meta=img_meta)
            # results.bboxes[:, 0::2].clamp_(0, ori_shape[1])
            # results.bboxes[:, 1::2].clamp_(0, ori_shape[0])

            results_list.append(results)
        return results_list
