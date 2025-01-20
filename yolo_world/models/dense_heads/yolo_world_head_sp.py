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
from mmdet.utils import ConfigType, OptConfigType, InstanceList, OptInstanceList
from mmdet.models.utils import (multi_apply, unpack_gt_instances,
                                filter_scores_and_topk)
from mmyolo.registry import MODELS, TASK_UTILS
from mmyolo.models.dense_heads import YOLOv8HeadModule, YOLOv8Head
from mmyolo.models.utils import gt_instances_preprocess
from mmcv.cnn.bricks import build_norm_layer


from .yolo_world_head import YOLOWorldHead

@MODELS.register_module()
class YOLOWorldHeadSP(YOLOWorldHead):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_levels = self.head_module.num_levels
    

    def loss_by_feat(
            self,
            cls_scores: Sequence[Tensor],
            bbox_preds: Sequence[Tensor],
            bbox_dist_preds: Sequence[Tensor],
            batch_gt_instances: Sequence[InstanceData],
            batch_img_metas: Sequence[dict],
            batch_gt_instances_ignore: OptInstanceList = None,
            attn_preds: Sequence[Tensor] = None,
            ) -> dict:
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
            attn_preds (list[Tensor]): Attention prediction for each scale
                level with shape (N, 1, H, W).
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
        
        # cla mask loss
        mask_gt = self.get_mask_gt(gt_bboxes, self.featmap_sizes_train, self.featmap_strides)
        loss_mask = self.cal_loss_mask(attn_preds, mask_gt)
        
                                   
        if self.world_size == -1:
            _, world_size = get_dist_info()
        else:
            world_size = self.world_size
        
        return dict(loss_cls=loss_cls * num_imgs * world_size,
                    loss_bbox=loss_bbox * num_imgs * world_size,
                    loss_dfl=loss_dfl * num_imgs * world_size,
                    loss_mask = loss_mask * num_imgs * world_size)

    def get_mask_gt(self, gt_bboxes, featmap_sizes, featmap_strides):
        """
        generate mask ground truth for each level from gt_bboxes
        Args:
            gt_bboxes: [batch, num_pred, 4] xyxy
            featmap_sizes: Sequence[tuple(H, W)], len(seq)=num_levels
            featmap_strides: Sequence[int], len(seq)=num_levels
        Returns:
            mask_gt: list([batch, 1, H, W]), len(list)=num_levels
        """
        batch_size, num_pred, _ = gt_bboxes.shape
        device = gt_bboxes.device
        num_levels = len(featmap_sizes)

        mask_gt = []
        for level, (featmap_size, stride) in enumerate(zip(featmap_sizes, featmap_strides)):
            H, W = featmap_size
            mask_level = torch.zeros((batch_size, 1, H, W), device=device, dtype=torch.uint8)
            scale_factor = torch.tensor([1/stride, 1/stride, 1/stride, 1/stride], device=device)
            scaled_bboxes = gt_bboxes * scale_factor[None, None, :]

            for b in range(batch_size):
                for n in range(num_pred):
                    x1, y1, x2, y2 = scaled_bboxes[b, n].cpu().numpy()
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    mask = np.zeros((H, W), dtype=np.uint8)
                    mask[y1:y2, x1:x2] = 1
                    mask_level[b, 0] |= torch.from_numpy(mask).to(device)

            mask_gt.append(mask_level)

        mask_gt = [mask.float() for mask in mask_gt]
        return mask_gt

    def cal_loss_mask(self, attn_preds, mask_gt):
        '''
        Calculate the loss of mask prediction for each level of the model using binary cross entropy loss.
        Args:
            attn_preds: list([batch, 1, H, W]), len(list)=num_levels
            mask_gt: list([batch, 1, H, W]), len(list)=num_levels
        Returns:
        loss_mask: tensor[1]
        '''
        num_levels = len(attn_preds)
        device = attn_preds[0].device
        total_loss = torch.tensor(0., device=device)
        
        for level in range(num_levels):
            pred = attn_preds[level]
            target = mask_gt[level]
            
            assert pred.shape[2:] == target.shape[2:]
            target = target.sum(dim=1, keepdim=True).clamp(0, 1)
            
            loss = F.binary_cross_entropy(pred, target, reduction='mean')
            
            total_loss += loss
        
        avg_loss = total_loss / num_levels
        return avg_loss
