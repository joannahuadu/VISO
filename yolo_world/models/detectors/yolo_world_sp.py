# Copyright (c) Tencent Inc. All rights reserved.
from typing import List, Tuple, Union
import torch
import torch.nn as nn
import logging
import json
import numpy as np
from torch import Tensor
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, InstanceList
from mmdet.models.utils import samplelist_boxtype2tensor
from mmyolo.models.detectors import YOLODetector
from mmyolo.registry import MODELS
from mmengine.logging import print_log

from .yolo_world import YOLOWorldDetector, SimpleYOLOWorldDetector
from mmdet.structures.bbox import get_box_type
from mmengine.structures import InstanceData

@MODELS.register_module()
class SimpleYOLOWorldDetectorSP(SimpleYOLOWorldDetector):
    """Implementation of YOLO World Series"""

    def __init__(self,
                 box_type: str = 'rbox',
                 utm_path: str = '',
                 with_utm: bool = False,
                 cloud_model: ConfigType = None,
                 with_cloud_model: bool = False,
                 cov_thr: float = 70,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.with_cloud_model = with_cloud_model
        self.with_utm = with_utm
        self.box_type = box_type
        if self.with_cloud_model:
            if cloud_model is None:
                raise ValueError("`cloud_model` cannot be None when `with_cloud_model` is True.")
            self.cloud_model = MODELS.build(cloud_model)
            self.cov_thr = cov_thr
        else:
            self.cloud_model = None
            self.cov_thr = None
        if self.with_utm:
            assert len(utm_path) > 0, "utm_path cannot be empty if `with_utm` is enabled."
            with open(utm_path, 'r', encoding='utf-8') as file:
                self.utm_attns = json.load(file)
            self.score_th = self.utm_attns['threshold']

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """
        if self.with_utm:
            utm = batch_data_samples[0].utms[0]
            text = batch_data_samples[0].texts[0]
            if utm in self.utm_attns:
                utm_attns = self.utm_attns[utm][text]
                if np.mean(utm_attns) < self.score_th:
                    results_list = []
                    empty_scores = torch.tensor([], device=batch_inputs.device)
                    empty_labels = torch.tensor([], device=batch_inputs.device)
                    _, box_type_cls = get_box_type(self.box_type)
                    empty_bboxes = box_type_cls(torch.tensor([]), device=batch_inputs.device)
                    empty_results = InstanceData(scores=empty_scores,
                                                labels=empty_labels,
                                                bboxes=empty_bboxes)
                    results_list.append(empty_results)
                    batch_data_samples = self.add_pred_to_datasample(
                        batch_data_samples, results_list)
                    return batch_data_samples
        
        img_feats, txt_feats, pred_score = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        if img_feats is None:
            results_list = []
            empty_scores = torch.tensor([], device=batch_inputs.device)
            empty_labels = torch.tensor([], device=batch_inputs.device)
            _, box_type_cls = get_box_type(self.box_type)
            empty_bboxes = box_type_cls(torch.tensor([]), device=batch_inputs.device)
            empty_results = InstanceData(scores=empty_scores,
                                        labels=empty_labels,
                                        bboxes=empty_bboxes)
            results_list.append(empty_results)
            if pred_score is not None:
                clouds_list = []
                clouds_list.append(InstanceData(scores=pred_score))
                
                batch_data_samples = self._add_pred_to_datasample(
                    batch_data_samples, results_list, clouds_list)
            else:
                batch_data_samples = self.add_pred_to_datasample(
                    batch_data_samples, results_list)
            return batch_data_samples
            
        # self.bbox_head.num_classes = self.num_test_classes
        self.bbox_head.num_classes = txt_feats[0].shape[0]
        if self.reparameterized:
            results_list = self.bbox_head.predict(img_feats,
                                                  batch_data_samples,
                                                  rescale=rescale)
        else:
            results_list = self.bbox_head.predict(img_feats,
                                                  txt_feats,
                                                  batch_data_samples,
                                                  rescale=rescale)
        if pred_score is not None:
            clouds_list = []
            clouds_list.append(InstanceData(scores=pred_score))
            batch_data_samples = self._add_pred_to_datasample(
                batch_data_samples, results_list, clouds_list)
        else:
            batch_data_samples = self.add_pred_to_datasample(
                batch_data_samples, results_list)
        return batch_data_samples
    
    def _add_pred_to_datasample(self, data_samples: SampleList,
                               results_list: InstanceList, clouds_list: InstanceList) -> SampleList:
        for data_sample, pred_instances, pred_clouds in zip(data_samples, results_list, clouds_list):
            data_sample.pred_instances = pred_instances
            data_sample.pred_clouds = pred_clouds
        samplelist_boxtype2tensor(data_samples)
        return data_samples
    
    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.
        """
        img_feats, txt_feats,_ = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        if img_feats is None:
            return

        if self.reparameterized:
            results = self.bbox_head.forward(img_feats)
        else:
            results = self.bbox_head.forward(img_feats, txt_feats)
        return results

    def extract_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract features."""
        # only image features
        img_feats, _ = self.backbone(batch_inputs, None)
        
        # cloud detection
        if self.with_cloud_model:
            cloud_cov = self.cloud_model.predict(img_feats,
                                                batch_data_samples)
            pred_score = cloud_cov[0]['scores']
            # print_log(
            #         f'{batch_data_samples[0].img_path}: {pred_score}.',
            #         logger='current',
            #         level=logging.INFO)
            if pred_score > self.cov_thr:
                return None, None, pred_score
        
        if not self.reparameterized:
            # use embeddings
            if isinstance(self.embeddings, list):
                txt_feats = self.embeddings[batch_data_samples[0].img_id][None]
            else:
                txt_feats = self.embeddings[None]
            if self.adapter is not None:
                txt_feats = self.adapter(txt_feats) + txt_feats
                txt_feats = nn.functional.normalize(txt_feats, dim=-1, p=2)
            txt_feats = txt_feats.repeat(img_feats[0].shape[0], 1, 1)
        else:
            txt_feats = None
        if self.with_neck:
            if self.mm_neck:
                img_feats = self.neck(img_feats, txt_feats)
            else:
                img_feats = self.neck(img_feats)
        if self.with_cloud_model:
            return img_feats, txt_feats, pred_score
        else:
            return img_feats, txt_feats, None