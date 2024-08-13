# Copyright (c) Tencent Inc. All rights reserved.
from typing import List, Tuple, Union
import torch
import torch.nn as nn
from torch import Tensor
from mmdet.structures import OptSampleList, SampleList
from mmyolo.models.detectors import YOLODetector
from mmyolo.registry import MODELS

from .yolo_world import YOLOWorldDetector, SimpleYOLOWorldDetector
from mmrotate.structures.bbox import RotatedBoxes
from mmengine.structures import InstanceData

@MODELS.register_module()
class SimpleYOLOWorldDetectorSP(SimpleYOLOWorldDetector):
    """Implementation of YOLO World Series"""

    def __init__(self,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """

        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        if img_feats is None:
            results_list=[]
            empty_scores = torch.tensor([], device=batch_inputs.device)
            empty_labels = torch.tensor([], device=batch_inputs.device)
            empty_bboxes = RotatedBoxes(torch.tensor([]), device=batch_inputs.device)
            empty_results = InstanceData(scores=empty_scores,
                                        labels=empty_labels,
                                        bboxes=empty_bboxes)
            results_list.append(empty_results)
            batch_data_samples = self.add_pred_to_datasample(
                batch_data_samples, results_list)
            return batch_data_samples
            
        self.bbox_head.num_classes = self.num_test_classes
        if self.reparameterized:
            results_list = self.bbox_head.predict(img_feats,
                                                  batch_data_samples,
                                                  rescale=rescale)
        else:
            results_list = self.bbox_head.predict(img_feats,
                                                  txt_feats,
                                                  batch_data_samples,
                                                  rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples
    
    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.
        """
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        if img_feats is None:
            return

        if self.reparameterized:
            results = self.bbox_head.forward(img_feats)
        else:
            results = self.bbox_head.forward(img_feats, txt_feats)
        return results