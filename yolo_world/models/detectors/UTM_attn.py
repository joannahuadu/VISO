# Copyright (c) Tencent Inc. All rights reserved.
from typing import List, Tuple, Union
import torch
import torch.nn as nn
import logging
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
class UTMAttnCollector(SimpleYOLOWorldDetector):
    """
    """

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
        results_list = []
        results_list.append(InstanceData(utm_attns=img_feats))
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

        return img_feats

    def extract_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract features."""
        # only image features
        img_feats, _ = self.backbone(batch_inputs, None)

        if not self.reparameterized:
            # use embeddings
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
        return img_feats, txt_feats