# Copyright (c) Tencent Inc. All rights reserved.
from typing import List, Tuple, Union
import torch
import torch.nn as nn
from torch import Tensor
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType
from mmyolo.models.detectors import YOLODetector
from mmyolo.registry import MODELS


@MODELS.register_module()
class CloudCoverageDetector(YOLODetector):
    """Implementation of YOLOW Series"""

    def __init__(self,
                 *args,
                 mm_neck: bool = False,
                 num_train_classes=80,
                 num_test_classes=80,
                 cloud_model: ConfigType,
                 with_cloud_model: bool = True,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.with_cloud_model = with_cloud_model
        if self.with_cloud_model:
            self.cloud_model = MODELS.build(cloud_model)
        else:
            self.cloud_model = None
            raise RuntimeError("`cloud_model` is missing.")

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples."""
        img_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        losses = self.cloud_model.loss(img_feats, batch_data_samples)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """

        img_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)

        results_list = self.cloud_model.predict(img_feats,
                                              batch_data_samples)

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
        img_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        results = self.cloud_model.forward(img_feats)
        return results

    def extract_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract features."""
        
        img_feats = self.backbone.forward_image(batch_inputs)
        return img_feats

