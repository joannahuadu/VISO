# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import os.path as osp
import re
import tempfile
import zipfile
from collections import OrderedDict, defaultdict
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from mmcv.ops import nms_quadri, nms_rotated
from mmengine.evaluator import BaseMetric
from mmengine.fileio import dump
from mmengine.logging import MMLogger

from mmrotate.evaluation import eval_rbbox_map
from mmyolo.registry import METRICS
from mmrotate.structures.bbox import rbox2qbox
from mmdet.evaluation.functional import eval_map

@METRICS.register_module()
class CloudMetric(BaseMetric):
    """fMoW cloud coverage evaluation metric.

    Args:
        metric (str | list[str]): Metrics to be evaluated. Only support
            'accurcy' now. If is list, the first setting in the list will
             be used to evaluate metric.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
        cov_thr (float): The threshold for determining accuracy in cloud coverage prediction.
            Predictions are considered accurate if the absolute difference between
            the predicted cloud coverage score and the ground truth is less than or
            equal to this threshold. This value is used to calculate the accuracy metric.
            Defaults to 5.
    """

    default_prefix: Optional[str] = 'fmow'

    def __init__(self,
                 is_infer: bool = False,
                 metric: Union[str, List[str]] = 'mse',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 cov_thr: float = 5,
                 ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['accuracy']
        if metric not in allowed_metrics:
            raise KeyError(f"metric should be one of 'mAP', but got {metric}.")
        self.metric = metric
        self.cov_thr = cov_thr
        self.is_infer = is_infer


    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for data_sample in data_samples:
            gt = copy.deepcopy(data_sample)
            gt_instances = gt['gt_instances']
            gt_clouds = gt['gt_clouds']
            if gt_instances == {}:
                ann = dict()
            else:
                ann = dict(
                    scores=gt_clouds['scores'].cpu().numpy(),
                    bboxes=gt_instances['bboxes'].cpu().numpy(),
                    labels=gt_instances['labels'].cpu().numpy())
            result = dict()
            if self.is_infer:
                pred = data_sample['pred_instances']
                pred_clouds = data_sample['pred_clouds']
                result['img_id'] = data_sample['img_id']
                result['bboxes'] = pred['bboxes'].cpu().numpy()
                result['labels'] = pred['labels'].cpu().numpy()
                result['scores'] = pred['scores'].cpu().numpy()
                result['clouds'] = pred_clouds['scores'].cpu().numpy()
            else:
                pred = data_sample['pred_instances']
                result['img_id'] = data_sample['img_id']
                result['clouds'] = pred['scores'].cpu().numpy()

            self.results.append((ann, result))

    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.
        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        gts, preds = zip(*results)
        gts_ = [gt['scores'] for gt in gts]
        dets = [pred['clouds'] for pred in preds]
        gts_ = np.concatenate(gts_, axis=0)
        dets = np.concatenate(dets, axis=0)

        eval_results = OrderedDict()

        if self.metric == 'accuracy':
            mse = np.mean((gts_ - dets) ** 2)
            eval_results['loss'] = mse
            accuracy = np.mean(np.abs(gts_ - dets) <= self.cov_thr)
            eval_results['acc'] = accuracy
        else:
            raise NotImplementedError(f"Metric {self.metric} is not implemented.")

        return eval_results
