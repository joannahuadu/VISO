# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import os.path as osp
import re
import tempfile
import zipfile
import logging
from collections import OrderedDict, defaultdict
from typing import List, Optional, Sequence, Union

import json
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
from mmengine.logging import print_log

@METRICS.register_module()
class UTMMetric(BaseMetric):
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
                 metric: Union[str, List[str]] = 'utm',
                 utm_path: str = '',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['utm']
        if metric not in allowed_metrics:
            raise KeyError(f"metric should be one of 'mAP', but got {metric}.")
        self.metric = metric
        self.utm_path = utm_path


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
            gt_utms = gt['utms']
            gt_texts = gt['texts']
            ann = dict(
                utms=gt_utms,
                texts=gt_texts)
            
            result = dict()
            pred = data_sample['pred_instances']
            result['img_id'] = data_sample['img_id']
            result['utm_attns_1'] = pred['utm_attns'][0].cpu().numpy()
            result['utm_attns_2'] = pred['utm_attns'][1].cpu().numpy()
            result['utm_attns_3'] = pred['utm_attns'][2].cpu().numpy()

            self.results.append((ann, result))
    
    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        gts, preds = zip(*results)
        
        # Initialize dictionary to store attns for each utm and text category
        attns_dict = {}

        # Iterate over all ground truths and predictions
        for gt, pred in zip(gts, preds):
            utm = gt['utms'][0]  # Assuming 'utms' is the utm key we want to store
            text = gt['texts'][0]  # Assuming 'texts' is a string, e.g., 'plane'
            utm_attns_1 = pred['utm_attns_1'].max()
            utm_attns_2 = pred['utm_attns_2'].max()
            utm_attns_3 = pred['utm_attns_3'].max()

            # Initialize storage for each utm and text if not already done
            if utm not in attns_dict:
                attns_dict[utm] = {}
            if text not in attns_dict[utm]:
                attns_dict[utm][text] = {'utm_attns_1': 0, 'utm_attns_2': 0, 'utm_attns_3': 0}
            
            # Append current attns to corresponding list
            attns_dict[utm][text]['utm_attns_1'] = np.maximum(attns_dict[utm][text]['utm_attns_1'], utm_attns_1)
            attns_dict[utm][text]['utm_attns_2'] = np.maximum(attns_dict[utm][text]['utm_attns_2'], utm_attns_2)
            attns_dict[utm][text]['utm_attns_3'] = np.maximum(attns_dict[utm][text]['utm_attns_3'], utm_attns_3)

        # Initialize final metrics dictionary
        metrics_dict = {}

        # Calculate mean for each utm_attns for each utm and text category
        for utm, texts in attns_dict.items():
            metrics_dict[utm] = {}
            print_log(f"utm: {utm}", logger='current', level=logging.INFO)
            for text, attns in texts.items():
                # print_log(f"{text}: {len(attns['utm_attns_1'])}", logger='current', level=logging.INFO)
                # utm_attns_1_mean = sum(attns['utm_attns_1']) / len(attns['utm_attns_1'])
                # utm_attns_2_mean = sum(attns['utm_attns_2']) / len(attns['utm_attns_2'])
                # utm_attns_3_mean = sum(attns['utm_attns_3']) / len(attns['utm_attns_3'])
                # utm_attns_1_max = np.max(np.stack(attns['utm_attns_1'], axis=0), axis=0)
                # utm_attns_2_max = np.max(np.stack(attns['utm_attns_2'], axis=0), axis=0)
                # utm_attns_3_max = np.max(np.stack(attns['utm_attns_3'], axis=0), axis=0)
                utm_attns_1_max = attns['utm_attns_1']
                utm_attns_2_max = attns['utm_attns_2']
                utm_attns_3_max = attns['utm_attns_3']                
                metrics_dict[utm][text] = [utm_attns_1_max, utm_attns_2_max, utm_attns_3_max]
        assert len(self.utm_path) > 0,  "The `utm_path` must not be empty."
        utm_dir = os.path.dirname(self.utm_path)
        if not os.path.exists(utm_dir):
            os.makedirs(utm_dir)
            print(f"Directory '{utm_dir}' created.")
        else:
            print(f"Directory '{utm_dir}' already exists.")
        # torch.save(metrics_dict, self.utm_path)
        with open(self.utm_path, 'w') as f:
            json.dump(metrics_dict, f, indent=4)  # indent=4 for better readability
        
        eval_results = OrderedDict()

        if self.metric == 'utm':
            eval_results['loss'] = len(metrics_dict)
        else:
            raise NotImplementedError(f"Metric {self.metric} is not implemented.")

        return eval_results
