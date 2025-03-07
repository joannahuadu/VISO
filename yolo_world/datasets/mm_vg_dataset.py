# Copyright (c) Tencent Inc. All rights reserved.
import copy
import json
import logging
import os
from typing import Callable, List, Union

from mmengine.logging import print_log
from mmengine.dataset.base_dataset import (
        BaseDataset, Compose, force_full_init)
from mmdet.datasets.coco import CocoDataset
from mmyolo.registry import DATASETS
from mmyolo.datasets.yolov5_coco import BatchShapePolicyDataset

from collections.abc import Mapping

@DATASETS.register_module()
class VisualGroundingDataset(BatchShapePolicyDataset, CocoDataset):
    """Visual-Grounding dataset."""

    PALETTE = [(220, 20, 60)]
    
    def __init__(self,
                 *args,
                 datasets: List[str] = None,
                 filter_anns: List[str] = None,
                 **kwargs):
        self.datasets = datasets
        self.filter_anns = filter_anns
        super().__init__(*args, **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        img_prefix = self.data_prefix['img_path']
        data_list = []
        raw_data = {}
        for dataset in self.datasets:
            raw_data[dataset] = []

        for ann in os.listdir(self.ann_file):
            if os.path.splitext(ann)[0] not in self.filter_anns:
                for dataset in self.datasets:
                    if dataset in ann and ann[0] == dataset[0]: #rsvg and dior_rsvg
                        raw_data[dataset].append(ann)
        img_id = 0
        for dataset, ann_list in raw_data.items():
            data_info = {}
            for ann in ann_list:
                with open(os.path.join(self.ann_file, ann), "r") as file:
                    for line in file:
                        data = json.loads(line)
                        image_id = data['image_id']
                        if image_id not in data_info:
                            data_info[image_id] = [data]
                        else:
                            data_info[image_id].append(data)
                
            for image_id, info_list in data_info.items():
                cat2id = {}
                texts = []
                for info in info_list:
                    cat_name = info['question']
                    if cat_name not in cat2id:
                        cat2id[cat_name] = len(cat2id)
                        texts.append([cat_name])
                parse_data = {"img_id": img_id, "img_path": os.path.join(img_prefix, dataset, image_id), "instances": [], "texts": texts}
                for info in info_list:
                    parse_data["instances"].append({"ignore_flag": 0, "bbox": [item for sublist in info["poly"] for item in sublist], 'bbox_label': cat2id[info["question"]]})
                    
                data_list.append(parse_data)
                img_id += 1
        
        return data_list