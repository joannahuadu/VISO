# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmdet.datasets.transforms import PackDetInputs as MMDET_PackDetInputs
from yolo_world.registry import TRANSFORMS
from mmcv.transforms import LoadAnnotations as MMCV_LoadAnnotations

@TRANSFORMS.register_module()
class PackDetInputs(MMDET_PackDetInputs):
    
    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_scores': 'scores'
    }
    

@TRANSFORMS.register_module()
class LoadAnnotations(MMCV_LoadAnnotations):
    def __init__(self,
                 with_cloud: bool = False,
                 box_type: str = None,
                 **kwargs) -> None:
        super(LoadAnnotations, self).__init__(**kwargs)
        self.with_cloud = with_cloud
        self.box_type = box_type
    
    def _load_bboxes(self, results: dict) -> None:
        """Private function to load bounding box annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.
        Returns:
            dict: The dict contains loaded bounding box annotations.
        """
        gt_bboxes = []
        for instance in results.get('instances', []):
            gt_bboxes.append(instance['bbox'])
        if self.box_type is None:
            results['gt_bboxes'] = np.array(
                gt_bboxes, dtype=np.float32).reshape((-1, 4))
        else:
            _, box_type_cls = get_box_type(self.box_type)
            results['gt_bboxes'] = box_type_cls(gt_bboxes, dtype=torch.float32)
            
    def _load_scores(self, results: dict) -> None:
        """Private function to load bounding box annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.
        Returns:
            dict: The dict contains loaded bounding box annotations.
        """
        gt_scores = []
        for instance in results.get('instances', []):
            gt_scores.append(instance['cloud_cover'])
        results['gt_scores'] = np.array(
            gt_scores, dtype=np.float32)
    
    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            dict: The dict contains loaded bounding box, label and
            semantic segmentation.
        """

        if self.with_bbox:
            self._load_bboxes(results)
        if self.with_cloud:
            self._load_scores(results)
        return results