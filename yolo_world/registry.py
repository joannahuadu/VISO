# Copyright (c) OpenMMLab. All rights reserved.
"""YOLO-World provides one registry node to support using modules across
projects.

Each node is a child of the root registry in MMEngine.
More details can be found at
https://mmengine.readthedocs.io/en/latest/tutorials/registry.html.
"""


from mmengine.registry import DATASETS as MMENGINE_DATASETS
from mmengine.registry import TRANSFORMS as MMENGINE_TRANSFORMS
from mmengine.registry import Registry

DATASETS = Registry('dataset', parent=MMENGINE_DATASETS, locations=['yolo_world.datasets'])
TRANSFORMS = Registry(
    'transform',
    parent=MMENGINE_TRANSFORMS,
    locations=['yolo_world.datasets.transformers'])
