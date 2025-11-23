# Copyright (c) Tencent Inc. All rights reserved.
import copy
import json
import logging
from typing import Callable, List, Union

from mmengine.logging import print_log
from mmengine.dataset.base_dataset import (
        BaseDataset, Compose, force_full_init)
from mmyolo.registry import DATASETS

from collections.abc import Mapping

@DATASETS.register_module()
class MultiModalDataset:
    """Multi-modal dataset."""

    PALETTE=[(165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
                    (138, 43, 226), (255, 128, 0), (255, 0, 255),
                    (0, 255, 255), (255, 193, 193), (0, 51, 153),
                    (255, 250, 205), (0, 139, 139), (255, 255, 0),
                    (147, 116, 116), (0, 0, 255)]
    def __init__(self,
                 dataset: Union[BaseDataset, dict],
                 class_text_path: str = None,
                 test_mode: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 lazy_init: bool = False,
                 replace_char: str = "-") -> None:
        self.dataset: BaseDataset
        if class_text_path is not None:
            self.class_texts = json.load(open(class_text_path, 'r'))
            # classes = tuple(item.replace(" ", replace_char)+'/' if self. for sublist in self.class_texts for item in sublist)
            classes =  tuple('/'.join(sublist).replace(" ", replace_char) if len(sublist)>0 else sublist[0].replace(" ", replace_char) for sublist in self.class_texts)
            if 'des' in class_text_path:
                classes = tuple(class_.split('_with')[0] for class_ in classes) 
            metainfo = dict()
            metainfo['classes'] = classes
            # metainfo['palette'] = self.PALETTE[:len(self.class_texts)]
            if dataset.get('metainfo') is not None:
                print_log('metainfo will be overridden by class_texts'
                        'and does not need to be set manually',
                        logger='current',
                        level=logging.WARNING)
            dataset['metainfo'] = metainfo
        else:
            self.class_texts = None
        if isinstance(dataset, dict):
            self.dataset = DATASETS.build(dataset)
        elif isinstance(dataset, BaseDataset):
            self.dataset = dataset
        else:
            raise TypeError(
                'dataset must be a dict or a BaseDataset, '
                f'but got {dataset}')
        print_log(self.dataset,
                logger='current',
                level=logging.INFO)
        # if class_text_path is not None:
            # self.class_texts = json.load(open(class_text_path, 'r'))
        ori_classes = self.dataset.metainfo['classes']
        assert len(ori_classes) == len(self.class_texts), \
            ('The number of classes in the dataset and the class text'
                'file must be the same.')
        # else:
            # self.class_texts = None

        self.test_mode = test_mode
        self._metainfo = self.dataset.metainfo
        self.pipeline = Compose(pipeline)

        self._fully_initialized = False
        if not lazy_init:
            self.full_init()

    @property
    def metainfo(self) -> dict:
        return copy.deepcopy(self._metainfo)

    def full_init(self) -> None:
        """``full_init`` dataset."""
        if self._fully_initialized:
            return

        self.dataset.full_init()
        self._ori_len = len(self.dataset)
        self._fully_initialized = True

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index."""
        data_info = self.dataset.get_data_info(idx)
        if self.class_texts is not None:
            data_info.update({'texts': self.class_texts})
        return data_info

    def __getitem__(self, idx):
        if not self._fully_initialized:
            print_log(
                'Please call `full_init` method manually to '
                'accelerate the speed.',
                logger='current',
                level=logging.WARNING)
            self.full_init()

        data_info = self.get_data_info(idx)

        if hasattr(self.dataset, 'test_mode') and not self.dataset.test_mode:
            data_info['dataset'] = self
        elif not self.test_mode:
            data_info['dataset'] = self
        return self.pipeline(data_info)

    @force_full_init
    def __len__(self) -> int:
        return self._ori_len


@DATASETS.register_module()
class MultiModalMixedDataset(MultiModalDataset):
    """Multi-modal Mixed dataset.
    mix "detection dataset" and "caption dataset"
    Args:
        dataset_type (str): dataset type, 'detection' or 'caption'
    """
    def __init__(self,
                 dataset: Union[BaseDataset, dict],
                 class_text_path: str = None,
                 dataset_type: str = 'detection',
                 test_mode: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 lazy_init: bool = False) -> None:
        self.dataset_type = dataset_type
        super().__init__(dataset,
                         class_text_path,
                         test_mode,
                         pipeline,
                         lazy_init)

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index."""
        data_info = self.dataset.get_data_info(idx)
        if self.class_texts is not None:
            data_info.update({'texts': self.class_texts})
        data_info['is_detection'] = 1 \
            if self.dataset_type == 'detection' else 0
        return data_info
