# Copyright (c) OpenMMLab. All rights reserved.
import itertools
from typing import Iterator, List, Optional, Sized, Union

import numpy as np
import torch
from mmengine.dataset import BaseDataset, DefaultSampler
from mmengine.dist import get_dist_info, sync_random_seed
from torch.utils.data import Sampler

from mmyolo.registry import DATA_SAMPLERS

@DATA_SAMPLERS.register_module()
class WeightedSampler(DefaultSampler):
    def __init__(self, dataset, weights_file, replacement=True, **kwargs):
        super().__init__(dataset, **kwargs)
        self.replacement = replacement

        import json
        with open(weights_file, 'r') as f:
            self.sample_weights_dict = json.load(f)

        self.weights = []
        for i in range(len(dataset)):
            img_info = dataset.get_data_info(i)
            filename = str(img_info['img_id'])
            weight = self.sample_weights_dict.get(filename, 1.0)
            self.weights.append(weight)

        self.weights = np.array(self.weights)
        self.prob = self.weights / self.weights.sum()

    def __iter__(self):
        indices = np.random.choice(
            len(self.dataset),
            size=len(self.dataset),
            replace=self.replacement,
            p=self.prob).tolist()
        
        if self.round_up:
            indices = (
                indices *
                int(self.total_size / len(indices) + 1))[:self.total_size]

        indices = indices[self.rank:self.total_size:self.world_size]
        return iter(indices)