# Copyright (c) OpenMMLab. All rights reserved.
import copy
import itertools
import logging
from typing import Dict, Optional

from mmengine.logging import print_log
from mmengine.model import is_model_wrapper
from mmyolo.registry import HOOKS, MODELS
from mmengine.hooks.hook import DATA_BATCH, Hook
from yolo_world.models.sputils import SPInfer

@HOOKS.register_module()
class SPHook(Hook):
    """A Hook to apply sparse convolution on the model during
    inference.
    
    """

    priority = 'NORMAL'

    def __init__(self):
        pass

    def before_test_epoch(self, runner) -> None:
        """We replace normal conv to sparse conv before test.

        Args:
            runner (Runner): The runner of the training process.
        """
        try:
            model = runner.model.module
        except Exception as e:
            model = runner.model
        sparse_part_name = ['neck']
        sparse_part  = [getattr(model, name) for name in sparse_part_name]
        for module in sparse_part:
            for name in module.sp_module:
                spm = getattr(module, name)
                for idx in range(len(spm)):
                    if spm[idx].is_sparse:
                        sp_infer = SPInfer(spm[idx].sp_type)
                        for name, m in zip(spm[idx].sparse_module_name, spm[idx].sparse_module_list):
                            sp_infer._replace_spinfer(name, m, spm[idx])
        
        sparse_part_name = ['bbox_head']
        sparse_part  = [getattr(model, name) for name in sparse_part_name]
        for module in sparse_part:
            for name in module.sp_module:
                spm = getattr(module, name)
                sp_infer = SPInfer(spm.sp_type)
                for idx, is_sparse in enumerate(spm.is_sparse_levels):
                    if is_sparse:
                        for m in spm.sparse_module_list:
                            sp_infer._replace_spinfer(str(idx), m[idx], m)