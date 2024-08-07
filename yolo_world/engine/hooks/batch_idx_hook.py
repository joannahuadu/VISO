from mmengine.hooks import Hook
from ...utils.runner_manager import GlobalClass
from typing import Dict, Optional, Sequence, Union
DATA_BATCH = Optional[Union[dict, tuple, list]]


class BatchIdxHook(Hook):
    """
    将runner对象存储到GlobalClass中
    """
    def before_run(self, runner) -> None:
        GlobalClass.get_instance('batch_idx', value = 0)
    def _before_iter(self,
                     runner,
                     batch_idx: int,
                     data_batch: DATA_BATCH = None,
                     mode: str = 'train') -> None:
        GlobalClass.get_instance('batch_idx').value= batch_idx
        
