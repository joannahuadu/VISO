from mmengine.hooks import Hook
from ..utils.runner_manager import GlobalClass
import json
from typing import Optional, Union
import os
DATA_BATCH = Optional[Union[dict, tuple, list]]


class VisInfoHook(Hook):
    def __init__(self, text_path):
        self.text_path = text_path # json路径
    """
    
    """
    def before_run(self, runner):
        class_texts = json.load(open(self.text_path, 'r'))
        # 转成list(str)的格式
        class_texts = [str(class_text) for class_text in class_texts]
        GlobalClass.get_instance('class_texts', value = class_texts)

    def before_test_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None) -> None:
        img_name = data_batch['data_samples'][0].img_path    
        img_name = os.path.basename(img_name) 
        instance = GlobalClass.get_instance('img_name', value = None)
        instance.value = img_name
