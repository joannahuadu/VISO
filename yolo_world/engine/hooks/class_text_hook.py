from mmengine.hooks import Hook
from ...utils.runner_manager import GlobalClass
import json
class ClassTextsHook(Hook):
    def __init__(self, text_path):
        self.text_path = text_path # json路径
    """
    
    """
    def before_run(self, runner):
        class_texts = json.load(open(self.text_path, 'r'))
        # 转成list(str)的格式
        class_texts = [str(class_text) for class_text in class_texts]
        GlobalClass.get_instance('class_texts', value = class_texts)
