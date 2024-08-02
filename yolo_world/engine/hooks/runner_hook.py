from mmengine.hooks import Hook
from ...utils.runner_manager import GlobalClass

class RunnerHook(Hook):
    """
    将runner对象存储到GlobalClass中
    """
    def before_run(self, runner):
        GlobalClass.get_instance('runner', value = runner)
