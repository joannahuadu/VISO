from mmengine.utils import ManagerMixin


class GlobalClass(ManagerMixin):
    def __init__(self, name, value):
        super().__init__(name)
        self.value = value
