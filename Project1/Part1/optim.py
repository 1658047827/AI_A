import numpy as np


class Optimizer:
    def __init__(self, model, lr):
        self.model = model
        self.lr = lr

    def step(self):
        for layer in self.model.module_list:
            if hasattr(layer, "params") and isinstance(layer.params, dict):
                for key in layer.params.keys():
                    layer.params[key] -= self.lr * layer.grads[key]

    def zero_grad(self):
        for layer in self.model.module_list:
            if hasattr(layer, "params") and isinstance(layer.params, dict):
                for key in layer.params.keys():
                    layer.params[key].fill(0.0)


class StepLR:
    def __init__(self, optimizer, step_size, gamma=0.1):
        """
        初始化 StepLR 。

        参数:
        - optimizer (Optimizer): 优化器对象，之后会更新其学习率。
        - step_size (int): 每训练 step_size 个 epoch ，更新一次学习率。
        - gamma (float, optional): 更新学习率的乘法因子，默认为 0.1 。
        """
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.step_cnt = 0

    def step(self):
        self.step_cnt += 1
        if self.step_cnt % self.step_size == 0:
            self.optimizer.lr *= self.gamma
            self.step_cnt = 0
