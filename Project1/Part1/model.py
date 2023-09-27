import numpy as np
from collections import defaultdict

from nn import Linear, Sigmoid, ReLU


class ANN:
    def __init__(self, layer_sizes):
        """
        初始化神经网络。

        参数:
        - layer_sizes (list[int]): 表示每个层的神经元数量，例如 [input_size, hidden_size, output_size] 。
        """
        self.module_list = []
        for i in range(len(layer_sizes) - 2):
            self.module_list.append(Linear(layer_sizes[i], layer_sizes[i + 1]))
            self.module_list.append(Sigmoid())
        self.module_list.append(Linear(layer_sizes[-2], layer_sizes[-1]))

    def forward(self, inputs):
        for layer in self.module_list:
            inputs = layer(inputs)
        return inputs

    def backward(self, grads):
        raise NotImplementedError
        for layer in reversed(self.module_list):
            grads = layer.backward(grads)


class MLP:
    def __init__(self, input_dim, output_dim):
        layer_sizes = [input_dim, output_dim]
        self.model = ANN(layer_sizes)

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def save_model(self):
        raise NotImplementedError

    def load_model(self):
        raise NotImplementedError
