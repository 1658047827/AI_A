import numpy as np
from collections import defaultdict


class Sigmoid:
    def __init__(self):
        self.outputs = None

    def forward(self, inputs):
        self.outputs = 1.0 / (1.0 + np.exp(-inputs))
        return self.outputs

    def backward(self, grads):
        outputs_grad_inputs = np.multiply(self.outputs, (1.0 - self.outputs))
        return np.multiply(grads, outputs_grad_inputs)


class ReLU:
    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.params = None

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.maximum(0, inputs)
        return self.outputs

    def backward(self, grads):
        raise NotImplementedError


class Softmax:
    def __init__(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class Linear:
    def __init__(
        self,
        input_size: int,
        output_size: int,
        weight_init=np.random.normal,
        bias_init=np.zeros,
    ):
        self.inputs = None
        self.params = defaultdict(lambda: {"weight": None, "bias": None})
        self.grads = defaultdict(lambda: {"weight": None, "bias": None})
        self.params["weight"] = weight_init(size=(input_size, output_size))
        self.params["bias"] = bias_init((1, output_size))

    def forward(self, inputs):
        self.inputs = inputs
        return np.matmul(self.inputs, self.params["weight"]) + self.params["bias"]

    def backward(self, grads):
        raise NotImplementedError


class MSELoss:
    def __init__(self):
        self.predicts = None
        self.labels = None
        self.batch_size = None
        self.loss = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)

    def forward(self, predicts, labels):
        self.predicts = predicts
        self.labels = labels
        self.batch_size = predicts.shape[0]
        loss = np.square((predicts - labels)) / 2
        return np.sum(loss) / self.batch_size

    def backward(self):
        """
        计算反向传播的所需的梯度。

        返回:
        - loss_grad_predicts (ndarray): 一个 (batch_size, output_dim) 的二维数组，一行对应一个 sample \
            ，每一列对应 loss 关于该输出分量的梯度。
        """
        loss_grad_predicts = (self.predicts - self.labels)
        return loss_grad_predicts