import numpy as np
from collections import defaultdict


class Module:
    def __init__(self) -> None:
        pass

    def __call__(self, *inputs):
        return self.forward(*inputs)

    def forward(self, *inputs):
        raise NotImplementedError

    def backward(self, *grads):
        raise NotImplementedError


class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.outputs = None

    def forward(self, inputs):
        self.outputs = 1.0 / (1.0 + np.exp(-inputs))
        return self.outputs

    def backward(self, grads):
        outputs_grad_inputs = np.multiply(self.outputs, (1.0 - self.outputs))
        return np.multiply(grads, outputs_grad_inputs)


class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, grads):
        outputs_grad_inputs = self.inputs > 0
        return np.multiply(grads, outputs_grad_inputs)


class LeakyReLU(Module):
    def __init__(self, alpha=0.1):
        super(LeakyReLU, self).__init__()
        self.inputs = None
        self.alpha = alpha

    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(self.alpha * inputs, inputs)

    def backward(self, grads):
        return np.where(self.inputs > 0, grads, self.alpha * grads)


class Softmax(Module):
    def __init__(self):
        super(Softmax, self).__init__()
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class Linear(Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        weight_init=np.random.normal,
        bias_init=np.zeros,
    ):
        super(Linear, self).__init__()
        self.inputs = None
        self.params = {"weight": None, "bias": None}
        self.grads = {"weight": None, "bias": None}
        self.params["weight"] = weight_init(size=(input_size, output_size))
        self.params["bias"] = bias_init((1, output_size))

    def forward(self, inputs):
        self.inputs = inputs
        return np.matmul(self.inputs, self.params["weight"]) + self.params["bias"]

    def backward(self, grads):
        """
        计算反向传播梯度。

        参数:
        - grads (ndarray): 一个 (batch_size, output_size) 的二维数组，反向传播过来的梯度值。

        返回:
        - (ndarray): 一个 (batch_size, input_size) 的二维数组，根据链式法则计算的反向传播梯度。
        """
        batch_size = grads.shape[0]
        self.grads["weight"] = np.matmul(self.inputs.T, grads) / batch_size
        self.grads["bias"] = np.sum(grads, axis=0) / batch_size
        return np.matmul(grads, self.params["weight"].T)


class MSELoss(Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.predicts = None
        self.labels = None
        self.batch_size = None

    def forward(self, predicts, labels):
        """
        计算损失函数。

        参数:
        - predicts (ndarray): 一个 (batch_size, output_dim) 的二维数组，模型预测结果。
        - labels (ndarray): 一个 (batch_size, output_dim) 的二维数组，目标结果。

        返回:
        - (float): 该批量数据的平均损失。
        """
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
            ，每一列对应 MSELoss 关于该输出分量的梯度。
        """
        loss_grad_predicts = self.predicts - self.labels
        return loss_grad_predicts
