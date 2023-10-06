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
    def __init__(self, dim):
        """
        初始化 Softmax 模块。

        参数:
        - dim (int): 指定 Softmax 沿着该维度计算，则所有沿着这个维度的元素和为 1 。

        注意:
        - 例如，对于一个形状为 (B, C, H, W)=(2, 3, 2, 3) 的张量 in ，经过 Softmax 得到张量 out :
            - 如果 dim=0 ，则会有 out[0][i][j][k] + out[1][i][j][k] = 1 。
            - 如果 dim=1 ，则会有 out[b][0][j][k] + out[b][1][j][k] + out[b][2][j][k] = 1 。
            - 如果 dim=2 ，则会有 out[b][i][0][k] + out[b][i][1][k] = 1 。
            - 如果 dim=3 ，则会有 out[b][i][j][0] + out[b][i][j][1] + out[b][i][j][2] = 1 。
        """
        super(Softmax, self).__init__()
        self.dim = dim

    def forward(self, inputs):
        max_inputs = np.max(inputs, axis=self.dim, keepdims=True)
        exp_logits = np.exp(inputs - max_inputs)
        sum_exp = np.sum(exp_logits, axis=self.dim, keepdims=True)
        self.softmax_scores = exp_logits / sum_exp
        return self.softmax_scores

    def backward(self, grads):
        sum = np.sum(grads * self.softmax_scores, axis=self.dim, keepdims=True)
        return self.softmax_scores * (grads - sum)


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
        - grads (ndarray): 形如 (batch_size, output_size) 的二维数组，反向传播过来的梯度值。

        返回:
        - (ndarray): 形如 (batch_size, input_size) 的二维数组，根据链式法则计算的反向传播梯度。
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
        - predicts (ndarray): 形如 (batch_size, output_dim) 的二维数组，模型预测结果。
        - labels (ndarray): 形如 (batch_size, output_dim) 的二维数组，目标结果。

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
        - loss_grad_predicts (ndarray): 形如 (batch_size, output_dim) 的二维数组，一行对应一个 sample \
            ，每一列对应 MSELoss 关于该输出分量的梯度。
        """
        loss_grad_predicts = self.predicts - self.labels
        return loss_grad_predicts


class CrossEntropyLoss(Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, predicts, labels):
        """
        计算交叉熵损失函数。

        参数:
        - predicts (ndarray): 形如 (batch_size, class_num) 的二维数组，第二维是样本预测在每个类别的后验概率。
        - labels (ndarray): 目标结果，形如 (batch_size, ) 的一维数组， labels[i] 存储第 i 个 sample \
            预期的结果类别索引 class_index 。
        """
        self.predicts = predicts
        self.labels = labels
        self.batch_size = predicts.shape[0]
        """
        使用 numpy 高级索引， predicts[np.arange(self.batch_size), labels] 会返回元素为:
        - predicts[0][labels[0]]
        - predicts[1][labels[1]]
        - predicts[2][labels[2]]
        - ...
        - predicts[n-2][labels[n-2]]
        - predicts[n-1][labels[n-1]]
        的形如 (batch_size, ) 的一维数组。其中 n = batch_size ，而 np.arange(self.batch_size) \
            = array([0, 1, 2, ..., n-1])
        """
        epsilon = 1e-15  # 防止 log(0) 的情况，添加一个极小的值
        loss = -np.sum(np.log(predicts[np.arange(self.batch_size), labels] + epsilon))
        return loss / self.batch_size

    def backward(self):
        """
        计算反向传播的所需的梯度。

        返回:
        - loss_grad_predicts (ndarray): 形如 (batch_size, class_num) 的二维数组，
        """
        epsilon = 1e-15  # 防止除以 0 的情况
        arange = np.arange(self.batch_size)
        loss_grad_predicts = np.zeros_like(self.predicts)
        loss_grad_predicts[arange, self.labels] = -1 / (
            self.predicts[arange, self.labels] + epsilon
        )
        return loss_grad_predicts
