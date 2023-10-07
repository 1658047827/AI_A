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
        """
        super(Softmax, self).__init__()
        self.dim = dim

    def forward(self, inputs):
        self.softmax_scores = softmax(inputs, dim=self.dim)
        return self.softmax_scores

    def backward(self, grads):
        sum = np.sum(grads * self.softmax_scores, axis=self.dim, keepdims=True)
        return self.softmax_scores * (grads - sum)


class Linear(Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
    ):
        super(Linear, self).__init__()
        self.inputs = None
        self.params = {"weight": None, "bias": None}
        self.grads = {"weight": None, "bias": None}
        sqrt_k = np.sqrt(1 / input_size)
        self.params["weight"] = np.random.uniform(
            low=-sqrt_k, high=sqrt_k, size=(input_size, output_size)
        )
        self.params["bias"] = np.random.uniform(
            low=-sqrt_k, high=sqrt_k, size=(1, output_size)
        )

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
        # 损失函数模块的 backward() 就已经除以 self.batch_size 了，所以已经是平均梯度
        self.grads["weight"] = np.matmul(self.inputs.T, grads)
        self.grads["bias"] = np.sum(grads, axis=0)
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
        return np.sum(loss) / self.batch_size  # 平均损失

    def backward(self):
        """
        计算反向传播的所需的梯度。

        返回:
        - loss_grad_predicts (ndarray): 形如 (batch_size, output_dim) 的二维数组，一行对应一个 sample \
            ，每一列对应 MSELoss 关于该输出分量的梯度。
        """
        loss_grad_predicts = self.predicts - self.labels
        return loss_grad_predicts / self.batch_size  # 平均梯度


class CrossEntropyLoss(Module):
    """
    nn.CrossEntropyLoss() = nn.LogSoftmax() + nn.NLLLoss() 。
    """

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, predicts, labels):
        """
        计算交叉熵损失函数。

        参数:
        - predicts (ndarray): 形如 (batch_size, class_num) 的二维数组，第二维是样本预测在每个类别的后验概率。
        - labels (ndarray): 目标结果，可以是形如 (batch_size, ) 的一维数组， labels[i] 存储第 i 个 sample \
            预期的结果类别索引 class_index 。也可以是形如 (batch_size, class_num) 的二维数组，代表实际各个类 \
            的概率，例如 one-hot 编码表示。
        """
        self.predicts = predicts
        self.labels = labels
        self.batch_size = predicts.shape[0]

        # 计算 softmax ，得到分类概率
        self.softmax_scores = softmax(predicts, dim=1)

        if predicts.shape == labels.shape:
            loss = -np.sum(labels * np.log(self.softmax_scores))
        else:
            e = labels[0]
            if not (
                np.isscalar(e) and np.issubsctype(np.asarray(e), np.integer)
            ):  # 确保是整数类型
                raise ValueError
            """
            使用 numpy 高级索引， softmax_scores[np.arange(self.batch_size), labels] 会返回元素为:
            - softmax_scores[0][labels[0]]
            - softmax_scores[1][labels[1]]
            - softmax_scores[2][labels[2]]
            - ...
            - softmax_scores[n-2][labels[n-2]]
            - softmax_scores[n-1][labels[n-1]]
            的形如 (batch_size, ) 的一维数组。其中 n = batch_size ，而 np.arange(self.batch_size) \
                = array([0, 1, 2, ..., n-1]) 。
            """
            loss = -np.sum(
                np.log(self.softmax_scores[np.arange(self.batch_size), labels])
            )
        return loss / self.batch_size  # 平均损失

    def backward(self):
        """
        计算反向传播的所需的梯度。

        返回:
        - loss_grad_predicts (ndarray): 形如 (batch_size, class_num) 的二维数组。
        """
        if self.predicts.shape == self.labels.shape:
            grads = self.softmax_scores - self.labels
        else:
            grads = self.softmax_scores.copy()
            grads[np.arange(self.batch_size), self.labels] -= 1
        return grads / self.batch_size  # 平均梯度


def softmax(input, dim):
    """
    计算 softmax 函数。

    参数:
    - input (ndarray): numpy.ndarray 多维数组。
    - dim (int): 指定 softmax 沿着该维度计算，则所有沿着这个维度的元素和为 1 。

    返回:
    - (ndarray): numpy.ndarray 多维数组。

    注意:
    - 例如，对于一个形状为 (B, C, H, W)=(2, 3, 2, 3) 的张量 in ，经过 softmax 得到张量 out :
         - 如果 dim=0 ，则会有 out[0][i][j][k] + out[1][i][j][k] = 1 。
         - 如果 dim=1 ，则会有 out[b][0][j][k] + out[b][1][j][k] + out[b][2][j][k] = 1 。
         - 如果 dim=2 ，则会有 out[b][i][0][k] + out[b][i][1][k] = 1 。
         - 如果 dim=3 ，则会有 out[b][i][j][0] + out[b][i][j][1] + out[b][i][j][2] = 1 。
    """
    exp_logits = np.exp(input - np.max(input, axis=dim, keepdims=True))
    softmax_scores = exp_logits / np.sum(exp_logits, axis=dim, keepdims=True)
    return softmax_scores
