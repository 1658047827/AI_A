线性层参考了torch文档中的定义

按照torch的风格，把Softmax和CELoss结合起来放到CrossEntropyLoss模块，也就是nn.CrossEntropyLoss() = nn.LogSoftmax() + nn.NLLLoss()

TODO: 

+ Linear添加控制是否使用bias参数
+ 理解Linear反向传播的过程（在docs.md中手写一遍）
+ 把Softmax和交叉熵等重要模块的求梯度补充在文档中，理解反向求导过程


参考资料：

+ [带你从零掌握迭代器及构建最简 DataLoader - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/340465632)
+ [DataLoader原理解析 (最简单版本实现) - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/549850590)
+ [PyTorch36.DataLoader源代码剖析 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/169497395)
+ [温故知新——前向传播算法和反向传播算法（BP算法）及其推导 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/71892752)
+ [PyTorch documentation — PyTorch 2.1 documentation](https://pytorch.org/docs/stable/index.html)



## 代码基本架构

代码架构上我积极参考了 PyTorch 的文档，实现了一个 torch-like 的深度学习框架，各个模块的接口都尽量向 torch 看齐，能比较方便地实现可伸缩易调整的网络结构：

+ `nn` 模块是框架核心部分，包含了 `Linear` 、`Sigmoid` 等基本网络结构的前后向传播逻辑。
+ `optim` 模块和训练优化器相关，包含了最基本的优化器 `Optimizer` 等。
+ `utils` 模块与训练数据加载有关，其中实现了 `DataLoader` 、`BatchSampler` 等。
+ `model` 模块是使用框架搭建的自定义模型，包含模型结构、训练逻辑、模型存取等。
+ `init` 模块中主要包含初始化时使用的函数，包括设置随机种子、日志设置、数据预处理等。

为了加速和方便计算表示，我使用了 `numpy` 库进行数学计算，并使用 `numpy.ndarray` 完全代替 `torch.Tensor` 进行向量化计算。

下面挑选代码中最重要的一些类进行分析，为节省字数，其中所有的注释都被删去，具体注释详见源码。

### Linear

```python
class Linear(Module):
    def __init__(self, input_size, output_size):
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
        self.grads["weight"] = np.matmul(self.inputs.T, grads)
        self.grads["bias"] = np.sum(grads, axis=0)
        return np.matmul(grads, self.params["weight"].T)
```

线性层很大程度地参考了 PyTorch 文档的定义，

### Softmax

考虑到 softmax 函数的计算过程需要在后续被复用，我把 softmax 的计算逻辑单独封装为 `softmax` 函数：

```python
def softmax(input, dim):
    exp_logits = np.exp(input - np.max(input, axis=dim, keepdims=True))
    softmax_scores = exp_logits / np.sum(exp_logits, axis=dim, keepdims=True)
    return softmax_scores
```

所以 `Softmax` 的前向传播就是简单地直接调用 `softmax` ，反向传播利用保存下来的 `softmax_scores` ：

```python
class Softmax(Module):
    def __init__(self, dim):
        super(Softmax, self).__init__()
        self.dim = dim

    def forward(self, inputs):
        self.softmax_scores = softmax(inputs, dim=self.dim)
        return self.softmax_scores

    def backward(self, grads):
        sum = np.sum(grads * self.softmax_scores, axis=self.dim, keepdims=True)
        return self.softmax_scores * (grads - sum)
```



### CrossEntropyLoss

```python
class CrossEntropyLoss(Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, predicts, labels):
        self.predicts = predicts
        self.labels = labels
        self.batch_size = predicts.shape[0]

        self.softmax_scores = softmax(predicts, dim=1)

        if predicts.shape == labels.shape:
            loss = -np.sum(labels * np.log(self.softmax_scores))
        else:
            e = labels[0]
            if not (np.isscalar(e) and np.issubsctype(np.asarray(e), np.integer)):
                raise ValueError
            loss = -np.sum(np.log(self.softmax_scores[np.arange(self.batch_size), labels]))
        return loss / self.batch_size

    def backward(self):
        if self.predicts.shape == self.labels.shape:
            grads = self.softmax_scores - self.labels
        else:
            grads = self.softmax_scores.copy()
            grads[np.arange(self.batch_size), self.labels] -= 1
        return grads / self.batch_size
```



### MSELoss

```python
class MSELoss(Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.predicts = None
        self.labels = None
        self.batch_size = None

    def forward(self, predicts, labels):
        self.predicts = predicts
        self.labels = labels
        self.batch_size = predicts.shape[0]
        loss = np.square((predicts - labels)) / 2
        return np.sum(loss) / self.batch_size

    def backward(self):
        loss_grad_predicts = self.predicts - self.labels
        return loss_grad_predicts / self.batch_size
```



### Sigmoid

```python
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
```



### ReLU

```python
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
```



### Optimizer

```python
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
            if hasattr(layer, "grads") and isinstance(layer.grads, dict):
                for key in layer.grads.keys():
                    layer.grads[key].fill(0.0)
```

按照 PyTorch 的文档，`Optimizer` 应当传入一个可迭代的 `params` 声明要被优化（调整）的模型权重，但是这又涉及到 torch 中的子模块注册机制，严格实现会增加很多意义不大的工作量。考虑到实验中只用到了类似 `ModuleList` 这样的结构，而且其中的子模块都是 `nn` 中的基本结构。故可以直接把整个 `model` 传入 `Optimizer` ，调整模型参数时直接遍历 `model.module_list` 即可。

`step()` 方法应在模型反向传播后调用，其遍历各个子模块的参数，根据学习率和各层的梯度进行参数调整。

`zero_grad()` 方法直接遍历各层模块并清零存储的梯度。

### DataLoader

## 对反向传播算法的理解



