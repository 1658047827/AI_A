import os
import logging
import time
import pickle
from optim import Optimizer, StepLR
from nn import Module, Linear, Sigmoid, ReLU, LeakyReLU, MSELoss


class ANN(Module):
    def __init__(self, layer_sizes):
        """
        初始化神经网络结构。

        参数:
        - layer_sizes (list[int]): 表示每个层的神经元数量，例如 [input_size, hidden_size, output_size] 。
        """
        super(ANN, self).__init__()
        self.module_list = []
        for i in range(len(layer_sizes) - 2):
            self.module_list.append(Linear(layer_sizes[i], layer_sizes[i + 1]))
            # 若用 ReLU 激活，学习率得调小，不然容易出现神经元死亡
            self.module_list.append(Sigmoid())
        self.module_list.append(Linear(layer_sizes[-2], layer_sizes[-1]))

    def forward(self, inputs):
        for layer in self.module_list:
            inputs = layer(inputs)
        return inputs

    def backward(self, grads):
        for layer in reversed(self.module_list):
            grads = layer.backward(grads)


class MLP:
    def __init__(self, input_dim, output_dim):
        layer_sizes = [input_dim, 32, 16, 8, output_dim]
        self.model = ANN(layer_sizes)
        self.best = {"train_loss": float("inf"), "valid_loss": float("inf")}  # 记录最佳结果

    def evaluate(self, valid_loader, criterion, metric):
        result = {}
        valid_loss = 0.0
        for batch in valid_loader:
            predicts = self.model(batch["x"])
            train_loss = criterion(predicts, batch["y"])
            valid_loss += train_loss
        valid_loss = valid_loss / len(valid_loader)
        result["valid_loss"] = valid_loss

        if metric is not None:
            raise NotImplementedError

        return result

    def fit(
        self, train_loader, valid_loader, epoches, learning_rate, save_path, **kwargs
    ):
        optimizer = Optimizer(self.model, learning_rate)
        # scheduler = StepLR(optimizer, 1000, 0.5)
        loss_fn = MSELoss()
        metric = None
        log_interval = kwargs.get("log_interval", 10)

        for epoch in range(1, epoches + 1):
            epoch_time_start = time.time()
            epoch_loss = 0.0
            for batch in train_loader:
                predicts = self.model(batch["x"])
                train_loss = loss_fn(predicts, batch["y"])
                epoch_loss += train_loss
                grads = loss_fn.backward()
                self.model.backward(grads)
                optimizer.step()
                optimizer.zero_grad()
            epoch_loss = epoch_loss / len(train_loader)
            epoch_time_elapsed = time.time() - epoch_time_start
            if epoch % log_interval == 0:
                logging.info(
                    "Train epoch {}/{}, learning rate: {}, train loss: {:.6f} [{:.2f}s]".format(
                        epoch, epoches, optimizer.lr, epoch_loss, epoch_time_elapsed
                    )
                )

            # evaluate
            result = self.evaluate(valid_loader, loss_fn, None)
            if result["valid_loss"] < self.best["valid_loss"]:
                logging.info(
                    "Best validation loss has been updated: {:.6f} -> {:.6f}".format(
                        self.best["valid_loss"], result["valid_loss"]
                    )
                )
                result["train_loss"] = epoch_loss
                self.best = result
                self.save_model(save_path)

            # scheduler.step()
        self.load_model(save_path)

    def predict(self, inputs):
        return self.model(inputs)

    def save_model(self, save_path="./model.pkl"):
        """
        同时保存 self.model 和 self.best 。

        参数:
        - save_path (str, optional): 对象序列化后保存到的文件的路径。
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as file:
            pickle.dump({"model": self.model, "best": self.best}, file)
        logging.info("Save model to {}".format(save_path))

    def load_model(self, load_path="./model.pkl"):
        with open(load_path, "rb") as file:
            model_dict = pickle.load(file)
            self.model = model_dict["model"]
            self.best = model_dict["best"]
        logging.info("Load model from {}".format(load_path))
