import os
import random
import numpy as np


def seed_everything(seed=42):
    """
    设置随机种子，用于确保在随机过程中得到可重复的结果。

    参数:
    - seed (int, optional): 随机种子，默认为 42 。
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def data_generator(data_path):
    """
    生成 f(x)=sin(x) 的训练数据。

    参数:
    - data_path (str): 将生成的数据保存到该路径。
    """
    x = np.random.rand(10000, 1) * 2 * np.pi - np.pi  # [-pi, pi)
    y = np.sin(x)
    np.savez(
        data_path,
        x_train=x[:8000],
        y_train=y[:8000],
        x_valid=x[8000:9000],
        y_valid=y[8000:9000],
        x_test=x[9000:],
        y_test=y[9000:],
    )


def data_preprocess():
    raise NotImplementedError
