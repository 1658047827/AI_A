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
