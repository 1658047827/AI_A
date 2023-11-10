import os
import sys
import random
import json
import logging
import hashlib
import cv2
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


def set_log(file_path):
    """
    配置日志处理程序，将日志输出到终端和指定的文件。

    参数:
    - file_path (str): 要保存日志的文件路径。
    """
    # 移除已有的所有日志处理程序，确保日志同时输出到终端和文件
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_handlers = [logging.StreamHandler(stream=sys.stdout)]
    if file_path is not None:
        log_handlers.append(logging.FileHandler(file_path))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s P%(process)d %(levelname)s %(message)s",
        handlers=log_handlers,
    )


def json_pretty_dump(obj, filename):
    """
    将 Python 对象以 JSON 格式存入文件，并使用缩进和分隔符进行格式化。

    参数:
    - obj (object): 要写入文件的 Python 对象。
    - filename (str): 写入文件的路径。
    """
    with open(filename, "w") as fw:
        json.dump(
            obj,
            fw,
            sort_keys=True,
            indent=4,
            separators=(",", ": "),
            ensure_ascii=False,
        )


def dump_args(args, record_path, log, mode):
    """
    用于保存参数和设置日志。

    参数:
    - args (dict): 需要保存的参数字典。
    - record_path (str): 参数保存路径。
    - log (bool): 设置是否保存到日志文件。
    - mode (str): 主程序的模式，为 "train", "test", "train_and_test" 中的一个。

    注意:
    - 为了使日志输出同时显示在文件和终端上，该函数会执行调用 set_log(log_file) 。
    """
    # 使用参数字典的哈希值作为ID
    hash_id = hashlib.md5(
        str(sorted([(k, v) for k, v in args.items()])).encode("utf-8")
    ).hexdigest()[0:8]
    args["hash_id"] = hash_id

    record_path = os.path.join(record_path, hash_id)
    os.makedirs(record_path, exist_ok=True)
    json_pretty_dump(args, os.path.join(record_path, "args.json"))
    if log:
        set_log(os.path.join(record_path, f"{mode}.log"))
    else:
        set_log(None)
    logging.info("Dump args: {}".format(json.dumps(args, indent=4)))
    return record_path


def data_generator(data_path):
    """
    生成 f(x)=sin(x) 的训练数据。

    参数:
    - data_path (str): 将生成的数据保存到该路径。
    """
    x = np.random.rand(10000, 1) * 2 * np.pi - np.pi  # [-pi, pi)
    y = np.sin(x)
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    np.savez(
        data_path,
        x_train=x[:8000],
        y_train=y[:8000],
        x_valid=x[8000:9000],
        y_valid=y[8000:9000],
        x_test=x[9000:],
        y_test=y[9000:],
    )


def data_preprocess(raw_data_path, data_path, shuffle=True):
    x_list = []
    y_list = []
    for item in os.listdir(raw_data_path):
        index = int(item) - 1  # 汉字的分类索引
        item_path = os.path.join(raw_data_path, item)
        for bmp in os.listdir(item_path):
            bmp_path = os.path.join(item_path, bmp)
            img = cv2.imread(bmp_path, cv2.IMREAD_GRAYSCALE)

            # TODO: 数据增强

            flatten_img = img.flatten()
            x_list.append(flatten_img)
            y_list.append(index)

    x = np.vstack(x_list)
    print("x generated, shape: {}".format(x.shape))
    y = np.array(y_list)
    print("y generated, shape: {}".format(y.shape))
    num_samples = x.shape[0]

    if shuffle:
        print("Shuffling x and y")
        random_indices = np.arange(num_samples)
        np.random.shuffle(random_indices)
        x = x[random_indices]
        y = y[random_indices]

    print("Splitting train, validation dataset")
    train_num = int(num_samples * 0.9)
    valid_num = num_samples - train_num

    print("Saving data")
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    np.savez(
        data_path,
        x_train=x[:train_num],
        y_train=y[:train_num],
        x_valid=x[-valid_num:],
        y_valid=y[-valid_num:],
    )


def test_preprocess(raw_data_path, data_path, shuffle=True):
    x_list = []
    y_list = []
    for item in os.listdir(raw_data_path):
        index = int(item) - 1  # 汉字的分类索引
        item_path = os.path.join(raw_data_path, item)
        for bmp in os.listdir(item_path):
            bmp_path = os.path.join(item_path, bmp)
            img = cv2.imread(bmp_path, cv2.IMREAD_GRAYSCALE)
            flatten_img = img.flatten()
            x_list.append(flatten_img)
            y_list.append(index)

    x = np.vstack(x_list)
    print("x generated, shape: {}".format(x.shape))
    y = np.array(y_list)
    print("y generated, shape: {}".format(y.shape))
    num_samples = x.shape[0]

    if shuffle:
        print("Shuffling x and y")
        random_indices = np.arange(num_samples)
        np.random.shuffle(random_indices)
        x = x[random_indices]
        y = y[random_indices]

    print("Saving data")
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    np.savez(data_path, x_test=x, y_test=y)


if __name__ == "__main__":
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    os.chdir(current_directory)

    seed_everything(42)

    test_preprocess("./data/char/test_data", "./data/char/test.npz")
    # data_preprocess("./data/char/train_raw", "./data/char/data.npz")

    exit(0)
