import os
import sys
import random
import torch
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
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def random_rotate(image, max_angle):
    angle = random.uniform(-max_angle, max_angle)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
    return rotated_image


def add_noise(image, noise_stddev=1):
    noise = np.random.normal(0, noise_stddev, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image


def data_preprocess(raw_data_path, data_path, augment=False):
    x_list = []
    y_list = []
    for item in os.listdir(raw_data_path):
        index = int(item) - 1  # 汉字的分类索引
        item_path = os.path.join(raw_data_path, item)
        for bmp in os.listdir(item_path):
            bmp_path = os.path.join(item_path, bmp)
            img = cv2.imread(bmp_path, cv2.IMREAD_GRAYSCALE)
            x_list.append(img)
            y_list.append(index)

    x = np.stack(x_list, axis=0)
    y = np.array(y_list)
    print("x.shape: {}, y.shape: {}".format(x.shape, y.shape))
    num_samples = x.shape[0]

    print("Shuffling x and y")
    random_indices = np.random.permutation(num_samples)
    x = x[random_indices]
    y = y[random_indices]

    print("Splitting train and validation dataset")
    train_num = int(num_samples * 0.9)  # 9:1
    valid_num = num_samples - train_num
    x_valid = x[-valid_num:]
    y_valid = y[-valid_num:]

    if augment:
        print("Augmenting train dataset")
        x_augment_list = []
        y_augment_list = []
        for i in range(train_num):
            # 原图片
            x_augment_list.append(x[i])
            y_augment_list.append(y[i])
            # 添加噪声后的图片
            x_augment_list.append(add_noise(x[i]))
            y_augment_list.append(y[i])

        x_train = np.stack(x_augment_list, axis=0)
        y_train = np.array(y_augment_list)
        print(f"x_train_aug.shape: {x_train.shape}, y_train_aug.shape: {y_train.shape}")
        train_aug_size = x_train.shape[0]

        print("Shuffling x_train_aug and y_train_aug")
        indices = np.random.permutation(train_aug_size)
        x_train = x_train[indices]
        y_train = y_train[indices]
    else:
        x_train = x[:train_num]
        y_train = y[:train_num]

    # 计算训练集的均值和标准差
    print("Calculating x_train's mean and std")
    mean = np.mean(x_train)
    std = np.std(x_train)

    print("Saving data")
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    np.savez(
        data_path,
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid,
        mean=mean,
        std=std,
    )


if __name__ == "__main__":
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    os.chdir(current_directory)

    seed_everything(42)

    # data_preprocess("./data/raw", "./data/data.npz", augment=False)
    data_preprocess("./data/raw", "./data/data_aug0.npz", augment=True)

    item_path = os.path.join("./data/raw", "1")
    bmp_path = os.path.join(item_path, "1.bmp")
    img: np.ndarray = cv2.imread(bmp_path, cv2.IMREAD_GRAYSCALE)
    print(img.shape)
    print(img.dtype)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)  # wait for any key
    # cv2.imshow("rotated_img", random_rotate(img, 90))
    # cv2.waitKey(0)  # wait for any key
    # cv2.imshow("noisy_img", add_noise(img))
    # cv2.waitKey(0)  # wait for any key

    # data_npz = np.load("./data/data.npz")
    # print(data_npz["x_train"].dtype)
    # print(data_npz["x_valid"].dtype)
    # print(data_npz["x_train"].shape)
    # print(data_npz["y_train"].shape)
    # print(data_npz["x_valid"].shape)
    # print(data_npz["y_valid"].shape)
    # print(data_npz["mean"])
    # print(data_npz["std"])

    data_aug0_npz = np.load("./data/data_aug0.npz")
    print(data_aug0_npz["x_train"].dtype)
    print(data_aug0_npz["x_valid"].dtype)
    print(data_aug0_npz["x_train"].shape)
    print(data_aug0_npz["y_train"].shape)
    print(data_aug0_npz["x_valid"].shape)
    print(data_aug0_npz["y_valid"].shape)
    print(data_aug0_npz["mean"])
    print(data_aug0_npz["std"])

    exit(0)
