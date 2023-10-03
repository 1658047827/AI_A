import os
import argparse
import logging
import numpy as np
from utils import Dataset, DataLoader
from init import seed_everything, dump_args


class CharDataset(Dataset):
    def __init__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


# 先将工作目录切换到 fit_char.py 所在文件夹下
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
os.chdir(current_directory)


parser = argparse.ArgumentParser()
parser.add_argument("--epoches", default=2000, type=int)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--learning_rate", default=0.03, type=float)
parser.add_argument("--record_path", default="./record/char", type=str)
parser.add_argument("--save_path", default="./save/char/best_model.pkl", type=str)
parser.add_argument("--data_path", default="./data/char/data.npz", type=str)
parser.add_argument("--random_seed", default=42, type=int)
parser.add_argument(
    "--mode", default="test", type=str, choices=["train", "test", "train_and_test"]
)
args = vars(parser.parse_args())
record_path = dump_args(args, args["record_path"], args["mode"])


if __name__ == "__main__":
    seed_everything(args["random_seed"])
