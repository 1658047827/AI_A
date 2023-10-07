import os
import argparse
import logging
import numpy as np
from utils import Dataset, DataLoader
from init import seed_everything, dump_args, data_preprocess
from model import MLPClassifier, accuracy
from nn import CrossEntropyLoss


class CharDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.len = x.shape[0]

    def __getitem__(self, index):
        return {"x": self.x[index], "y": self.y[index]}

    def __len__(self):
        return self.len


# 先将工作目录切换到 fit_char.py 所在文件夹下
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
os.chdir(current_directory)


parser = argparse.ArgumentParser()
parser.add_argument("--epoches", default=2000, type=int)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--learning_rate", default=0.0005, type=float)
parser.add_argument("--record_path", default="./record/char", type=str)
parser.add_argument("--save_path", default="./save/char/best_model.pkl", type=str)
parser.add_argument("--raw_data_path", default="./data/char/train_raw", type=str)
parser.add_argument("--data_path", default="./data/char/data.npz", type=str)
parser.add_argument("--random_seed", default=42, type=int)
parser.add_argument(
    "--mode", default="train", type=str, choices=["train", "test", "train_and_test"]
)
args = vars(parser.parse_args())
record_path = dump_args(args, args["record_path"], args["mode"])


if __name__ == "__main__":
    seed_everything(args["random_seed"])

    if not os.path.exists(args["data_path"]):
        data_preprocess(args["raw_data_path"], args["data_path"], shuffle=True)
    data_npz = np.load(args["data_path"])

    dataset_train = CharDataset(data_npz["x_train"], data_npz["y_train"])
    dataloader_train = DataLoader(dataset_train, args["batch_size"], shuffle=True)

    dataset_valid = CharDataset(data_npz["x_valid"], data_npz["y_valid"])
    dataloader_valid = DataLoader(dataset_valid, args["batch_size"], shuffle=False)

    classifier = MLPClassifier(28 * 28, 12)
    if os.path.exists(args["save_path"]):
        classifier.load_model(args["save_path"])
    if args["mode"] == "train" or args["mode"] == "train_and_test":
        classifier.fit(
            train_loader=dataloader_train,
            valid_loader=dataloader_valid,
            epoches=args["epoches"],
            learning_rate=args["learning_rate"],
            save_path=None,
            log_interval=1,
        )

    # my test
    if args["mode"] != "test" and args["mode"] != "train_and_test":
        exit(0)
    dataset_test = CharDataset(data_npz["x_test"], data_npz["y_test"])
    dataloader_test = DataLoader(dataset_test, args["batch_size"], shuffle=False)

    loss = 0.0
    acc = 0.0
    loss_func = CrossEntropyLoss()
    for batch in dataloader_test:
        result = classifier.predict(batch["x"])
        loss += loss_func(result["predicts"], batch["y"])
        acc += accuracy(result["predicts"], batch["y"])
    loss /= len(dataloader_test)
    acc /= len(dataloader_test)
    logging.info("Test mean loss: {:.6f}, mean accuracy: {:.6f}".format(loss, acc))
