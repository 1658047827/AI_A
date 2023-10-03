import os
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from utils import Dataset, DataLoader
from init import seed_everything, dump_args, data_generator
from model import MLP


class SinDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.len = x.shape[0]

    def __getitem__(self, index):
        return {"x": self.x[index], "y": self.y[index]}

    def __len__(self):
        return self.len


# 先将工作目录切换到 fit_sin.py 所在文件夹下
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
os.chdir(current_directory)


parser = argparse.ArgumentParser()
parser.add_argument("--epoches", default=2000, type=int)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--learning_rate", default=0.03, type=float)
parser.add_argument("--record_path", default="./record/sin", type=str)
parser.add_argument("--save_path", default="./save/sin/best_model.pkl", type=str)
parser.add_argument("--data_path", default="./data/sin/data.npz", type=str)
parser.add_argument("--random_seed", default=42, type=int)
parser.add_argument(
    "--mode", default="test", type=str, choices=["train", "test", "train_and_test"]
)
args = vars(parser.parse_args())
record_path = dump_args(args, args["record_path"], args["mode"])


if __name__ == "__main__":
    seed_everything(args["random_seed"])

    if not os.path.exists(args["data_path"]):
        data_generator(args["data_path"])
    data_npz = np.load(args["data_path"])

    dataset_train = SinDataset(data_npz["x_train"], data_npz["y_train"])
    dataloader_train = DataLoader(dataset_train, args["batch_size"], shuffle=True)

    dataset_valid = SinDataset(data_npz["x_valid"], data_npz["y_valid"])
    dataloader_valid = DataLoader(dataset_valid, args["batch_size"], shuffle=False)

    mlp = MLP(1, 1)
    if os.path.exists(args["save_path"]):
        mlp.load_model(args["save_path"])
    if args["mode"] == "train" or args["mode"] == "train_and_test":
        mlp.fit(
            train_loader=dataloader_train,
            valid_loader=dataloader_valid,
            epoches=args["epoches"],
            learning_rate=args["learning_rate"],
            save_path=args["save_path"],
            log_interval=50,
        )

    # my test
    if args["mode"] != "test" and args["mode"] != "train_and_test":
        exit(0)
    dataset_test = SinDataset(data_npz["x_test"], data_npz["y_test"])
    dataloader_test = DataLoader(dataset_test, args["batch_size"], shuffle=False)

    delta = 0.0
    for batch in dataloader_test:
        predicts = mlp.predict(batch["x"])
        delta += np.sum(np.abs(predicts - batch["y"])) / args["batch_size"]
    delta /= len(dataloader_test)
    logging.info("Test mean error: {:.6f}".format(delta))

    # 绘图对比
    fig, ax = plt.subplots()
    x = np.linspace(-np.pi, np.pi, 1000)
    y = np.sin(x)
    y_fit = mlp.predict(x.reshape(1000, 1)).reshape(1000)
    ax.scatter(x, y, label="Real Sin", s=5)
    ax.plot(x, y_fit, "r", label="Fitted Sin")
    ax.set_title("Real Sin vs Fitted Sin")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    plt.savefig(
        os.path.join(record_path, "sin_comparison.png"), dpi=300, bbox_inches="tight"
    )
    plt.show()
