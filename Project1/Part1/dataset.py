import numpy as np
from utils import Dataset


class ToTensor:
    """
    [0, 255] -> [0.0, 1.0]
    """
    def __init__(self):
        pass

    def __call__(self, pic):
        return pic / 255.0


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, pic):
        return (pic - self.mean) / self.std


class SinDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return {"x": self.x[index], "y": self.y[index]}

    def __len__(self):
        return self.x.shape[0]


class CharDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __getitem__(self, index):
        image = self.x[index]
        label = self.y[index]
        if self.transform is not None:
            image = self.transform(image)
        return {"x": image, "y": label}

    def __len__(self):
        return self.x.shape[0]
