from torch.utils.data import Dataset
from torchvision.transforms import Resize, ToTensor


class CharDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.len = x.shape[0]

    def __getitem__(self, index):
        return {"x": self.x[index], "y": self.y[index]}

    def __len__(self):
        return self.len
