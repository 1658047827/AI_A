import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self, in_channels, out_features):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(16, 120, kernel_size=5)
        self.fc6 = nn.Linear(120, 84)
        self.fc7 = nn.Linear(84, out_features)

    def forward(self, x):
        x = self.pool2(F.relu(self.conv1(x)))
        x = self.pool4(F.relu(self.conv3(x)))
        x = F.relu(self.conv5(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        y = self.fc7(x)
        return y
