import torch.nn as nn


class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.final = nn.Conv2d(32, 2, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        x = self.block1(x)
        x = self.final(x)
        return x
