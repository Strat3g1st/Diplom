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
        self.pool1 = nn.MaxPool2d((2, 2))
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.final = nn.Conv2d(32, 2, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        x = self.pool1(self.block1(x))
        x = self.up1(self.block2(x))
        x = self.block3(x)
        x = self.final(x)
        return x
