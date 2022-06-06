import torch.nn as nn
import torch


class CNNNet(nn.Module):
    def __init__(self):
        # p = 0.5
        super(CNNNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.Dropout2d(p),
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.Dropout2d(p)
        )
        self.pool1 = nn.MaxPool2d((2, 2))
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.Dropout2d(p),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.Dropout2d(p),
        )
        self.pool2 = nn.MaxPool2d((2, 2))
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.Dropout2d(p),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.Dropout2d(p)
        )
        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.block4 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.Dropout2d(p),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.Dropout2d(p)
        )
        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.block5 = nn.Sequential(
            nn.Conv2d(96, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.Dropout2d(p),
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.Dropout2d(p)
        )
        self.final = nn.Conv2d(32, 2, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        out1 = self.block1(x)
        out_pool1 = self.pool1(out1)
        out2 = self.block2(out_pool1)
        out_pool2 = self.pool1(out2)
        out3 = self.block3(out_pool2)
        out_up1 = self.up1(out3)

        # return out_up1
        out4 = torch.cat((out_up1, out2), dim=1)
        out4 = self.block4(out4)
        out_up2 = self.up2(out4)
        out5 = torch.cat((out_up2, out1), dim=1)
        out5 = self.block5(out5)
        out = self.final(out5)
        return out
