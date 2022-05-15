import torch.nn as nn
import torch


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
        self.pool2 = nn.MaxPool2d((2, 2))
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.pool3 = nn.MaxPool2d((2, 2))
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.block5 = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.block6 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.block7 = nn.Sequential(
            nn.Conv2d(96, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.final = nn.Conv2d(32, 2, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        out1 = self.block1(x)
        out_pool1 = self.pool1(out1)
        out2 = self.block2(out_pool1)
        out_pool2 = self.pool1(out2)
        out3 = self.block3(out_pool2)
        out_pool3 = self.pool1(out3)
        out4 = self.block4(out_pool3)
        out_up1 = self.up1(out4)

        # return out_up1
        out5 = torch.cat((out_up1, out3), dim=1)
        out5 = self.block5(out5)
        out_up2 = self.up2(out5)
        out6 = torch.cat((out_up2, out2), dim=1)
        out6 = self.block6(out6)
        out_up3 = self.up3(out6)
        out7 = torch.cat((out_up3, out1), dim=1)
        out7 = self.block7(out7)
        out = self.final(out7)
        return out
