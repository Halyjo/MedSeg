import torch
import torch.nn as nn


class TestNet(nn.Module):
    def __init__(self, drop_rate=0.5):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.AdaptiveAvgPool2d(1)
        print("WARNING!: YOU ARE USING TESTNET")

    def forward(self, inputs, pooling=None):
        pred_img = self.sigmoid(self.conv(inputs))
        pred_cls = self.pool(pred_img).squeeze(3).squeeze(2)
        # print("WARNING!: YOU ARE USING TESTNET")
        if pooling=="gap":
            return pred_cls, pred_img
        return pred_img
