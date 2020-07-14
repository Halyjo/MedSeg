import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class TverskyLoss(nn.Module):
    """
    Modified version of src copied from https://github.com/assassint2017/MICCAI-LITS2017
    """    
    def __init__(self, alpha=None, beta=None):
        super().__init__()

        if alpha is None and beta is None:
            self.alpha = 0.3
            self.beta = 0.7
        else:
            assert alpha+beta==1, "alpha and beta must sum to 1."
            self.alpha = alpha
            self.beta = beta

    def forward(self, pred, target):
        pred = pred.squeeze(dim=1)
        smooth = 1
        num = (pred * target).sum(dim=1).sum(dim=1).sum(dim=1) 
        false_pos = (pred * (1 - target)).sum(dim=1).sum(dim=1).sum(dim=1)
        false_neg = ((1 - pred) * target).sum(dim=1).sum(dim=1).sum(dim=1)
        den = (num + self.alpha * false_pos + self.beta * false_neg + smooth)
        dice = num/den
        return torch.clamp((1 - dice).mean(), 0, 2)


class MSELossPixelCount(nn.Module):
    """
    Mean Squared Error Loss of pixelcound given predictions and labels as images with binary values.
    """
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, pred, target):
        pred = torch.round(pred)
        pred = pred.type(torch.bool).sum()
        target = target.type(torch.bool).sum()
        return self.mse(pred, target)


class BCELossBinary(nn.Module):
    """
    Binary cross entopy loss (occurance of lesion in image or not)
    given predictions and labels as images with binary values.
    """
    def __init__(self):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss


class CELoss(nn.Module):
    """
    Cross entropy loss for multidimentional images (3D+).
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.celoss = nn.CrossEntropyLoss(**kwargs)

    def forward(self, pred, lab):
        pred = pred.view(1, 3, -1)
        lab = lab.view(1, -1)
        return self.celoss(pred, lab.long())


class WeightedCrossEntropyLoss(nn.Module):
    """WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf.
    Source: https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/losses.py.
    """

    def __init__(self, ignore_index=-1):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        weight = self._class_weights(pred)
        pred = pred.view(1, 3, -1)
        lab = target.view(1, -1)
        return F.cross_entropy(input, target, weight=weight, ignore_index=self.ignore_index)

    @staticmethod
    def _class_weights(pred):
        # normalize the input first
        pred = F.softmax(pred, dim=1)
        flattened = torch.flatten(pred)
        nominator = (1. - flattened).sum(-1)
        denominator = flattened.sum(-1)
        class_weights = Variable(nominator / denominator, requires_grad=False)
        return class_weights


class DiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred = pred.squeeze(dim=1)
        smooth = 1
        # dice系数的定义
        num = 2 * (pred * target).sum()
        den = pred.sum() + target.sum() + smooth
        dice = num/den

        # 返回的是dice距离
        return torch.clamp((1 - dice).mean(), 0, 1)

if __name__ == "__main__":
    pass