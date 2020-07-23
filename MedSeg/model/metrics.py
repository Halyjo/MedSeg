"""
Metrics.
"""

import numpy as np
import scipy.spatial as spatial
import scipy.ndimage.morphology as morphology
import torch


class Metrics():
    def __init__(self, pred_mask, real_mask, mode=None):
        """Class to calculate metrics given prediction mask and segmented mask.

            Arguments
            ---------
                real_mask, pred_mask : torch.Tensor
                    shape: Any as long as real_mask and pred_mask have the same shape.
                        Eg: (N, C, D, H, W)
                            N: Number of samples
                            C: Number of channels
                            D: Depth
                            H: Height
                            W: Width

        """
        self.real_mask = real_mask.type(torch.bool)
        self.pred_mask = torch.round(pred_mask).type(torch.bool)
        self.mode = "" if mode is None else mode + "_"

    def get_dice_coefficient(self):
        """

        :return: dice系数 dice系数的分子 dice系数的分母(后两者用于计算dice_global)
        """
        intersection = (self.real_mask * self.pred_mask).sum().float()
        union = self.real_mask.sum().float() + self.pred_mask.sum().float()
        # if union.sum() == 0:
        #     dice = 1
        # else:
        dice = (2 * intersection + 1e-5) / (union + 1e-5)
        return dice, 2 * intersection, union

    def get_jaccard_index(self, to_memory=False):
        intersection = (self.real_mask * self.pred_mask).sum().float() + 1e-5
        union = (self.real_mask | self.pred_mask).sum().float() + 1e-5
        # if (self.real_mask.sum().float() + self.real_mask.sum().float()) == 0:
        #     result = 1
        # else:
        result = intersection / union
        return result

    def get_VOE(self):
        """Volumetric Overlap Error
        """

        return 1 - self.get_jaccard_index()

    def get_RVD(self):
        """Relative Volume Difference
        """

        return (self.pred_mask.sum().float() - self.real_mask.sum().float()) / self.real_mask.sum().float()

    def get_FNR(self):
        """False negative rate
        """
        fn = self.real_mask.sum().float() - (self.real_mask * self.pred_mask).sum().float()
        union = (self.real_mask | self.pred_mask).sum().float()

        return fn / union

    def get_FPR(self):
        """False positive rate
        """
        fp = self.pred_mask.sum().float() - (self.real_mask * self.pred_mask).sum().float()
        union = (self.real_mask | self.pred_mask).sum().float()

        return fp / union

    def get_conmat(self, n_classes=2):
        """Calculate confusion matrix from torch tensors.

            Arguments
            ---------
                self.pred_mask, self.real_mask : torch.tensor
                    shape: any but equal
                    Prediction mask and label mask to compare.

                [n_classes] : int
                    Default: list(range(# unique values in lab-tensor))
        """
        mat = torch.zeros((n_classes, n_classes))
        for i in range(n_classes):
            for j in range(n_classes):
                mat[i, j] = torch.sum((self.real_mask == i) * (self.pred_mask == j))
        return mat

    def get_metric_dict(self):
        """
        Get dictionary with all metrics (except confusion matrix) in a
        format that is easely converted to a pandas DataFrame.
        """
        diceparts = self.get_dice_coefficient()
        metric_dict = {
            f'{self.mode}dice': diceparts[0],
            f'{self.mode}dice_numerator': diceparts[1],
            f'{self.mode}dice_denominator': diceparts[2],
            f'{self.mode}iou': self.get_jaccard_index(),
            f'{self.mode}voe': self.get_VOE(),
            f'{self.mode}rvd': self.get_RVD(),
            f'{self.mode}fnr': self.get_FNR(),
            f'{self.mode}fpr': self.get_FPR(),
            # f'{self.mode}conmat': self.get_conmat(),
        }
        return metric_dict


def test_Metrics():
    n = 100
    w = 50
    iou = []
    dice = []
    lab = torch.zeros((n, n))
    lab[:w, :w] = 1
    for i in range(n):
        pred = torch.zeros((n, n))
        pred[i:i+w, :w] = 1
        M = Metrics(pred, lab)
        dice.append(M.get_dice_coefficient()[0])
        iou.append(M.get_jaccard_index())

    import matplotlib.pyplot as plt
    plt.plot(np.arange(len(iou)), dice, label="dice")
    plt.plot(np.arange(len(iou)), iou, label="iou")
    plt.legend()
    plt.show()
    plt.plot(dice, iou)
    plt.xlabel("dice")
    plt.ylabel("iou")

    plt.show()

    intersection = 6
    union = 11
    sum_of_parts = 17

    true_dice = 2*intersection/sum_of_parts
    true_iou = intersection/union

    M = Metrics(pred, lab)
    assert np.isclose(true_dice, M.get_dice_coefficient()[0].item())
    assert np.isclose(true_iou, M.get_jaccard_index().item())


if __name__ == "__main__":
    test_Metrics()
    exit()
    pred = torch.randint(0, 2, (10, 10))
    lab = torch.randint(0, 2, (10, 10))
    metrics = Metrics(pred, lab)
    md = metrics.get_metric_dict()
    for k, v in md.items():
        print("{:<20}: {}".format(k, *v))

