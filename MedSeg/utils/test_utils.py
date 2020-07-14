import utils
import torch
import numpy as np
import SimpleITK as sitk
import os
import shutil


def test_conmat():
    lab = torch.tensor([0,0,0,0,1,1,1,1,0])
    pred = torch.tensor([0,0,0,0,0,0,0,1,1])
    true_neg = 4
    true_pos = 1
    false_neg = 3
    false_pos = 1
    true_conmat = torch.tensor([[true_neg, false_pos],
                                [false_neg, true_pos]], dtype=torch.float32)
    confusion_matrix = utils.conmat(pred, lab, n_classes=2)
    assert torch.all(confusion_matrix == true_conmat)


def test_load_volume():
    path = "../datasets/original/train/ct/volume-0.nii"
    vol_np = utils.load_volume(path)

    # Type
    msg = "Problem in utils.load_volume: Return type is {} and should be {}."
    assert isinstance(vol_np, np.ndarray), msg.format(type(vol_np), np.ndarray)


def test_store():
    path1 = "../datasets/preprocessed_2d/train/slices/slice_00001.npy"
    path2 = "../datasets/preprocessed_2d/train/slices/slice_00002.npy"
    slice1 = torch.tensor(np.load(path1)).unsqueeze(dim=0)
    slice2 = torch.tensor(np.load(path2)).unsqueeze(dim=0)
    batch1 = torch.cat([slice1, slice2], dim=0)

    path1 = "../datasets/original/train/ct/volume-0.nii"
    path2 = "../datasets/original/train/ct/volume-1.nii"
    vol1 = torch.tensor(utils.load_volume(path1))[None, 0:20]
    vol2 = torch.tensor(utils.load_volume(path2))[None, 0:20]
    batch2 = torch.cat([vol1, vol2], dim=0)

    dst = "deleteme/"
    if not os.path.exists(dst):
        os.mkdir(dst)
    utils.store(batch1, dst, format='npy')
    utils.store(batch1, dst, format='nii')
    utils.store(batch2, dst, format='npy')
    utils.store(batch2, dst, format='nii')
    shutil.rmtree(dst)


if __name__ == "__main__":
    test_conmat()
    test_load_volume()
    test_store()