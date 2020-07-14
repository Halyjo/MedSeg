"""
Modified version of template example by Branislav Hollander: 
    https://github.com/branislav1991/PyTorchProjectFramework.
"""

import json
import SimpleITK as sitk
import numpy as np
import os
import torch
from torch.optim import lr_scheduler
import pandas as pd
import matplotlib.pyplot as plt
import random


def plot_metrics(path):
    """Plot metrics stored as csv-file at path using pandas. 
    
        Arguments
        ---------
            path : str
                path to csv-file to get data to plot.
    """
    df = pd.read_csv(path)
    print(df.mean())
    print(df.std())


def update_cumu_dict(cumu_dict, info_dict):
    """
    Update dictionary with cumulative information with one step of info from infodict.
    Note: Only keys that are present in info_dict will be stored. Only matching keys will
    be updated.

        Arguments
        --------
            cumu_dict : dict
                Format: {"key1": [value11, value12, ...],
                         "key2": [value21, value22, ...]}
            info_dict : dict
                Format: {"key1": value13, 
                         "key2": value23}

        Returns
        -------
            cumu_dict : dict
                Updated with info from info_dict.
    """
    for key, value in info_dict.items():
        if key in cumu_dict:
            cumu_dict[key].append(info_dict[key])
    return cumu_dict


def load_volume(path):
    """
    Loads volumetric image from nii-file.

        Arguments
        ---------
            path : str
                Path to nii-file to be loaded.

        Returns
        -------
            vol : numpy.ndarray
    """
    vol_sitk = sitk.ReadImage(path, sitk.sitkInt16)
    vol = sitk.GetArrayFromImage(vol_sitk)
    return vol


def load_slice(path):
    return np.load(path)


def store(img_batch, dst: str, counter, format='npy'):
    """Store list of 3d torch.tensors to dst as SimpleITK images.

        Arguments
        ---------
            img_batch : torch.Tensor
                shape: (N, C, d1, d2, ..., dn)
    """
    format_choices = ['nii', 'npy']
    assert format in format_choices, f"format must be among: {format_choices}"

    for i in range(img_batch.shape[0]):
        pred = img_batch[i, ...]
        pred = pred.squeeze(dim=0)
        pred_np = pred.detach().cpu().numpy()
        name = "prediction_{:05}".format(next(counter))
        if format == 'nii':
            name = name + ".nii"
            pred_sitk = sitk.GetImageFromArray(pred_np)
            sitk.WriteImage(pred_sitk, os.path.join(dst, name))
        elif format == 'npy':
            np.save(os.path.join(dst, name), pred_np)

        print("Stored {} in {}".format(name, dst))
            

def counter():
    i = 0
    while True:
        i += 1
        yield i


def compute_store_pred(net, dataloader, dst: str, device='cuda'):
    """ Compute and store predictions.

        Arguments
        ---------
            net : torch model
                Trained net.
            dataloader : torch dataloader
                Path to data to predict segmentation of.
            dst : str
                destination
            device : ['cpu', 'cuda']
                If Nvidia gpu is available, set to 'cuda' to apply it.
        
        Returns
        -------
            Resultlist : list
                List of predictions and labels on the format:
                    [[pred0, lab0], [pred1, lab1], ...]
                For calculating metrics.

    """
    resultlist = []
    net.eval()
    for i, sample in enumerate(dataloader):
        vol = sample['vol'].to(device)
        lab = sample['seg'].to(device)
        pred = torch.round(net(vol))
        resultlist.append([pred, lab])
        ## Squeeze channels since binary. 
        pred = pred.squeeze(dim=1)
        ## Squeeze dp-s since batch size=1.
        pred = pred.squeeze(dim=0)
        pred_sitk = sitk.GetImageFromArray(pred.detach().cpu().numpy())
        sitk.WriteImage(pred_sitk, os.path.join(dst, "prediction_{:02}.nii".format(i)))
    return resultlist


def one_hot(v, nclasses, device='cuda'):
    """
    One hot conversion for volumetric images
    """
    if len(v.shape) != 4:
        raise ValueError("Function does not support the data shape.")

    dimofinterest = len(v.shape)-3
    outshape = list(v.shape)
    outshape.insert(dimofinterest, nclasses)
    ## Insert as channel dimension
    outv = torch.zeros(outshape).to(device)
    outv.scatter_(1, v[:, None, ...].long(), value=1)
    return outv


def save_volume(vol, path):
    """
    Saves volumetric image to nii-file.

        Arguments
        ---------
            path : str
                Path for nii-file to be stored as.
                

        Returns
        -------
            None
    """
    array = sitk.GetImageFromArray(vol)
    vol_sitk = sitk.WriteImage(array, path)


def get_conv_shape(inputshape, Cin, Cout, 
                        kernel_size=3, stride=1, 
                        padding=0, dilation=1, T=False,
                        output_padding=0):
        """
        Finds outshape of 3d convolution or transposed convolution (T=True) 
        given inshape on format: (N, Cin, D, H, W).
        """
        if isinstance(output_padding, int):
            output_padding = (output_padding, output_padding, output_padding)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)
        
        outshape = [inputshape[0], Cout]
        if T:
            for i in range(3):
                s = (inputshape[i+2] - 1)*stride[i] - 2*padding[i] + dilation[i]*(kernel_size[i]-1) + output_padding[i] + 1
                outshape.append(s)
        else:
            for i in range(3):
                s = (inputshape[i+2] + 2*padding[i] - dilation[i]*(kernel_size[i]-1) - 1)//stride[i] + 1
                outshape.append(s)
        return outshape


def ensure_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def conmat(pred, lab, n_classes=None):
    """
    Calculate confusion matrix from torch tensors.

        Arguments
        ---------
            pred, lab : torch.tensor
                shape: any but equal
                Prediction mask and label mask to compare.

            [n_classes] : int
                Default: list(range(# unique values in lab-tensor))
    """
    n_classes = n_classes if n_classes is not None else len(torch.unique(lab))
    mat = torch.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            mat[i, j] = torch.sum((lab == i) & (pred == j))
    return mat


def print_shapes(path: str) -> None:
    """
    Prints out shapes of all volumes (.nii-files) in folder specified as path.
    """

    allfiles = sorted(os.listdir(os.path.join(path)))
    files = filter(lambda x: x[-3:] in ["nii", ".gz"], allfiles)


    shapes = []
    for f in files:
        sitk_img = sitk.ReadImage(os.path.join(path, f))
        array = sitk.GetArrayFromImage(sitk_img)
        shapes.append(array.shape)
    print(np.array(shapes))


if __name__ == "__main__":
    plot_metrics("../datasets/preprocessed_quarter_size/te_metrics_test_run12.csv")
    exit()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to folder of files to inspect shapes of")
    args = parser.parse_args()
    print_shapes(args.path)
    # print_shapes("../datasets/preprocessed_quarter_size/train/volumes/")
    # print_shapes("../datasets/preprocessed_quarter_size/train/labels_liver/")
    # print_shapes("../datasets/preprocessed_quarter_size/train/labels_lesion/")