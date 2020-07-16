"""
Modified version of: 
https://github.com/assassint2017/MICCAI-LITS2017/blob/master/data_prepare/get_training_set.py

"""

import os
import numpy as np
import sys
# sys.path.append(os.path.split(sys.path[0])[0])
import shutil
from time import time

import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
from scipy import ndimage, misc
from config import config
from utils import counter


def preprocess2d(mode='train'):
    """
    Split up all 3d volumes into separate 2d slice images and store as separate files.
    Volumes are fetched from directory specified in config. 
    """
    ## Generator to generate indices for slices.
    numgen = counter()
    dirpaths = [
        config[f"dst2d_labels_liver_path"],
        config[f"dst2d_labels_lesion_path"],
        config[f"dst2d_slices_path"],
    ]
    for p in dirpaths:
        if not os.path.exists(p):
            os.makedirs(p)

    ## Processing
    start = time()
    for f in tqdm(os.listdir(config[f"{config["mode"]}_volumes_path"]), desc=f"{config["mode"]} 2d processing"):
        ## Get Volume matrices
        ct = sitk.ReadImage(os.path.join(config[f"{config["mode"]}_volumes_path"], f), sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)

        seg = sitk.ReadImage(os.path.join(config[f"{config["mode"]}_labels_path"], f.replace('volume', 'segmentation')), sitk.sitkUInt8)
        seg_array = sitk.GetArrayFromImage(seg)
        ## TODO: Change if looking for tumors
        ## Make all cancer labels to liverlabels (for liver segmentation, not tumor segmentation)
        liver_array = np.zeros_like(seg_array)
        lesion_array = np.zeros_like(seg_array)

        liver_array[seg_array >= 1] = 1
        lesion_array[seg_array > 1] = 1
        
        ## Clip upper and lower values of CT images
        ct_array[ct_array > config["upper"]] = config["upper"]
        ct_array[ct_array < config["lower"]] = config["lower"]

        ## Pick out relevant slices
        z = np.any(liver_array, axis=(1, 2))
        start_slice, end_slice = np.where(z)[0][[0, -1]]
        start_slice = max(0, start_slice - config["expand_slice"])
        end_slice = min(liver_array.shape[0] - 1, end_slice + config["expand_slice"])

        ct_array = ct_array[start_slice:end_slice, :, :]
        liver_array = liver_array[start_slice:end_slice, :, :]
        lesion_array = lesion_array[start_slice:end_slice, :, :]
        if ct_array.shape[0] != liver_array.shape[0]:
            breakpoint()

        ## Store each slice as .mat numpy array
        for i in range(ct_array.shape[0]):
            idx = next(numgen)
            slice_filename = "slice_{:05}".format(idx)
            segmentation_filename = "segmentation_{:05}".format(idx)
            np.save(os.path.join(config[f"dst2d_slices_path"], slice_filename), 
                    ct_array[i, ...])
            np.save(os.path.join(config[f"dst2d_labels_liver_path"], segmentation_filename),
                    liver_array[i, ...])
            np.save(os.path.join(config[f"dst2d_labels_lesion_path"], segmentation_filename),
                    lesion_array[i, ...])

    print("Finished {} preprocessing in: {:02}s".format(config["mode"], time() - start))


def preprocess3d(mode='train'):
    """
    Preprocess original data given configuration (only port for preprocessing).
    """
    ## Create dst folders if missing
    dirpaths = [
        config[f"dst_{config["mode"]}_labels_liver_path"],
        config[f"dst_{config["mode"]}_labels_lesion_path"],
        config[f"dst_{config["mode"]}_volumes_path"],
    ]
    for p in dirpaths:
        if not os.path.exists(p):
            os.mkdir(p)

    ## Processing
    start = time()
    for f in tqdm(os.listdir(config[f"{config["mode"]}_volumes_path"]), desc=f"{config["mode"]} processing"):
        ## Get Volume matrices
        
        ct = sitk.ReadImage(os.path.join(config[f"{config["mode"]}_volumes_path"], f), sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)

        seg = sitk.ReadImage(os.path.join(config[f"{config["mode"]}_labels_path"], f.replace('volume', 'segmentation')), sitk.sitkUInt8)
        seg_array = sitk.GetArrayFromImage(seg)
        ## TODO: Change if looking for tumors
        ## Make all cancer labels to liverlabels (for liver segmentation, not tumor segmentation)
        liver_array = np.zeros_like(seg_array)
        lesion_array = np.zeros_like(seg_array)

        liver_array[seg_array >= 1] = 1
        lesion_array[seg_array > 1] = 1
        
        ## Clip upper and lower values of CT images
        ct_array[ct_array > config["upper"]] = config["upper"]
        ct_array[ct_array < config["lower"]] = config["lower"]

        # 对CT数据在横断面上进行降采样,并进行重采样,将所有数据的z轴的spacing调整到1mm:
        # Downsampling the CT data on the cross section and re-sampling, adjusting the z-spacing of all data to 1mm
        # ct_array = ndimage.zoom(ct_array, (ct.GetSpacing()[-1] / config["slice_thickness"], config["down_scale"], config["down_scale"]), order=3)
        # seg_array = ndimage.zoom(seg_array, (ct.GetSpacing()[-1] / config["slice_thickness"], config["down_scale"]*2, config["down_scale"]*2), order=0)
        
        ct_array = ndimage.zoom(ct_array, (ct.GetSpacing()[-1] / config["slice_thickness"], config["down_scale"], config["down_scale"]), 
                                order=3)
        liver_array = ndimage.zoom(liver_array, (ct.GetSpacing()[-1] / config["slice_thickness"], config["down_scale"]*2, config["down_scale"]*2), 
                                 order=1)
        lesion_array = ndimage.zoom(lesion_array, (ct.GetSpacing()[-1] / config["slice_thickness"], config["down_scale"]*2, config["down_scale"]*2), 
                                 order=1)
        # seg_array = ndimage.zoom(seg_array, (ct.GetSpacing()[-1] / config["slice_thickness"], 1, 1), order=0)
        # 找到肝脏区域开始和结束的slice，并各向外扩张slice
        z = np.any(liver_array, axis=(1, 2))
        start_slice, end_slice = np.where(z)[0][[0, -1]]

        # 两个方向上各扩张slice
        start_slice = max(0, start_slice - config["expand_slice"])
        end_slice = min(liver_array.shape[0] - 1, end_slice + config["expand_slice"])

        ## Make depth divisible by 8 by cropping
        depth = end_slice - start_slice
        rest = depth % 8
        low_rest = rest//2
        high_rest = low_rest + rest%2

        start_slice += low_rest
        end_slice -= high_rest


        # 如果这时候剩下的slice数量不足size，直接放弃该数据，这样的数据很少,所以不用担心
        if end_slice - start_slice + 1 < config["size"]:
            print('!!!!!!!!!!!!!!!!')
            print(f, 'have too little slice', ct_array.shape[0])
            print('!!!!!!!!!!!!!!!!')
            continue

        ct_array = ct_array[start_slice:end_slice, :, :]
        liver_array = liver_array[start_slice:end_slice, :, :]
        lesion_array = lesion_array[start_slice:end_slice, :, :]
        if ct_array.shape[0] != liver_array.shape[0]:
            breakpoint()
        # 最终将数据保存为nii
        new_ct = sitk.GetImageFromArray(ct_array)

        new_ct.SetDirection(ct.GetDirection())
        new_ct.SetOrigin(ct.GetOrigin())
        new_ct.SetSpacing((ct.GetSpacing()[0] * int(1 / config["down_scale"]), ct.GetSpacing()[1] * int(1 / config["down_scale"]), config["slice_thickness"]))

        liver_sitk_lab = sitk.GetImageFromArray(liver_array)

        liver_sitk_lab.SetDirection(ct.GetDirection())
        liver_sitk_lab.SetOrigin(ct.GetOrigin())
        liver_sitk_lab.SetSpacing((ct.GetSpacing()[0], ct.GetSpacing()[1], config["slice_thickness"]))

        lesion_sitk_lab = sitk.GetImageFromArray(lesion_array)

        lesion_sitk_lab.SetDirection(ct.GetDirection())
        lesion_sitk_lab.SetOrigin(ct.GetOrigin())
        lesion_sitk_lab.SetSpacing((ct.GetSpacing()[0], ct.GetSpacing()[1], config["slice_thickness"]))

        ## Store image in config["new_ct_path"]
        sitk.WriteImage(new_ct, os.path.join(config[f"dst_{config["mode"]}_volumes_path"], f))
        sitk.WriteImage(liver_sitk_lab, os.path.join(config[f"dst_{config["mode"]}_labels_liver_path"], f.replace('volume', 'segmentation').replace('.nii', '.nii.gz')))
        sitk.WriteImage(lesion_sitk_lab, os.path.join(config[f"dst_{config["mode"]}_labels_lesion_path"], f.replace('volume', 'segmentation').replace('.nii', '.nii.gz')))
    print("Finished {} preprocessing in: {:02}s".format(config["mode"], time() - start))


if __name__ == '__main__':
    # preprocess2d('train')
    # preprocess2d('test')    
    preprocess3d('train')
    preprocess3d('test')

