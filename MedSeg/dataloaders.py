from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import torch
import os
from utils import load_volume, load_slice
# from tqdm import tqdm
import config


class Testset(Dataset):
    def __init__(self, length=10, sidelength=512):
        super().__init__()
        self.sl = sidelength
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx > len(self):
            raise IndexError

        return {'vol': torch.randn(1, self.sl, self.sl, self.sl),
                'lab': torch.randint(0, 3, (self.sl, self.sl, self.sl))}


class LiTSDataset(Dataset):
    """LiTS dataset."""
    _options_focus = ['liver', 'lesion']
    # _options_label_type = ['segmentation', 'pixelcount', 'binary']
    def __init__(self, datapath, focus="liver", transform=None):  # label_type='segmentation', 
        """
            Arguments
            ---------
                datapath : str
                    path to dir where LiTS data are stored.
                    Assumes there exists paths: 
                    - path/volumes/<volumes nii-files>
                    - path/labels_liver/<segmentation nii-files> in corresponding order.
                
                transform : list
                    Transforms to apply to data.
        """
        super().__init__()
        ## Assertions about inputs
        focus_msg = "focus must be among {}".format(self._options_focus)
        assert focus in self._options_focus, focus_msg

        self.datapath = datapath
        self.volpath = os.path.join(datapath, "volumes/")
        self.volnames = os.listdir(self.volpath)
        self.focus = focus
        self.labpath = os.path.join(datapath, f"labels_{focus}/")
        self.labnames = os.listdir(self.labpath)
        self.transform = transform
        
    def __len__(self):
        return len(self.volnames)

    def __getitem__(self, idx):
        """
            Returns
            -------
                sample : dict
                    'vol': torch.tensor of input volumetric image
                    'lab': torch.tensor of input volumetric image
                idx : int
                    Index of image as stored in files.
        """
        vol = load_volume(os.path.join(self.volpath, self.volnames[idx]))
        vol = vol[None, ...]
        lab = load_volume(os.path.join(self.labpath, self.labnames[idx]))

        vol = torch.tensor(vol, dtype=torch.float32)
        lab = torch.tensor(lab, dtype=torch.float32)
        
        sample = {'vol': vol, 'lab': lab}

        if self.transform:
            self.transform(sample)
        return sample


class LiTSDataset2d(Dataset):
    """LiTS dataset converted to separate 2d slices."""
    _options_focus = ['liver', 'lesion']
    def __init__(self, datapath, focus="liver", max_length=None, transform=None):
        """
            Arguments
            ---------
                datapath : str
                    path to dir where LiTS data are stored.
                    Assumes there exists paths: 
                    - path/slices/<slice .npy-files>
                    - path/labels_liver/<segmentation npy-files> with corresponding indices.
                [focus] : ['liver', 'lesion']
                    Default: 'liver'
                    What segmentation labels to use.
                [max_length] : int
                    Number of samples to use.
                [transform] : list
                    Transforms to apply to data.
        """
        super().__init__()
        ## Assertions about inputs
        focus_msg = "focus must be among {}".format(self._options_focus)
        assert focus in self._options_focus, focus_msg

        self.datapath = datapath
        self.slicepath = os.path.join(datapath, "slices/")
        self.slicenames = os.listdir(self.slicepath)
        self.max_length = max_length
        self.focus = focus
        self.labpath = os.path.join(datapath, f"labels_{focus}/")
        self.labnames = os.listdir(self.labpath)
        self.transform = transform
        
    def __len__(self):
        if self.max_length is None:
            return len(self.slicenames)
        else:
            return self.max_length

    def __getitem__(self, idx):
        """
            Returns
            -------
                sample : dict
                    'slice': torch.tensor of input slice image
                    'lab': torch.tensor of input slice label
                idx : int
                    Index of image as stored in files.
        """
        img = load_slice(os.path.join(self.slicepath, self.slicenames[idx]))
        lab = load_slice(os.path.join(self.labpath, self.labnames[idx]))

        img = torch.tensor(img, dtype=torch.float32).unsqueeze(dim=0)
        lab = torch.tensor(lab, dtype=torch.float32).unsqueeze(dim=0)
        
        sample = {'vol': img, 'lab': lab}

        if self.transform:
            self.transform(sample)
        return sample


if __name__ == "__main__":
    dataset = Testset(length=5)
    dataset = LiTSDataset2d("datasets/preprocessed_2d/train/", max_length=30)
    dataloader = DataLoader(dataset, batch_size=20)
    
    ## View data with dataloader
    sample = next(iter(dataloader))
    vol, lab = sample.values()
    print(vol.shape, lab.shape)
