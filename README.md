# MedSeg
Image segmentation of CT images (LiTS dataset). In the future also weakly labeled deep learning image segmentation of CT Images. The model with losses and optimizers are stored in MedSeg/model. Training and testing is done with `train_one_epoch` and `test_one_epoch` stored in *train.py* and *test.py* respectively. *controller2d.py* is the main controller for analyzing the data as independent 2d image slices. *preprocessing.py* preforms the preprocessing given the original LiTS training set of 130 CT volumetric images.

Segmentation of LiTS CT volumetric image dataset. The project was started on a template made by Branislav Hollander (https://github.com/branislav1991/PyTorchProjectFramework). The whole project is heavely influenced by https://github.com/assassint2017/MICCAI-LITS2017.


## How to run
Run the following lines from terminal to train network with respect to lesion segmentations with a run id of 1. Note that focus, mode and runid are all specified in *MedSeg/config.py*, but will be overridden with the terminal arguments if provided.
```
$ pip install -r requirements.txt
$ python controller2d.py --mode train --focus lesion --seed 0 --runid 1
```
`mode`: 'train' or 'test' (without quotes). If testing is chosen, an already trained model will be applied. The model state should be stored under the key "model_state_dict" in a .pth-file. The path to the .pth-file should specified as a string in *MedSeg/config.py* under the key "init_2d_model_state".

`focus`: 'liver' or 'lesion' (without quotes). Decide which labels to use. Both cases use binary segmentation maps with a value of 1 for all pixels corresponding to image pixels of the specified focus.

`seed`: integer or None. Make a result reproducible by specifying a number. All runs given the same seed will preduce the exact same result. If None or not specified any result is partly affected by psudorandom precesses and will not be possible to reproduce exactly.

`runid`: integer. Identification number of run to make it easier to distriguish saved states and monitoring of training and testing.

It is strongly advised to apply a Nvidia gpu.
The training can be performed on a CPU, but it will take more than several days and probably several weeks to complete.


## Requirements
Specified in *MedSeg/requirements.txt*.
