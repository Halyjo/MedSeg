"""
Controller with overview of training and testing deep learining inference net
on CT images sliced up into 2d images.
"""

import utils
import numpy as np
import os
from pprint import pprint
from config import config
from dataloaders import LiTSDataset, LiTSDataset2d, Testset
from preprocessing import preprocess3d
from torch.utils.data import DataLoader
import torch
from model import VNet2d, TverskyLoss, DiceLoss, MSELossPixelCount, DeepVNet2d
from train import train_one_epoch
from test import test_one_epoch
import wandb
from torchsummary import summary
import SimpleITK as sitk
import pandas as pd


## Use GPU if available
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def controller_2d(mode: str, focus: str):
    """
    Controller function for training and testing VNet
    on Volumetric CT images of livers with and without lesions.
    """
    print("Starting!!! :D")
    
    ## Load data sets
    tr_path = os.path.join(config["dst_2d_path"], "train/")
    te_path = os.path.join(config["dst_2d_path"], "test/")

    tr_set = LiTSDataset2d(tr_path, focus=focus, max_length=None)
    te_set = LiTSDataset2d(te_path, focus=focus, max_length=None)
    
    ## Init and load model if specified in config
    net = VNet2d(drop_rate=config["drop_rate"])
    # net = DeepVNet2d(drop_rate=config["drop_rate"])
    # net = NestedUNet(2, 1)

    print(config["init_2d_model_state"])
    if config["init_2d_model_state"] is not None:
        print("Attemt fetching of state dict at: ", config["init_2d_model_state"])
        state_dict = torch.load(config["init_2d_model_state"])
        net.load_state_dict(state_dict)
        print("Successfully loaded net")
    net.to(device)
    
    ## Get resulting metrics and store predictions
    if mode == 'test':
        net.eval()

        tr_dataloader = DataLoader(tr_set)
        if not os.path.exists(config[f"dst2d_train_pred_{focus}_path"]):
            os.mkdir(config[f"dst2d_train_pred_{focus}_path"])
        train_info = test_one_epoch(net, tr_dataloader, device, 
                                    1, 1, wandblog=False, dst_path=config[f"dst2d_train_pred_{focus}_path"])
        tr_df = pd.DataFrame(train_info)
        tr_df.to_csv(os.path.join(config["dstpath"], 
                                  "tr_metrics_{}_{}_run{:02}.csv".format(mode, focus, config["runid"])))

        te_dataloader = DataLoader(te_set)
        test_info = test_one_epoch(net, te_dataloader, device, 
                                   1, 1, wandblog=False, dst_path=None)
        te_df = pd.DataFrame(test_info)
        te_df.to_csv(os.path.join(config["dstpath"], 
                                  "te_metrics_{}_{}_run{:02}.csv".format(mode, focus, config["runid"])))
        exit()

    ## Monitor process with weights and biases
    wandb.init(config=config)

    optimizer = torch.optim.Adam(params=net.parameters(), 
                                 **config["optim_opts"])
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
    #                                                  config["lr_milestones"], 
    #                                                  config["lr_milestone_scalar"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, config["lr_decay_rate"], last_epoch=-1)
    # critic = TverskyLoss()
    critic = DiceLoss()
    for epoch in range(config["max_epochs"]):
        print(f"Epoch: {epoch}.")

        ## Make dataloaders
        print("Loading data ...")
        tr_dataloader = DataLoader(tr_set, batch_size=config["batch_size"], shuffle=True)
        te_dataloader = DataLoader(te_set, batch_size=config["batch_size"], shuffle=True)

        epochlength = len(tr_dataloader) + len(te_dataloader)

        print("Training ...")
        train_info = train_one_epoch(net, optimizer, critic, tr_dataloader, device, epoch, epochlength)

        ## Get dice global for training
        train_dice_global = np.sum(train_info['train_dice_numerator']) / np.sum(train_info['train_dice_denominator'])
        print("Global train dice at epoch {}: {}".format(epoch, train_dice_global))
        wandb.log({"train_dice_global": train_dice_global})

        print("Testing ...")
        test_info = test_one_epoch(net, te_dataloader, device, epoch + len(tr_dataloader)/epochlength, epochlength)

        ## Get dice global for testing
        test_dice_global = np.sum(test_info['test_dice_numerator']) / np.sum(test_info['test_dice_denominator'])
        print("Global test dice at epoch {}: {}".format(epoch, test_dice_global))
        wandb.log({"test_dice_global": test_dice_global})

        scheduler.step()
        
        netname = str(type(net)).strip("'>").split(".")[1]
        state_dict_path = "datasets/saved_states/{}_runid_{:02}_epoch{:02}.pth".format(netname, config["runid"], epoch)
        if epoch % (config["checkpoint_interval"] - 1)==0 and epoch != 0:
            torch.save(net.state_dict(), state_dict_path)
        print() ## Blank line


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="train: Training and computing predictions.\ntest: \
        Only computing predictions using model specified as init_model_state in config.py.",
        default='train', choices=('test', 'train'))
    parser.add_argument("--focus", help="Which segmentation maps to use. Segment out livers or lesions.",
        default='train', choices=('liver', 'lesion'))
    parser.add_argument("--seed", type=int, help="Make code reproducible by setting a seed.", default=None)
    parser.add_argument("--runid", type=int, help="Id number for current run to distiguish saved states.")
    args = parser.parse_args()
    config["runid"] = args.runid
    if args.seed is not None:
        utils.ensure_reproducibility(args.seed)
    controller_2d(args.mode, args.focus)
