"""
Controller with overview over processes.
"""

import utils
import os
import numpy as np
from pprint import pprint
from config import config
from dataloaders import LiTSDataset, Testset
from preprocessing import preprocess3d
from torch.utils.data import DataLoader
import torch
from model import VNet, TverskyLoss, CELoss, init, WeightedCrossEntropyLoss, DiceLoss
from train import train_one_epoch
from test import test_one_epoch
import wandb
from torchsummary import summary
import SimpleITK as sitk
import pandas as pd

## Use GPU if available
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def controller_3d(mode: str, focus: str):
    """
    Controller function for training and testing VNet
    on Volumetric CT images of livers with and without lesions.
    """
    print("Starting!!! :D")
    
    breakpoint()
    ## Load data sets
    tr_path = os.path.join(config["dstpath"], "train/")
    te_path = os.path.join(config["dstpath"], "test/")

    tr_set = LiTSDataset(tr_path, focus=focus)
    te_set = LiTSDataset(te_path, focus=focus)
    
    ## Initialize and load model if specified in config
    net = VNet(training=True, drop_rate=config["drop_rate"], binary_output=True)
    if config["init_model_state"] is not None:
        print("Attemt fetching of state dict at: ", config["init_model_state"])
        state_dict = torch.load(config["init_model_state"])
        net.load_state_dict(state_dict)
        print("Successfully loaded net")
    net.to(device)
    
    ## Test performance and store predictions if specified
    if mode == 'test':
        net.eval()

        tr_dataloader = DataLoader(tr_set)
        train_info = test_one_epoch(net, tr_dataloader, device, 
                                    1, 1, wandblog=False, dst_path=None)
        tr_df = pd.DataFrame(train_info)
        tr_df.to_csv(os.path.join(config["dstpath"], 
                                  "tr_metrics_{}_{}_run{:02}.csv".format(mode, focus, config["runid"])))

        te_dataloader = DataLoader(te_set)
        test_info = test_one_epoch(net, te_dataloader, device, 
                                   1, 1, wandblog=False)
        te_df = pd.DataFrame(test_info)
        te_df.to_csv(os.path.join(config["dstpath"], 
                                  "te_metrics_{}_{}_run{:02}.csv".format(mode, focus, config["runid"])))
        exit()

    ## Monitor process with weights and biases
    wandb.init(config=config)

    optimizer = torch.optim.Adam(params=net.parameters(), 
                                 **config["optim_opts"])

    critic = TverskyLoss(**config["loss_opts"])
    
    ## Training loop
    for epoch in range(config["max_epochs"]):
        print(f"Epoch: {epoch}.")

        ## Dataloaders
        print("Loading data ...")
        tr_dataloader = DataLoader(tr_set, shuffle=True)
        te_dataloader = DataLoader(te_set, shuffle=True)

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

        ## Save model state at checkpoints.
        netname = str(type(net)).strip("'>").split(".")[1]
        state_dict_path = "datasets/saved_states/{}_runid_{:02}_epoch{:02}.pth".format(netname, config["runid"], epoch)
        if epoch % (config["checkpoint_interval"] - 1)==0 and epoch != 0:
            torch.save(net.state_dict(), state_dict_path)


if __name__ == "__main__":
    ## Get arguments
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

    ## Set run id
    config["runid"] = args.runid
    if args.seed is not None:
        utils.ensure_reproducibility(args.seed)

    ## Run model
    controller_3d(args.mode, args.focus)
