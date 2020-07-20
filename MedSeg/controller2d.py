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
from model import VNet2d, TverskyLoss, DiceLoss, MSEPixelCountLoss, DeepVNet2d
from train import train_one_epoch
from test import test_one_epoch
import wandb
from torchsummary import summary
import SimpleITK as sitk
import pandas as pd


## Use GPU if available
device = ('cuda' if torch.cuda.is_available() else 'cpu')


def controller_2d():
    """
    Controller function for training and testing VNet
    on Volumetric CT images of livers with and without lesions.
    """
    print("Starting...")
    
    ## Load data
    full_dataset = LiTSDataset2d(config["dst_2d_path"], focus=config["focus"])
    workers = config["num_workers"]
    
    ## Split data into train and test
    train_proportion = config["train_proportion"]
    len_train = int(len(full_dataset) * train_proportion)
    len_test = len(full_dataset) - len_train
    tr_set, te_set = torch.utils.data.random_split(full_dataset, (len_train, len_test))
    ## Init and load model if specified in config
    net = VNet2d(drop_rate=config["drop_rate"])
    # net = DeepVNet2d(drop_rate=config["drop_rate"])

    ## Load model if specified in config
    print(config["init_2d_model_state"])
    if config["init_2d_model_state"] is not None:
        print("Attemt fetching of state dict at: ", config["init_2d_model_state"])
        state_dict = torch.load(config["init_2d_model_state"])["model_state_dict"]
        net.load_state_dict(state_dict)
        print("Successfully loaded net")
    net.to(device)
    
    ## If only testing, run through net, get resulting metrics and store predictions.
    if config["mode"] == 'test':
        net.eval()

        tr_dataloader = DataLoader(tr_set, num_workers=workers, pin_memory=True)
        if not os.path.exists(config[f"dst2d_pred_{config['focus']}_path"]):
            os.mkdir(config[f"dst2d_pred_{config['focus']}_path"])
        train_info = test_one_epoch(net, tr_dataloader, device,
                                    1, 1, wandblog=False, dst_path=config[f"dst2d_pred_{config['focus']}_path"])
        tr_df = pd.DataFrame(train_info)
        tr_df.to_csv(os.path.join(config["dstpath"],
                                  "tr_metrics_{}_{}_run{:02}.csv".format(config["mode"], config["focus"], config["runid"])))

        te_dataloader = DataLoader(te_set, num_workers=workers, pin_memory=True)
        test_info = test_one_epoch(net, te_dataloader, device, 
                                   1, 1, wandblog=False, dst_path=None)
        te_df = pd.DataFrame(test_info)
        te_df.to_csv(os.path.join(config["dstpath"], 
                                  "te_metrics_{}_{}_run{:02}.csv".format(config["mode"], config["focus"], config["runid"])))
        exit()

    ## Monitor process with weights and biases
    wandb.init(config=config)

    ## Optimizer
    optimizer = torch.optim.Adam(params=net.parameters(), 
                                 **config["optim_opts"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, config["lr_decay_rate"], last_epoch=-1)
    
    ## Loss
    if config["label_type"] == 'segmentation':
        # critic = DiceLoss(**config["loss_opts"])
        critic = TverskyLoss(**config["loss_opts"])
    elif config["label_type"] == 'pixelcount':
        critic = MSEPixelCountLoss(**config["loss_opts"])
    
    for epoch in range(config["max_epochs"]):
        print(f"Epoch: {epoch}.")

        ## Make dataloaders
        print("Loading data ...")
        tr_dataloader = DataLoader(tr_set, batch_size=config["batch_size"], 
                                   shuffle=True, num_workers=workers, pin_memory=True)
        te_dataloader = DataLoader(te_set, batch_size=config["batch_size"], 
                                   shuffle=True, num_workers=workers, pin_memory=True)

        epochlength = len(tr_dataloader) + len(te_dataloader)

        print("Training ...")
        train_info = train_one_epoch(net, optimizer, critic, 
                                     tr_dataloader, device, epoch, epochlength)

        print("Testing ...")
        test_info = test_one_epoch(net, te_dataloader, device, 
                                     epoch, epochlength)

        scheduler.step()
        
        ## Checkpoint storage (with prediction exmple)
        netname = str(type(net)).strip("'>").split(".")[1]
        saved_states_folder = os.path.join("datasets/saved_states/runid_{:03}/".format(config["runid"]))
        if epoch % (config["checkpoint_interval"] - 1) == 0: # and epoch != 0:
            if not os.path.exists(saved_states_folder):
                os.mkdir(saved_states_folder)
            state_name = "{}_runid_{:02}_epoch{:02}.pth".format(netname, config["runid"], epoch)
            state_dict_path = os.path.join(saved_states_folder, state_name)
            
            torch.save({"model_state_dict": net.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "loss": np.mean(train_info["loss"]),  ## mean loss of epoch
                        "config": config,
                        }, state_dict_path)

            # torch.save(net.state_dict(), state_dict_path)
            ## Store example prediction
            imageidx = torch.randint(0, len(te_set), (1,))
            ex_loader = DataLoader(torch.utils.data.Subset(te_set, imageidx),
                                   num_workers=workers, pin_memory=True)
            test_one_epoch(net, ex_loader, device, epoch, epochlength,
                           wandblog=False, dst_format='npy', dst_path=config["dst2d_fig_path"])
        print()


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
    parser.add_argument("--label_type", choices=['segmentation', 'pixelcount', 'binary'], 
                        help="What information to apply from the labels. \
                            Eg. if binary, only image level information is \
                            provided to the network during training.",
                        default='segmentation')
    args = parser.parse_args()
    if args.seed is not None:
        utils.ensure_reproducibility(args.seed)
    
    ## Override prespecified config if arguments are given from commandline.
    config["runid"] = args.runid
    config["focus"] = args.focus
    config["mode"] = args.mode
    config["label_type"] = args.label_type

    controller_2d()
