import torch
import torch.nn as nn
import utils
from config import config
from model import Metrics
import wandb
import numpy as np


def train_one_epoch(net, optimizer, critic, dataloader, 
                    device, epoch, epochlength, wandblog=True):

    """Go through training data once and adjust weighs of net.

        Arguments
        ---------
            net : torch neural net
            optimizer : torch optimizer
            critic : torch loss object
            dataloader : torch dataloader
            device : str
                Options: 'cpu', 'cuda'
            epoch : int
                Current epoch
            epochlength : int
                Total number of samples in one epoch including testing.
            [wandblog] : bool
                Default: True
                Update monitoring values to weights and biases.
            !![return_pred] : bool!! Not usable because of memory consumption
                Default: False
                If True, prediction is returned

        Returns
        -------
            infodict : dict
                Dict of info about process which is sent to weights and biases
                to monitor process there.
            [pred] : List[torch.Tensor]
                Returned only if return_pred is set to True.            
    """
    net.train()
    cuminfodict = {
        "epoch": [],
        "loss": [],
        "train_FNR": [],
        "train_FPR": [],
        "train_RVD": [],
        "train_dice": [],
        "train_dice_numerator": [],
        "train_dice_denominator": [],
        "train_iou": [],
        "train_conmat": []
    }

    for i, sample in enumerate(dataloader):
        optimizer.zero_grad()
        vol = sample['vol'].to(device, non_blocking=True)
        lab = sample['lab'].to(device, non_blocking=True)
        outputs = net(vol)
        pred = outputs[-1]
        losses = []
        for output_part in outputs:
            losses.append(critic(output_part, lab))

        loss = sum(losses[:-1])*config["alpha"] + losses[-1]
        loss.backward()
        optimizer.step()
        # onehot_lab = utils.one_hot(lab, nclasses=3)

        metrics = Metrics(torch.round(pred), lab)
        diceparts = metrics.get_dice_coefficient()
        infodict = {"epoch": epoch, # + i/epochlength,
                    "loss": loss.item(),
                    "train_FNR": metrics.get_FNR().detach().cpu().numpy(),
                    "train_FPR": metrics.get_FPR().detach().cpu().numpy(),
                    "train_RVD": metrics.get_RVD().detach().cpu().numpy(), 
                    "train_dice": diceparts[0].detach().cpu().numpy(),
                    "train_dice_numerator": diceparts[1].detach().cpu().numpy(),
                    "train_dice_denominator": diceparts[2].detach().cpu().numpy(),
                    "train_iou": metrics.get_jaccard_index().detach().cpu().numpy(),
                    "train_conmat": metrics.get_conmat().detach().cpu().numpy()}
        utils.update_cumu_dict(cuminfodict, infodict)
        if wandblog:
            wandb.log({"detailed_loss": loss.item()})
    ## Infologging
    for key in cuminfodict:
        cuminfodict[key] = np.mean(cuminfodict[key], axis=0)
    if wandblog:
        wandb.log(cuminfodict)

    return cuminfodict

