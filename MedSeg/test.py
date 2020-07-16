import utils
from model import Metrics
import wandb
import torch
import config


def test_one_epoch(net, dataloader, device, epoch, epochlength, wandblog=True, dst_path=None, dst_format=None): # , return_pred=False
    """Go through testing data once, measure performance
    and send result to weights and biases

        Arguments
        ---------
            net : torch neural net
            dataloader : torch dataloader
            device : str
                Options: 'cpu', 'cuda'
            epoch : int
                Current epoch
            epochlength : int
                Total number of samples in one epoch including testing.
            [wandblog] : bool
                Default: True
                Send monitoring data to weights and biases.
            !![return_pred] : bool!! Not usable because of memory consumption
                Default: False
                If True, predictions are returned.
            [dst_path] : str path
                Default: None, i.e. Do not store preductions.
                Path to destination folder to store predictions at.
            [dst_format] : str
                Format of storage. Options are decided in utils.store().


        Returns
        -------
            infodict : dict
                Dict of info about process which is sent to weights and biases
                to monitor process there.
            
    """
 
    net.eval()
    cuminfodict = {
        "epoch": [],
        "test_FNR": [],
        "test_FPR": [],
        "test_dice": [],
        "test_iou": [],
        "test_dice_numerator": [],
        "test_dice_denominator": [],
        "test_conmat": [],
    }

    for i, sample in enumerate(dataloader):
        vol = sample['vol'].to(device, non_blocking=True)
        lab = sample['lab'].to(device, non_blocking=True)
        pred = torch.round(net(vol))
        # onehot_lab = utils.one_hot(lab, nclasses=3)

        metrics = Metrics(pred, lab)
        
        diceparts = metrics.get_dice_coefficient()
        infodict = {"epoch": epoch + i/epochlength,
                    "test_FNR": metrics.get_FNR().detach().cpu().numpy(),
                    "test_FPR": metrics.get_FPR().detach().cpu().numpy(),
                    "test_dice": diceparts[0].detach().cpu().numpy(),
                    "test_dice_numerator": diceparts[1].detach().cpu().numpy(),
                    "test_dice_denominator": diceparts[2].detach().cpu().numpy(),
                    "test_iou": metrics.get_jaccard_index().detach().cpu().numpy(),
                    "test_conmat": metrics.get_conmat().detach().cpu().numpy()}
        utils.update_cumu_dict(cuminfodict, infodict)
        if wandblog:
            wandb.log(infodict)

        ## Store predictions
        if dst_path is not None:
            utils.store(pred, dst_path, sample['store_idx'],
                        format=dst_format, epoch=epoch, focus=config["focus"])

    return cuminfodict

        