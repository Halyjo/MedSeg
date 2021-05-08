import utils
from model import Metrics
import wandb
import torch
from config import config
import numpy as np


def test_one_epoch(net, dataloader, device, epoch, epochlength, wandblog=True, dst_path=None, dst_format=None):
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
        "test_fnr": [],
        "test_fpr": [],
        "test_voe": [],
        "test_rvd": [],
        "test_dice": [],
        "test_iou": [],
        "test_dice_numerator": [],
        "test_dice_denominator": [],
        "test_classification_accuracy": [],
        # "test_conmat": [],
    }

    for i, sample in enumerate(dataloader):
        vol = sample['vol'].to(device, non_blocking=True)
        lab_seg = sample['lab'].to(device, non_blocking=True)
        pred_soft = net(vol)
        pred = torch.round(pred_soft)
        # onehot_lab = utils.one_hot(lab_seg, nclasses=3)

        ## Log metrics
        metrics = Metrics(pred, lab_seg, mode='test')
        infodict = metrics.get_metric_dict()
        for key in infodict:
            infodict[key] = infodict[key].detach().cpu().numpy()
        infodict.update({"epoch": epoch})

        utils.update_cumu_dict(cuminfodict, infodict)

        ## Classification accuracy
        if (config["label_type"] == "binary") and wandblog:
            pred = pred.view(pred.size(0), -1).sum(-1) > 0
            lab = lab_seg.view(lab_seg.size(0), -1).sum(-1) > 0
            acc = (pred == lab).float().mean().detach().cpu().numpy()
            wandb.log({"Test Positive Predictions": pred.sum(), "Test Positive Labels": lab.sum(), "Test Accuracy": float(acc)})

        ## Store prediction
        if dst_path is not None:
            # to_store = torch.cat([vol, lab, pred])
            utils.store(pred_soft, dst_path, sample['store_idx'],
                        format=dst_format, epoch=epoch, focus=config["focus"])
            if wandblog:
                for i in range(pred_soft.size(0)):
                    wandb.log({"Example Prediction": [wandb.Image(pred_soft[i, 0, ...].detach().cpu().numpy()*255,
                                                                  caption="Prediction "+str(config["runid"])+str(epoch)+str(sample["store_idx"][i]))]})
                    wandb.log({"Example Image": [wandb.Image(vol[i, 0, ...].cpu().numpy()*255, 
                                                             caption="Image "+str(config["runid"])+str(epoch)+str(sample["store_idx"][i]))]})
                    wandb.log({"Example Label": [wandb.Image(lab_seg[i, 0, ...].cpu().numpy()*255, 
                                                             caption="Label "+str(config["runid"])+str(epoch)+str(sample["store_idx"][i]))]})
    ## Infologging
    for key in cuminfodict:
        cuminfodict[key] = np.mean(cuminfodict[key], axis=0)
    if wandblog:
        wandb.log(cuminfodict)

    return cuminfodict

        