"""
Configuration file. 

Modified verison of code copied from source: 
    https://github.com/assassint2017/MICCAI-LITS2017
"""

import os
# import torch
#######################################
################# RunID ###############
#######################################
config = dict(
    ###################################
    ############# Project #############
    ###################################
    project="MedSeg",
    runid = 0,
    focus = 'liver',  ## Must be 'liver' or 'lesion'
    mode = 'train',  ## Must be 'train' or 'test'
    ###################################
    ########## Main datapaths #########
    ###################################
    ## path to raw LiTS unaltered data,
    srcpath = 'datasets/original/',
    ## path to destination of 3d preprocessing,
    dstpath = 'datasets/preprocessed_quarter_size/',
    ## Path to destination of 2d preprocessing.
    dst_2d_path = 'datasets/preprocessed_2d/',
    ##########################,
    ###### Preprocessing #####,
    ##########################,
    ## Clipping values to consentrate on relevant range,
    ## https://www.google.com/search?q=typical+CT+HU+range+for+liver&oq=typical+CT+HU+range+for+liver&aqs=chrome..69i57.8870j0j4&sourceid=chrome&ie=UTF-8,
    ## (liver values usually revelves45-65 HU ?),
    upper = 200,
    lower = -200,
    ## volume depth (number of slices),
    size = 48,
    ## Cross sectional down scaling,
    down_scale = 1,
    ## Only use the liver and 20 slices of the liver as training samples,
    expand_slice = 20,
    ## Normalize the spacing of all data on the z-axis to 1mm,
    slice_thickness = 1,
    ##########################,
    ######### Model ##########,
    ##########################,
    train_proportion = 0.8,
    drop_rate = 0.3,
    ## Weight on losses 1, 2 and 3. Weight on loss 4 is 1.
    alpha = 0, # 0.33,
    num_workers = 2,
    ## Type of info to use from labels:
    ## Options: ['segmentation', 'pixelcount', 'binary']
    label_type = 'segmentation',
    ## DataLoader does not support any more yet,
    batch_size = 5,
    max_epochs = 200,
    ## Optimizer and loss,
    optim_opts = {'lr': 0.01},
    lr_decay_rate = 0.90,
    loss_opts = {},
    ## Store model with metadata at given intervals.
    checkpoint_interval = 3,
    init_2d_model_state = None,# "datasets/saved_states/runid_2009/VNet2d_runid_2009_epoch59.pth",
    init_model_state = None, #"datasets/saved_states/ResUnet_runid_12_epoch30.pth",
)

config.update(
    dict(
    ###########################################################
    ########### Specific paths to parts of data ###############
    ###########################################################
    train_volumes_path = os.path.join(config["srcpath"], 'train/ct/'),
    train_labels_path = os.path.join(config["srcpath"], 'train/seg/'),
    test_volumes_path = os.path.join(config["srcpath"], 'test/ct/'),
    test_labels_path = os.path.join(config["srcpath"], 'test/seg/'),
    ## For 3d
    dst_train_volumes_path = os.path.join(config["dstpath"], 'train/volumes/'),
    dst_test_volumes_path = os.path.join(config["dstpath"], 'test/volumes/'),
    dst_train_labels_liver_path = os.path.join(config["dstpath"], 'train/labels_liver/'),
    dst_train_labels_lesion_path = os.path.join(config["dstpath"], 'train/labels_lesion/'),
    dst_test_labels_liver_path = os.path.join(config["dstpath"], 'test/labels_liver/'),
    dst_test_labels_lesion_path = os.path.join(config["dstpath"], 'test/labels_lesion/'),
    dst_train_pred_liver_path = os.path.join(config["dstpath"], 'train/pred_liver/'),
    dst_train_pred_lesion_path = os.path.join(config["dstpath"], 'train/pred_lesion/'),
    dst_test_pred_liver_path = os.path.join(config["dstpath"], 'test/pred_liver/'),
    dst_test_pred_lesion_path = os.path.join(config["dstpath"], 'test/pred_lesion/'),
    ## For 2d
    dst2d_slices_path = os.path.join(config["dst_2d_path"], 'slices/'),
    dst2d_labels_liver_path = os.path.join(config["dst_2d_path"], 'labels_liver/'),
    dst2d_labels_lesion_path = os.path.join(config["dst_2d_path"], 'labels_lesion/'),
    dst2d_pred_liver_path = os.path.join(config["dst_2d_path"], 'pred_liver/'),
    dst2d_pred_lesion_path = os.path.join(config["dst_2d_path"], 'pred_lesion/'),
    dst2d_fig_path = os.path.join(config["dst_2d_path"], "figures/")
    )
)
