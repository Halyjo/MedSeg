import torch.nn as nn
import torch


class VGG(nn.Module):
    """
    Fully Convolutional Network implementation. 

    Arguments:
        trainable_upsampling: (bool)
            Default: True
            If false, bilinear upsampling is used. If true,
            a fractional strided convolution is used.

    Initialization:
        By default, Conv2d layers in pytorch are initialized with
        He(Kaiming) initialization which is ideal in this case
        since we have applied relu-activations for all layers except
        the last where we use softmax. Note that the softmax is implicit
        in the loss function (CrossEntropyLoss).

    """
    def __init__(self, trainable_upsampling=True):
        super(VGG, self).__init__()
        self.block1 = nn.Sequential(
            *([nn.Conv2d(1, 64, 3, padding=1),  # padding 1 to end up with the same size
              nn.ReLU(),
              nn.Conv2d(64, 64, 3, padding=1),
              nn.ReLU(),
              nn.MaxPool2d(2)
            ]))
        self.block2 = nn.Sequential(
            *([
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]))
        self.block3 = nn.Sequential(
            *([
                nn.Conv2d(128, 256, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]))
        self.block4 = nn.Sequential(
            *([
                nn.Conv2d(256, 512, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(512, 512, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(512, 512, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]))
        self.block5 = nn.Sequential(
            *([
                nn.Conv2d(512, 512, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(512, 512, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(512, 12, 3, padding=1),  ## 512, 512
                nn.ReLU(),
############################################################
############## Remove these in CAM Mod #####################
############################################################
                # nn.MaxPool2d(2),
            ]))
        # self.block6 = nn.Sequential(
        #     *([
        #         ## Equivalent to fully connected layers
        #         nn.Conv2d(512, 4096, 1, padding=0),r
        #         nn.ReLU(),
        #         nn.Conv2d(4096, 4096, 1, padding=0),
        #         nn.ReLU(),
        #         nn.Conv2d(4096, 12, 1, padding=0),
        #         nn.ReLU(),
        #     ]))
############################################################
############################################################
############################################################
        if trainable_upsampling:
            # self.upsampling = nn.ConvTranspose2d(12, 12, 
            #                                      kernel_size=60,
            #                                      stride=30)
            self.upsampling = nn.ConvTranspose2d(12, 1, 
                                                 kernel_size=16,
                                                 stride=16)

        else:
            self.upsampling = nn.UpsamplingBilinear2d(size=(360, 480))
        ## Is done implicitly in CrossEntropyLoss
        # self.softmax = nn.Softmax2d()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        # x = self.block6(x)
        x = self.upsampling(x)
        # x = self.sigmoid(x) ## Done implicitly in loss
        return x
        