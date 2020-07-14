"""
Modified version of a great net that can be found:
https://github.com/assassint2017/MICCAI-LITS2017/
"""
"""

网络定义脚本
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class VNet(nn.Module):
    """
    Modified version of a net that can be found:
    https://github.com/assassint2017/MICCAI-LITS2017/.
    # parameters: 9498260
    """
    def __init__(self, training, drop_rate=0.5, binary_output=True):
        super().__init__()
        self.training = training
        self.drop_rate = drop_rate
        self.binary_output = binary_output

        self.encoder_stage1 = nn.Sequential(
            nn.Conv3d(1, 16, 3, 1, padding=1),
            nn.BatchNorm3d(16),
            nn.PReLU(16),

            nn.Conv3d(16, 16, 3, 1, padding=1),
            nn.BatchNorm3d(16),
            nn.PReLU(16),
        )

        self.encoder_stage2 = nn.Sequential(
            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.BatchNorm3d(32),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.BatchNorm3d(32),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.BatchNorm3d(32),
            nn.PReLU(32),
        )

        self.encoder_stage3 = nn.Sequential(
            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.BatchNorm3d(64),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=2, dilation=2),
            nn.BatchNorm3d(64),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=4, dilation=4),
            nn.BatchNorm3d(64),
            nn.PReLU(64),
        )

        self.encoder_stage4 = nn.Sequential(
            nn.Conv3d(128, 128, 3, 1, padding=3, dilation=3),
            nn.BatchNorm3d(128),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=4, dilation=4),
            nn.BatchNorm3d(128),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=5, dilation=5),
            nn.BatchNorm3d(128),
            nn.PReLU(128),
        )

        self.decoder_stage1 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 1, padding=1),
            nn.BatchNorm3d(256),
            nn.PReLU(256),

            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.BatchNorm3d(256),
            nn.PReLU(256),

            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.BatchNorm3d(256),
            nn.PReLU(256),
        )

        self.decoder_stage2 = nn.Sequential(
            nn.Conv3d(128 + 64, 128, 3, 1, padding=1),
            nn.BatchNorm3d(128),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.BatchNorm3d(128),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.BatchNorm3d(128),
            nn.PReLU(128),
        )

        self.decoder_stage3 = nn.Sequential(
            nn.Conv3d(64 + 32, 64, 3, 1, padding=1),
            nn.BatchNorm3d(64),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.BatchNorm3d(64),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.BatchNorm3d(64),
            nn.PReLU(64),
        )

        self.decoder_stage4 = nn.Sequential(
            nn.Conv3d(32 + 16, 32, 3, 1, padding=1),
            nn.BatchNorm3d(32),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.BatchNorm3d(32),
            nn.PReLU(32),
        )

        self.down_conv1 = nn.Sequential(
            nn.Conv3d(16, 32, 2, 2),
            nn.BatchNorm3d(32),
            nn.PReLU(32)
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv3d(32, 64, 2, 2),
            nn.BatchNorm3d(64),
            nn.PReLU(64)
        )

        self.down_conv3 = nn.Sequential(
            nn.Conv3d(64, 128, 2, 2),
            nn.BatchNorm3d(128),
            nn.PReLU(128)
        )

        self.down_conv4 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 1, padding=1),
            nn.BatchNorm3d(256),
            nn.PReLU(256)
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 2, 2),
            nn.BatchNorm3d(128),
            nn.PReLU(128)
        )

        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 2, 2),
            nn.BatchNorm3d(64),
            nn.PReLU(64)
        )

        self.up_conv4 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 2, 2),
            nn.BatchNorm3d(32),
            nn.PReLU(32)
        )

        # Kartleggingen i stor skala (256 * 256)
        # Følgende Skalaer avtar sekvensielt
        if self.binary_output:
            self.map4 = nn.Sequential(
                nn.Conv3d(32, 1, 1, 1),
                nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=True),
                nn.Sigmoid()
            )

            # 128*128 尺度下的映射
            self.map3 = nn.Sequential(
                nn.Conv3d(64, 1, 1, 1),
                nn.Upsample(scale_factor=(2, 4, 4), mode='trilinear', align_corners=True),
                nn.Sigmoid()
            )

            # 64*64 尺度下的映射
            self.map2 = nn.Sequential(
                nn.Conv3d(128, 1, 1, 1),
                nn.Upsample(scale_factor=(4, 8, 8), mode='trilinear', align_corners=True),
                nn.Sigmoid()
            )

            # 32*32 尺度下的映射
            self.map1 = nn.Sequential(
                nn.Conv3d(256, 1, 1, 1),
                nn.Upsample(scale_factor=(8, 16, 16), mode='trilinear', align_corners=True),
                nn.Sigmoid()
            )
        else:
            self.map4 = nn.Sequential(
                nn.Conv3d(32, 3, 1, 1),
                nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=True),
                # nn.Softmax(dim=1)
            )

            # 128*128 尺度下的映射
            self.map3 = nn.Sequential(
                nn.Conv3d(64, 3, 1, 1),
                nn.Upsample(scale_factor=(2, 4, 4), mode='trilinear', align_corners=True),
                # nn.Softmax(dim=1)
            )

            # 64*64 尺度下的映射
            self.map2 = nn.Sequential(
                nn.Conv3d(128, 3, 1, 1),
                nn.Upsample(scale_factor=(4, 8, 8), mode='trilinear', align_corners=True),
                # nn.Softmax(dim=1) # Included in CrossEntropy
            )

            # 32*32 尺度下的映射
            self.map1 = nn.Sequential(
                nn.Conv3d(256, 3, 1, 1),
                nn.Upsample(scale_factor=(8, 16, 16), mode='trilinear', align_corners=True),
                # nn.Softmax(dim=1)
            )

    def forward(self, inputs):
        long_range1 = self.encoder_stage1(inputs) + inputs
        short_range1 = self.down_conv1(long_range1)
        long_range2 = self.encoder_stage2(short_range1) + short_range1
        long_range2 = F.dropout(long_range2, self.drop_rate, self.training)
        short_range2 = self.down_conv2(long_range2)
        long_range3 = self.encoder_stage3(short_range2) + short_range2
        long_range3 = F.dropout(long_range3, self.drop_rate, self.training)
        short_range3 = self.down_conv3(long_range3)
        long_range4 = self.encoder_stage4(short_range3) + short_range3
        long_range4 = F.dropout(long_range4, self.drop_rate, self.training)
        short_range4 = self.down_conv4(long_range4)

        outputs = self.decoder_stage1(long_range4) + short_range4
        outputs = F.dropout(outputs, self.drop_rate, self.training)
        output1 = self.map1(outputs)
        short_range6 = self.up_conv2(outputs)
        outputs = self.decoder_stage2(torch.cat([short_range6, long_range3], dim=1)) + short_range6
        outputs = F.dropout(outputs, 0.3, self.training)
        output2 = self.map2(outputs)
        short_range7 = self.up_conv3(outputs)
        outputs = self.decoder_stage3(torch.cat([short_range7, long_range2], dim=1)) + short_range7
        outputs = F.dropout(outputs, 0.3, self.training)
        output3 = self.map3(outputs)
        short_range8 = self.up_conv4(outputs)
        outputs = self.decoder_stage4(torch.cat([short_range8, long_range1], dim=1)) + short_range8
        output4 = self.map4(outputs)

        if self.training is True:
            return output1, output2, output3, output4
        else:
            return output4

    def print_shapes(self, inshape, **kwargs):
        """
        # prints all expected shapes through network without computing the result. 
        Only supports conv3d and convtransposed3d.
            
            Arguments
            ---------
                inshape : tuple
                    Format: (N, Cin, D, H, W)
        """
        for key, value in self._modules.items():
            # print(key)
            for m in value.modules():
                if isinstance(m, nn.modules.conv.Conv3d):
                    estr = "net.get_conv_shape(" + str(inshape) + ", " + m.extra_repr() + ")"
                    inshape = eval(estr)
                    print("\t", inshape)
                elif isinstance(m, nn.modules.conv.ConvTranspose3d):
                    estr = "net.get_conv_shape(" + str(inshape) + ", " + m.extra_repr() + ", T=True)"
                    print("\t", eval(estr))

    @staticmethod
    def get_conv_shape(inputshape, Cin, Cout, 
                        kernel_size=3, stride=1, 
                        padding=0, dilation=1, T=False,
                        output_padding=0):
        """
        Finds outshape of 3d convolution or transposed convolution (T=True) 
        given inshape on format: (N, Cin, D, H, W).
        """
        if isinstance(output_padding, int):
            output_padding = (output_padding, output_padding, output_padding)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)
        
        outshape = [inputshape[0], Cout]
        if T:
            for i in range(3):
                s = (inputshape[i+2] - 1)*stride[i] - 2*padding[i] + dilation[i]*(kernel_size[i]-1) + output_padding[i] + 1
                outshape.append(s)
        else:
            for i in range(3):
                s = (inputshape[i+2] + 2*padding[i] - dilation[i]*(kernel_size[i]-1) - 1)//stride[i] + 1
                outshape.append(s)
        return outshape


class PyramidBlock(nn.Module):
    def __init__(self, in_channels, intermed_channels, **args):
        super().__init__()
        self.in_channels = in_channels
        self.intermed_channels = intermed_channels

        self.conv1 = nn.Sequential(
            nn.Conv3d(self.in_channels, self.intermed_channels, 1, 1),
            nn.BatchNorm3d(in_channels),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(self.in_channels, self.intermed_channels, 3, 4, padding=1),
            nn.BatchNorm3d(in_channels),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(self.in_channels, self.intermed_channels, 3, 8, padding=1),
            nn.BatchNorm3d(in_channels),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv3d(self.in_channels, self.intermed_channels, 3, 16, padding=1),
            nn.BatchNorm3d(in_channels),
            nn.ReLU()
        )

        self.upsample2 = nn.Upsample(scale_factor=(1, 4, 4), mode='trilinear', align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=(1, 8, 8), mode='trilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=(1, 16, 16), mode='trilinear', align_corners=True)

        self.convout = nn.Sequential(
            nn.Conv3d(self.in_channels + 4*self.intermed_channels, 1, 3, 1, padding=1),
            nn.BatchNorm3d(self.in_channels),
            nn.ReLU()
        )

    def forward(self, inputs):
        out1 = self.conv1(inputs)
        out2 = self.upsample2(self.conv2(inputs))
        out3 = self.upsample3(self.conv3(inputs))
        out4 = self.upsample4(self.conv3(inputs))
        out = self.convout(torch.cat([inputs, out1, out2, out3, out4]))
        return out


def init(module):
    if isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d):
        nn.init.kaiming_normal_(module.weight.data, 0.25)
        nn.init.constant_(module.bias.data, 0)


if __name__ == "__main__":
    inshape = (1, 1, 170, 256, 256)
    net = ResUNet(training=True)
    net.print_shapes(inshape)