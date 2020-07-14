"""
2d Version of Vnet but one step deeper.
"""
import torch
import torch.nn as nn


class DeepVNet2d(nn.Module):
    """
    Modified version of a net that can be found:
    https://github.com/assassint2017/MICCAI-LITS2017/.
    """
    def __init__(self, drop_rate=0.5):
        super().__init__()
        self.drop_rate = drop_rate

        self.encoder_stage1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, padding=1),
            nn.BatchNorm2d(16),
            nn.PReLU(16),

            nn.Conv2d(16, 16, 3, 1, padding=1),
            nn.BatchNorm2d(16),
            nn.PReLU(16),
        )

        self.encoder_stage2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(32),

            nn.Conv2d(32, 32, 3, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(32),

            nn.Conv2d(32, 32, 3, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(32),
        )

        self.encoder_stage3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(64),

            nn.Conv2d(64, 64, 3, 1, padding=2, dilation=2),
            nn.BatchNorm2d(64),
            nn.PReLU(64),

            nn.Conv2d(64, 64, 3, 1, padding=4, dilation=4),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )

        self.encoder_stage4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, padding=3, dilation=3),
            nn.BatchNorm2d(128),
            nn.PReLU(128),

            nn.Conv2d(128, 128, 3, 1, padding=4, dilation=4),
            nn.BatchNorm2d(128),
            nn.PReLU(128),

            nn.Conv2d(128, 128, 3, 1, padding=5, dilation=5),
            nn.BatchNorm2d(128),
            nn.PReLU(128),
        )

        self.encoder_stage5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, padding=3, dilation=3),
            nn.BatchNorm2d(256),
            nn.PReLU(256),

            nn.Conv2d(256, 256, 3, 1, padding=4, dilation=4),
            nn.BatchNorm2d(256),
            nn.PReLU(256),

            nn.Conv2d(256, 256, 3, 1, padding=5, dilation=5),
            nn.BatchNorm2d(256),
            nn.PReLU(256),
        )

        self.decoder_stage0 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, padding=1),
            nn.BatchNorm2d(512),
            nn.PReLU(512),

            nn.Conv2d(512, 512, 3, 1, padding=1),
            nn.BatchNorm2d(512),
            nn.PReLU(512),

            nn.Conv2d(512, 512, 3, 1, padding=1),
            nn.BatchNorm2d(512),
            nn.PReLU(512),
        )

        self.decoder_stage1 = nn.Sequential(
            nn.Conv2d(256+128, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(256),

            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(256),

            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(256),
        )

        self.decoder_stage2 = nn.Sequential(
            nn.Conv2d(128 + 64, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(128),

            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(128),

            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(128),
        )

        self.decoder_stage3 = nn.Sequential(
            nn.Conv2d(64 + 32, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(64),

            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(64),

            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )

        self.decoder_stage4 = nn.Sequential(
            nn.Conv2d(32 + 16, 32, 3, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(32),

            nn.Conv2d(32, 32, 3, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(32),
        )

        self.down_conv1 = nn.Sequential(
            nn.Conv2d(16, 32, 2, 2),
            nn.BatchNorm2d(32),
            nn.PReLU(32)
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 2, 2),
            nn.BatchNorm2d(64),
            nn.PReLU(64)
        )

        self.down_conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 2, 2),
            nn.BatchNorm2d(128),
            nn.PReLU(128)
        )

        self.down_conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 2, 2),
            nn.BatchNorm2d(256),
            nn.PReLU(256)
        )

        self.down_conv5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, padding=1),
            nn.BatchNorm2d(512),
            nn.PReLU(512)
        )

        self.up_conv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, 2),
            nn.BatchNorm2d(256),
            nn.PReLU(256)
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, 2),
            nn.BatchNorm2d(128),
            nn.PReLU(128)
        )

        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, 2),
            nn.BatchNorm2d(64),
            nn.PReLU(64)
        )


        self.dp1 = nn.Dropout2d(self.drop_rate)
        self.dp2 = nn.Dropout2d(self.drop_rate)
        self.dp3 = nn.Dropout2d(self.drop_rate)
        self.dp4 = nn.Dropout2d(self.drop_rate)
        self.dp5 = nn.Dropout2d(self.drop_rate)
        self.dp6 = nn.Dropout2d(self.drop_rate)
        self.dp7 = nn.Dropout2d(self.drop_rate)
        self.dp8 = nn.Dropout2d(self.drop_rate)


        self.map3 = nn.Sequential(
            nn.Conv2d(64, 1, 1, 1),
            # nn.ConvTranspose2d(1, 1, 2, 2, padding=0),
            nn.Upsample(scale_factor=(2, 2), mode='bilinear'),
            nn.Sigmoid()
        )

        self.map2 = nn.Sequential(
            nn.Conv2d(128, 1, 1, 1),
            # nn.ConvTranspose2d(1, 1, 4, 4, padding=0),
            nn.Upsample(scale_factor=(4, 4), mode='bilinear'),
            nn.Sigmoid()
        )

        self.map1 = nn.Sequential(
            nn.Conv2d(256, 1, 1, 1),
            # nn.ConvTranspose2d(1, 1, 8, 8, padding=0),
            nn.Upsample(scale_factor=(8, 8), mode='bilinear'),
            nn.Sigmoid()
        )

        self.map0 = nn.Sequential(
            nn.Conv2d(512, 1, 1, 1),
            # nn.ConvTranspose2d(1, 1, 16, 16, padding=0),
            nn.Upsample(scale_factor=(16, 16), mode='bilinear'),
            nn.Sigmoid()
        )


    def forward(self, inputs):
        ## Encoding
        long_range1 = self.encoder_stage1(inputs) + inputs
        short_range1 = self.down_conv1(long_range1)

        long_range2 = self.encoder_stage2(short_range1) + short_range1
        long_range2 = self.dp1(long_range2)
        short_range2 = self.down_conv2(long_range2)

        long_range3 = self.encoder_stage3(short_range2) + short_range2
        long_range3 = self.dp2(long_range3)
        short_range3 = self.down_conv3(long_range3)

        long_range4 = self.encoder_stage4(short_range3) + short_range3
        long_range4 = self.dp3(long_range4)
        short_range4 = self.down_conv4(long_range4)

        ####### Bonus depth encoding block #######
        long_range5 = self.encoder_stage5(short_range4) + short_range4
        long_range5 = self.dp4(long_range5)
        short_range5 = self.down_conv5(long_range5)
        #######

        ## Decoding

        ####### Bonus depth decoding block #######
        outputs = self.decoder_stage0(long_range5) + short_range5
        outputs = self.dp5(outputs)
        output0 = self.map0(outputs)
        short_range6 = self.up_conv1(outputs)
        #######
        
        outputs = self.decoder_stage1(torch.cat([short_range6, long_range4], dim=1)) + short_range6
        outputs = self.dp6(outputs)
        output1 = self.map1(outputs)
        short_range7 = self.up_conv2(outputs)

        outputs = self.decoder_stage2(torch.cat([short_range7, long_range3], dim=1)) + short_range7
        outputs = self.dp7(outputs)
        output2 = self.map2(outputs)
        short_range8 = self.up_conv3(outputs)
        
        outputs = self.decoder_stage3(torch.cat([short_range8, long_range2], dim=1)) + short_range8
        outputs = self.dp8(outputs)
        output3 = self.map3(outputs)

        if self.training is True:
            return output0, output1, output2, output3
        else:
            return output3

    def print_shapes(self, inshape, **kwargs):
        """
        # prints all expected shapes through network without computing the result. 
        Only supports conv2d and convtransposed2d.
            
            Arguments
            ---------
                inshape : tuple
                    Format: (N, Cin, D, H, W)
        """
        for key, value in self._modules.items():
            # print(key)
            for m in value.modules():
                if isinstance(m, nn.modules.conv.Conv2d):
                    estr = "net.get_conv_shape(" + str(inshape) + ", " + m.extra_repr() + ")"
                    inshape = eval(estr)
                    print("\t", inshape)
                elif isinstance(m, nn.modules.conv.ConvTranspose2d):
                    estr = "net.get_conv_shape(" + str(inshape) + ", " + m.extra_repr() + ", T=True)"
                    print("\t", eval(estr))

    @staticmethod
    def get_conv_shape(inputshape, Cin, Cout, 
                        kernel_size=3, stride=1, 
                        padding=0, dilation=1, T=False,
                        output_padding=0):
        """
        Finds outshape of 2d convolution or transposed convolution (T=True) 
        given inshape on format: (N, Cin, D, H, W).
        """
        if isinstance(output_padding, int):
            output_padding = (output_padding, output_padding)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        
        outshape = [inputshape[0], Cout]
        if T:
            for i in range(2):
                s = (inputshape[i+2] - 1)*stride[i] - 2*padding[i] + dilation[i]*(kernel_size[i]-1) + output_padding[i] + 1
                outshape.append(s)
        else:
            for i in range(2):
                s = (inputshape[i+2] + 2*padding[i] - dilation[i]*(kernel_size[i]-1) - 1)//stride[i] + 1
                outshape.append(s)
        return outshape


if __name__ == "__main__":
    from torchsummary import summary
    net = VNet2d()
    net.print_shapes((1, 1, 512, 512))
    summary(net, (1, 512, 512))