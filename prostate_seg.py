import torch.nn as nn
import torch
import numpy as np
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
import config

'''
t2-w image , dicom, (384,384)
mask image , png, (384,384); label 1 and 2 in the mask indicates the pheripheral and transitional zones, respectively. Boundaries were enclosed by 255
'''

'''
building block for the encoder and decoder
stores two convolution, one batch normalization, and one leaky relu activation
'''
class Block(nn.Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.conv1 = nn.Conv2d(inChannels,outChannels,3,1,1)
        self.conv2 = nn.Conv2d(outChannels, outChannels,3,1,1)
        self.batchnorm = nn.BatchNorm2d(outChannels)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.conv1(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchnorm(x)
        x = self.relu(x)

        return x

'''
384 ->  192 -> 96 -> 48

1 -> 3 -> 16 -> 32

'''
class Encoder(nn.Module):
    def __init__(self,channels= [1,3,16,32]):
        super().__init__()
        self.encBlocks = nn.ModuleList(
            [Block(channels[i],channels[i+1])
                   for i in range(len(channels)-1)]
        )
        self.pool = nn.MaxPool2d(2)
    def forward(self,x):
        blocks = []
        for block in self.encBlocks:
            x = block(x)
            blocks.append(x)
            x = self.pool(x)
        return blocks
        # for i in range(len(self.encBlocks)):
        #     block = self.encBlocks[i]
        #     x = block(x)
        #     blocks.append(x)
        #     if i == 0:
        #         x = self.pool2(x)
        #     else:
        #         x = self.pool1(x)
        # return blocks

'''
spatial dimension 48 -> 96 -> 192
channels: 32 -> 16 -> 3
'''
class Decoder(nn.Module):
    def __init__(self,channels = [32,16,3]):
        super().__init__()
        self.channels = channels
        # self.upsample1 = nn.ConvTranspose2d()
        # self.upsample2 = nn.ConvTranspose2d()
        self.upconvs = nn.ModuleList(
            [nn.ConvTranspose2d(channels[i],channels[i+1],2,2)
             for i in range(len(channels) - 1)]
        )
        self.dec_blocks = nn.ModuleList(
            [Block(channels[i],channels[i+1])
                for i in range(len(channels) - 1)]
        )
    def forward(self,x,encFeatures):
        for i in range(len(self.channels) -1):
            x = self.upconvs[i](x)
            #if i == len(self.channels) - 2:
            #    block = self._block_upsample_(self.channels[i],self.channels[i+1],3,2)
            #else:
            #    block = self._block_upsample_(self.channels[i], self.channels[i + 1],2, 2)
            #x = block(x)
            encFeat = self.crop(encFeatures[i],x)
            x = torch.cat([x,encFeat],dim = 1)
            x = self.dec_blocks[i](x)
    #def _block_upsample_(self,inChannels,outChannels,kernel,stride,batchnorm=True):
    #    return nn.ConvTranspose2d(in_channels = inChannels,out_channels = outChannels,kernel_size=kernel,stride = stride),

    def crop(self,encFeatures,x):
        (_,_,H,W) = x.shape
        encFeatures = CenterCrop([H,W])(encFeatures)

        return encFeatures



class U_net(nn.Module):
    """
    implements u_net architecture
    """
    def __init__(self,encChannels=[1,3,16,32],
                 decChannels = [32,16,3],
                 nbClassses = 3,retainDim = True,
                 outSize= (config.INPUT_IMAGE_HEIGHT,config.INPUT_IMAGE_WIDTH)):
        super().__init__()
        self.encoder = Encoder(encChannels)
        self.decoder = Decoder(decChannels)

        self.head = nn.Conv2d(decChannels[-1],nbClassses,1)
        self.retainDim = retainDim
        self.OutSize = outSize

    def forward(self,x):
        encFeatures = self.encoder(x)
        decFeatures = self.decoder(encFeatures[::-1][0],
                                   encFeatures[::-1][1:])
        map = self.head(decFeatures)

        if self.retainDim:
            map = F.interpolate(map,self.outSize)

        return map











