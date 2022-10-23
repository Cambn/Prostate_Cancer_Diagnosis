import torch.nn as nn
import torch
import numpy as np
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
import cv2
import config

'''
t2-w image , dicom, (384,384)
mask image , png, (384,384); label 1 and 2 in the mask indicates the pheripheral and transitional zones, respectively. Boundaries were enclosed by 255.
'''

'''
building block for the encoder and decoder
stores two convolution, one batch normalization, and one leaky relu activation
'''
class Block(nn.Module):
    def __init__(self, inChannels, outChannels,dropout = False):
        super().__init__()
        self.conv1 = nn.Conv2d(inChannels, outChannels,3,1,1)
        self.conv2 = nn.Conv2d(outChannels, outChannels,3,1,1)
        self.batchnorm = nn.BatchNorm2d(outChannels)
        self.relu = nn.LeakyReLU()
        self.d = dropout
        self.dropout = nn.Dropout(p=0.25)

    def forward(self,x):
        x = self.conv1(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        if self.d:
            x = self.dropout(x)

        return x

'''
input shape: 256 -> 128 -> 64 -> 32

# num of channel: 1 -> 32 -> 64 -> 128 -> 256
'''
class Encoder(nn.Module):
    def __init__(self,channels= [1,32,64,128,256]):
        super().__init__()
        self.encBlocks = nn.ModuleList(
            [Block(channels[i],channels[i+1],True if i >= 2 else False)
                   for i in range(len(channels)-1)]
        )
        self.pool = nn.MaxPool2d(2)
    def forward(self,x):
        blocks = []
        for block in self.encBlocks:
            x = block(x)
            blocks.append(x)
            x = self.pool(x)
            #print(x.shape)
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
    def __init__(self,channels = [256,128,64,32]):
        super().__init__()
        self.channels = channels
        
        self.upconvs = nn.ModuleList(
            [nn.ConvTranspose2d(channels[i],channels[i+1],2,2)
             for i in range(len(channels) - 1)]
        )
        self.dec_blocks = nn.ModuleList(
            [Block(channels[i],channels[i+1],True if i <= 1 else False)
                for i in range(len(channels) - 1)]
        )
    def forward(self,x,encFeatures):
        for i in range(len(self.channels) -1):
            #print(i)
            #print(r'x before upconvs:{}'.format(x.shape))
            x = self.upconvs[i](x)
            #print(r'x after upconvs:{}'.format(x.shape))
            #print(r'enc feature:{}'.format(encFeatures[i].shape))
            encFeat = self.crop(encFeatures[i],x)
            #print(r'enc shape:{}'.format(encFeat.shape))
            x = torch.cat([x,encFeat],dim = 1)
            #print(r'after cat: {}'.format(x.shape))
            x = self.dec_blocks[i](x)
            #print(r'after dec_block: {}'.format(x.shape))
        return x
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
    def __init__(self,encChannels=[1,16,32,64],
                 decChannels = [64,32,16],
                 nbClassses = 4,retainDim = True,
                 outSize= (config.INPUT_IMAGE_HEIGHT,config.INPUT_IMAGE_WIDTH)):
        super().__init__()
        self.encoder = Encoder(encChannels)
        self.decoder = Decoder(decChannels)

        self.head = nn.Conv2d(decChannels[-1],nbClassses,1)
        #self.sigmoid = nn.Sigmoid()
        self.retainDim = retainDim
        self.OutSize = outSize

    def forward(self,x):
        encFeatures = self.encoder(x)
        decFeatures = self.decoder(encFeatures[::-1][0],
                                   encFeatures[::-1][1:])
        map = self.head(decFeatures)
        #map = self.sigmoid(map)

        if self.retainDim:
            map = F.interpolate(map,self.OutSize)

        return map











