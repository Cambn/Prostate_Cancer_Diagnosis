import torch.nn as nn
import torch
import numpy as np
from torchvision import transforms
from torch.cuda import is_available
from data_loader import load_data
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
import pydicom
import os
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from sklearn.model_selection import train_test_split

'''
t2-w image , dicom, (384,384,3)
mask image , png, (384,384,3)
'''

'''
building block for the encoder and decoder
stores two convolution, one batch normalization, and one leaky relu activation
'''
class Block(nn.Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.conv1 = nn.Conv2d(inChannels,outChannels,3)
        self.batchnorm = nn.BatchNorm2d(outChannels)
        self.leakyRelu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(outChannels,outChannels,3)
    def forward(self,x):
        return self.conv2(self.leakyRelu(self.conv1(x)))

'''
spatial dimension 384 -> 128 -> 64 -> 32
channels:         3   -> 16  -> 32 -> 64
'''
class Encoder(nn.Module):
    def __init__(self,channels= [3,16,32,64]):
        super().__init__()
        self.encBlocks = nn.ModuleList(
            [Block(channels[i],channels[i+1])
                   for i in range(len(channels)-1)]
        )
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(3)
    def forward(self,x):
        blocks = []
        for i in range(len(self.encBlocks)):
            block = self.encBlocks[i]
            x = block(x)
            blocks.append(x)
            if i == 0:
                x = self.pool2(x)
            else:
                x = self.pool1(x)
        return blocks

'''
spatial dimension 32 -> 64 -> 128 -> 384
channels: 64 -> 32 -> 16 -> 3
'''
class Decoder(nn.Module):
    def __init__(self,channels = [64,32,16,3]):
        super().__init__()
        self.channels = channels
        # self.upsample1 = nn.ConvTranspose2d()
        # self.upsample2 = nn.ConvTranspose2d()
        # self.upconvs = nn.ModuleList(
        #     [nn.ConvTranspose2d(channels[i],channels[i+1])
        #      for i in range(len(channels) - 1)]
        # )
        self.dec_blocks = nn.ModuleList(
            [Block(channels[i],channels[i+1])
                for i in range(len(channels) - 1)]
        )
    def forward(self,x,encFeatures):
        for i in range(len(self.channels) -1):
            if i == len(self.channels) - 2:
                block = self._block_upsample_(self.channels[i],self.channels[i+1],3,2)
            else:
                block = self._block_upsample_(self.channels[i], self.channels[i + 1],2, 2)
            x = block(x)
            encFeat = self.crop(encFeatures[i],x)
            x = torch.cat([x,encFeat],dim = 1)
            x = self.dec_blocks[i](x)
    def _block_upsample_(self,inChannels,outChannels,kernel,stride,batchnorm=True):
        return nn.ConvTranspose2d(in_channels = inChannels,out_channels = outChannels,kernel_size=kernel,stride = stride),

    def crop(self,encFeatures,x):
        (_,_,H,W) = x.shape
        encFeatures = CenterCrop([H,W])(encFeatures)

        return encFeatures



class U_net:
    """
    implements u_net architecture
    """
    def __init__(self,encChannels=[3,16,32,64],
                 decChannels = [64,32,16,3],
                 nbClassses = 1):
        super().__init__()
        self.encoder = Encoder(encChannels)
        self.decoder = Decoder(decChannels)

        self.head = nn.Conv2d(decChannels[-1],nbClassses,1)
        # self.retainDim = retainDim
        # self.OutSize = outSize

    def forward(self,x):
        encFeatures = self.encoder(x)
        decFeatures = self.decoder(encFeatures[::-1][0],
                                   encFeatures[::-1][1:])
        map = self.head(decFeatures)

        # if self.retainDim:
        #     map = F.interpolate(map,self.outSize)

        return map




if __name__ == '__main__':
    loader = load_data()
    d = 'DATASET/'
    train_test = 0.3
    rand_state = 42
    loader.get_path(d)
    (im_path, mask_path) = loader.load_path()

    DEVICE = 'cuda' if is_available() else 'epu'
    # define threshold to filter weak predictions
    THRESHOLD = 0.5
    lr, num_epoch, batch_size = 0.001, 40, 64
    BASE_OUTPUT = "output"
    imagePaths = sorted(list(im_path))
    maskPaths = sorted(list(mask_path))
    split = train_test_split(imagePaths,maskPaths,test_size=train_test,random_state = rand_state)
    '''
    total number of images for dicom and mask: 4104
    # of training batch: 2872
    '''
    (trainImages,testImages) = split[:2]
    (trainMasks, testMasks) = split[2:]
    transforms = transforms.Compose([transforms.ToPILImage(),
                                     transforms.ToTensor()])
    print('Images {}'.format(len(imagePaths)))
    print('trainImages {}'.format(len(trainImages)))
    print('trainMasks {}'.format(len(trainMasks)))






