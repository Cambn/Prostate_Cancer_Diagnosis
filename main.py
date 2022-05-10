import os
from data_loader import load_path,FetchImage
from prostate_seg import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from torchvision import transforms
from sklearn.model_selection import train_test_split
import config


if __name__ == '__main__':
    loader = load_path()
    d = config.DATASET_MAIN_BRUNCH
    train_test = config.TEST_SPLIT
    rand_state = 42
    loader.get_path(d)
    (im_path, mask_path) = loader.load_path()

    DEVICE = config.DEVICE
    # define threshold to filter weak predictions
    THRESHOLD = config.THRESHOLD

    imagePaths = sorted(list(im_path))
    maskPaths = sorted(list(mask_path))
    split = train_test_split(imagePaths,maskPaths,test_size=train_test,random_state = rand_state)
    '''
    total number of images for dicom and mask: 4104
    # of training batch: 2749
    '''
    (train_paths_Images,train_paths_Masks) = split[:2]
    (test_paths_Images, test_paths_Masks) = split[2:]
    transforms = transforms.Compose([transforms.ToPILImage(),
                                     transforms.ToTensor()])
    _train = FetchImage(train_paths_Images,train_paths_Masks,transforms)
    _test = FetchImage(test_paths_Images, test_paths_Masks, transforms)
    train_Loader = DataLoader(_train,shuffle = True, batch_size = config.BATCH_SIZE,
                              pin_memory = config.PIN_MEMORY,num_workers = os.cpu_count()-1)
    test_Loader = DataLoader(_test, shuffle=True, batch_size=config.BATCH_SIZE,
                              pin_memory=config.PIN_MEMORY, num_workers=os.cpu_count() - 1)
    unet = U_net().to(config.DEVICE)



    # enc_block = decoder_block(3,1)
    # x = torch.randn(1,3,5,5)
    # print(enc_block(x).shape)
    # print(train_loader[1][1])
    # plt.imshow(train_loader[1][0])
    # plt.show()
    # print('Images {}'.format(len(imagePaths)))
    # print('trainImages {}'.format(len(trainImages)))
    # print('trainMasks {}'.format(len(trainMasks)))