import os
import re
import numpy as np
from data_loader import *
from prostate_seg import *
import torch
# import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MultiLabelMarginLoss,CrossEntropyLoss
import time
from tqdm import tqdm
from torchvision import transforms
from sklearn.model_selection import train_test_split
import config

import logging
if __name__ == '__main__':
    loader = path_loader()

    loader.get_path(config.DATASET_MAIN_BRUNCH)
    (im_path, mask_path) = loader.load_path()
    split = train_test_split(im_path,mask_path,test_size=config.TEST_SPLIT,random_state = config.RAND_STATE)
    train_paths_Images, test_paths_Images, train_paths_Masks, test_paths_Masks = split[0], split[1], split[2], split[3]
    transforms = transforms.Compose([transforms.ToPILImage(),
                                     transforms.ToTensor()])
    _train = FetchImage(train_paths_Images, train_paths_Masks, transforms)
    _test = FetchImage(test_paths_Images, test_paths_Masks, transforms)
    train_Loader = DataLoader(_train, shuffle=True, batch_size=config.BATCH_SIZE,
                              pin_memory=config.PIN_MEMORY, num_workers=4)
    test_Loader = DataLoader(_test, shuffle=True, batch_size=config.BATCH_SIZE,
                             pin_memory=config.PIN_MEMORY, num_workers=4)
    for (i, (x, y)) in enumerate(train_Loader):
        print(i, x.shape,y.shape)

