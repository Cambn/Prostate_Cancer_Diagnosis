import numpy as np
from torchvision import transforms
import torch
from PIL import Image
import cv2
import pydicom
import os

class load_path:
    def __init__(self):
        self.im = []
        self.gt_mask = []
        self.patient_name = []
        self.mask_path = []
        self.image_path = []
    def load_path(self,transform = None):
        return (self.image_path,self.mask_path)
    def get_path(self,main_brunch:str):
        list_under_main = os.listdir(main_brunch)
        for prostatex in list_under_main:
            if prostatex.__contains__('Prostatex'):
                prostatex_dir = main_brunch + prostatex + '/'
                list_under_prostatex = os.listdir(prostatex_dir)
                for folders in list_under_prostatex:
                    if folders == 't2_tse_tra':
                        t2_dir = prostatex_dir + folders + '/'
                        list_dir_1 = os.listdir(t2_dir)
                        for images in list_dir_1:
                            if images.endswith('.dcm'):
                                image_pa = t2_dir + images
                                self.image_path.append(image_pa)
                    elif folders == 'mask':
                        mask_dir = prostatex_dir + folders + '/'
                        list_dir_mask = os.listdir(mask_dir)
                        for masks in list_dir_mask:
                            if masks.endswith('.png'):
                                mask_pa = mask_dir + masks
                                self.mask_path.append(mask_pa)

class FetchImage:
    def __init__(self, imagePaths, maskPaths, transforms = False):
        # store the image and mask filepaths, and augmentation
        # transforms
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
        self.transforms = transforms

    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.imagePaths)

    def __getitem__(self, idx):
        # grab the image path from the current index
        imagePath = self.imagePaths[idx]
        maskPath = self.maskPaths[idx]
        # load the image from disk, swap its channels from BGR to RGB,
        # and read the associated mask from disk in grayscale mode
        image = pydicom.dcmread(imagePath).pixel_array.astype(int)
        print(image.shape)
        mask = cv2.imread(maskPath, 0)
        # check to see if we are applying any transformations
        if self.transforms:
            # apply the transformations to both image and its mask
            image = torch.tensor(image)
            mask = torch.tensor(mask)
        # return a tuple of the image and its mask
        return (image, mask)
