import numpy as np
import cv2
import pydicom
import os
import config
from torch.utils.data import Dataset


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

class FetchImage(Dataset):
    def __init__(self, imagePaths, maskPaths, transforms = None):
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
        # load the image from disk
        # and read the associated mask from disk in grayscale mode
        image = pydicom.dcmread(imagePath).pixel_array
        image = self.gray_2d(image)
        mask = cv2.imread(maskPath,0)
        mask = self.gray_2d(mask)

        dim = (config.INPUT_IMAGE_WIDTH, config.INPUT_IMAGE_HEIGHT)
        if image.shape[0] != config.INPUT_IMAGE_HEIGHT or image.shape[1] != config.INPUT_IMAGE_WIDTH:
            print(image.shape)
            image = cv2.resize(image, dim)
            print('Shape of image from path {}: {}'.format(imagePath, image.shape))
        if mask.shape[0] != config.INPUT_IMAGE_HEIGHT or mask.shape[1] != config.INPUT_IMAGE_WIDTH:
            # print('Shape of mask from path {}: {}'.format(maskPath, mask.shape))
            print(mask.shape)
            mask = cv2.resize(mask, dim)
            # print('Shape of mask from path {}: {}'.format(maskPath, mask.shape))

        # check to see if we are applying any transformations
        if self.transforms is not None:
            # apply the transformations to both image and its mask
            image = self.transforms(image)
            mask = self.transforms(mask)


        # return a tuple of the image and its mask
        return (image, mask)

    def gray_2d(self,arr):
        # Convert to float to avoid overflow or underflow losses.
        img_2d = arr.astype(float)
        # Rescaling grey scale between 0-255
        if img_2d.max() != 0:
            _N = img_2d.max()
        else:
            _N = 1
        img_2d_scaled = (np.maximum(img_2d,0) / _N) * 255.0
        # convert to int32 for tensor
        image_2d_int32 = img_2d_scaled.astype(int)

        return image_2d_int32

'''
384 * 384

320 * 320

640 * 640



'''