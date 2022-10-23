import numpy as np
import cv2
import pydicom
import os
import config
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import re

class path_loader:
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

                patient_image_folder, mask_image_folder = prostatex_dir + 't2_tse_tra/', prostatex_dir + 'mask/'

                image_list = os.listdir(patient_image_folder)
                reg = re.compile(r'.+\.dcm$')
                reg_image_list = list(filter(reg.findall, image_list))

                total_num = len(reg_image_list)
                idx = np.random.choice(np.arange(total_num))
                image_dir = patient_image_folder + reg_image_list[idx]
                mask_dir = mask_image_folder + reg_image_list[idx][:-4]+ '.png'
                self.image_path.append(image_dir)
                self.mask_path.append(mask_dir)


class FetchImage(Dataset):
    def __init__(self, imagePaths, maskPaths, transforms, hist = True):
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
        self.transforms = transforms
        self.hist = hist


    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.imagePaths)

    def __getitem__(self, idx):
        # grab the image path from the current index
        imagePath = self.imagePaths[idx]
        maskPath = self.maskPaths[idx]
        # load the image from disk
        # convert it to uint8
        # and read the associated mask from disk in grayscale mode
        image = pydicom.dcmread(imagePath).pixel_array
        image_uint8 = cv2.convertScaleAbs(image, alpha=(255.0 / 65535.0))

        mask = cv2.imread(maskPath,0)

        # down scale the iamge to 256 * 256
        dim = (config.INPUT_IMAGE_WIDTH, config.INPUT_IMAGE_HEIGHT)
        image_pre = self.preprocessor(arr = image_uint8,dim = dim, hist = self.hist,img = True)
        mask_pre = self.preprocessor(arr = mask,dim = dim, mask = True)

        if self.transforms is not None:
            image_final = self.transforms(image_pre)
            mask_final = self.transforms(mask_pre)
        # return a tuple of the image and its mask
        return (image_final, image_final)

    def preprocessor(self,arr,dim,hist = False,mask = False,img = False):
        if img:
            if arr.shape[0] != config.INPUT_IMAGE_HEIGHT or arr.shape[1] != config.INPUT_IMAGE_WIDTH:
                arr_resize = cv2.resize(arr, dim)
                if hist:
                    clahe = cv2.createCLAHE(clipLimit = 3.5)
                    output_array = clahe.apply(arr_resize) + 30
                else: output_array = arr_resize
            else:
                if hist:
                    clahe = cv2.createCLAHE(clipLimit = 3.5)
                    output_array = clahe.apply(arr) + 30
                else: output_array = arr
        elif mask:
            idx2 = (arr == 3)
            arr[idx2] = 2
            
            if arr.shape[0] != config.INPUT_IMAGE_HEIGHT or arr.shape[1] != config.INPUT_IMAGE_WIDTH:
                arr_resize = cv2.resize(arr, dim,,interpolation = cv2.INTER_NEAREST)
            else: arr_resize = arr
            output_array = np.zeros((4,config.INPUT_IMAGE_HEIGHT,config.INPUT_IMAGE_WIDTH),dtype = 'uint8')
            output_array[0] = (arr_resize==0).astype(int)
            output_array[1] = (arr_resize==1).astype(int)
            output_array[2] = (arr_resize==2).astype(int)
            output_array[3] = (arr_resize==255).astype(int)

        return output_array

'''
384 * 384

320 * 320

640 * 640



'''