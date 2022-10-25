import numpy as np
import cv2
import pydicom
import os
import bm3d
import config
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from scipy.ndimage.interpolation import shift,rotate
import re
from tqdm import tqdm


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

class dataset_preperation():
    def __init__(self,im_path,mask_path,train_test_split,train = True):
        self.img_dataset = []
        self.mask_dataset = []
        self.im_path = im_path
        self.mask_path = mask_path
        self.train = train
        
    def dicom_image_preparation(self,img):
        img = img.astype(float)
        if img.shape[0] != config.INPUT_IMAGE_HEIGHT or img.shape[1] != config.INPUT_IMAGE_WIDTH:
            dim = (config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)
            img = cv2.resize(img, dim,interpolation = cv2.INTER_NEAREST)
        img_normalized = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX)
        img_final = bm3d.bm3d(img_normalized, sigma_psd=30/255, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
        return img_normalized

    def mask_preparation(self,mask):
        idx = (mask == 3)
        mask[idx] = 2            
        if mask.shape[0] != config.INPUT_IMAGE_HEIGHT or mask.shape[1] != config.INPUT_IMAGE_WIDTH:
            dim = (config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)
            mask = cv2.resize(mask, dim,interpolation = cv2.INTER_NEAREST)
        return mask

    def image_transition(self,img,direction:str,pixel_val:int):
        if direction == 'left':
            img_t = shift(img,[0,-pixel_val])
        elif direction == 'right':
            img_t = shift(img,[0,pixel_val])
        elif direction == 'up':
            img_t = shift(img,[-pixel_val,0])
        else:
            img_t = shift(img,[pixel_val,0])
        return img_t

    def image_rotation(self,img,degree:int):
        return rotate(img,degree)
    def read_preprocess_dicom_mask(self,verbose = False):
        if verbose:
            if self.train: 
                print('Loading and Processing datasets for Training...')
                print('...')
            else:
                print('Loading and Processing datasets for Testing...')
                print('...')
        for i,m in tqdm(zip(self.im_path,self.mask_path)):
            img = pydicom.dcmread(i).pixel_array
            img_prepared = self.dicom_image_preparation(img)
            mask = cv2.imread(m,0)
            mask_prepared = self.mask_preparation(mask)
            self.img_dataset.append(img_prepared)
            self.mask_dataset.append(mask_prepared)
            if verbose and i == self.im_path[-1]:
                print('Loading images and masks finished.')
            if self.train:
                ## transition left and right by 5 pixels
                num_shift = 5
                self.img_dataset.append(self.image_transition(img_prepared,'left',num_shift))
                self.img_dataset.append(self.image_transition(img_prepared,'right',num_shift))
                self.mask_dataset.append(self.image_transition(mask_prepared,'left',num_shift))
                self.mask_dataset.append(self.image_transition(mask_prepared,'right',num_shift))
                ## rotation 90 and 270 degrees
                d_1, d_2 = 90, 270
                self.img_dataset.append(self.image_rotation(img,d_1))
                self.img_dataset.append(self.image_rotation(img,d_2))
                self.mask_dataset.append(self.image_rotation(mask_prepared,d_1))
                self.mask_dataset.append(self.image_rotation(mask_prepared,d_2))
                if verbose and i == self.im_path[-1]:
                    print('Loading augmented images and masks finished.')
                    print(f'Augmentation: shift pics left and right by {num_shift} pixels. \nRotate pics by {d_1} and {d_2} degrees')

        return [self.img_dataset,self.mask_dataset]
    
class FetchImage(Dataset):
    def __init__(self, image_dataset, mask_dataset, transformation):
        self.image_dataset = image_dataset
        self.mask_dataset = mask_dataset
        self.transformation = transformation

    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.image_dataset)
    
    def mask_dim_exp(self,mask):
        output_mask = np.zeros((mask.shape[0],mask.shape[1],4),dtype = 'uint8')
        output_mask[...,0] = (mask==0).astype(int)
        output_mask[...,1] = (mask==1).astype(int)
        output_mask[...,2] = (mask==2).astype(int)
        output_mask[...,3] = (mask==255).astype(int)
        return output_mask
    
    def __getitem__(self, idx):
        # grab the imagefrom the current index
        image = self.image_dataset[idx]
        image = cv2.convertScaleAbs(image, alpha=255/image.max())
        mask = self.mask_dataset[idx]
        mask = self.mask_dim_exp(mask)
        if self.transformation is not None:
            image_final = self.transformation(image)
            image_final = transforms.Normalize(0,1)(image_final)
            mask_final = self.transformation(mask)
            # return a tuple of the image and its mask
        return (image_final, mask_final)
'''
384 * 384

320 * 320

640 * 640



'''