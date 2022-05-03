import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
import cv2
import pydicom
import os

class load_data:
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

