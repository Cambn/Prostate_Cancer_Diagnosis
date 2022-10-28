import config

from data_loader import *
import warnings
warnings.simplefilter("ignore", UserWarning)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config import *
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
import pydicom
from torch.nn import BCEWithLogitsLoss,CrossEntropyLoss

if __name__ == '__main__':
    model_name = config.VERSION + '.pth'
    plot_folder_predict = config.PLOT_FOLDER + config.VERSION + '/predict/'
    if not os.path.isdir(plot_folder_predict):
        os.makedirs(plot_folder_predict)
    complete_model_path = config.MODEL_FOLDER + model_name
    binary = True
    unet = torch.load(complete_model_path)
    unet = unet.to(config.DEVICE)
    lossFunc = BCEWithLogitsLoss()
    unet.eval()
    '''
    get one of the images for testing
    img_1 -> pixel_array of image
    img_uint8 -> img in the format of uint8
    
    '''
    test_img = ['DATASET/Prostatex-0000/t2_tse_tra/IM-0002-0010.dcm',
                'DATASET/Prostatex-0001/t2_tse_tra/IM-0007-0009.dcm',
                'DATASET/Prostatex-0002/t2_tse_tra/IM-0010-0007.dcm']
    test_mask = ['DATASET/Prostatex-0000/mask/IM-0002-0010.png',
                 'DATASET/Prostatex-0001/mask/IM-0007-0009.png',
                 'DATASET/Prostatex-0002/mask/IM-0010-0007.png']
    test_data = dataset_preperation(test_img, test_mask, False)
    test_dataset = test_data.read_preprocess_dicom_mask('both', binary, True)

    img, mask = test_dataset[0], test_dataset[1]

    transformation = True
    data = FetchImage(img, mask, transformation)

    loader = DataLoader(data, shuffle=True, batch_size=config.BATCH_SIZE,
                              pin_memory=config.PIN_MEMORY, num_workers=4)


    with torch.no_grad():
        for (i,(x,y)) in enumerate(loader):
            (x,y) = (x.to(config.DEVICE), y.to(config.DEVICE))
            pred = unet(x)

            print(lossFunc(pred,y.float()))
            config.plot_figure(x,pred,y,0,plot_folder_predict, False)