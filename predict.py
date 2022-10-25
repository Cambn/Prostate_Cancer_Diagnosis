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
    model_name = 'unet_10_22_v3.pth'
    complete_model_path = config.MODEL_FOLDER + model_name
    unet = torch.load(complete_model_path)
    unet = unet.to(config.DEVICE)
    lossFunc = BCEWithLogitsLoss()
    unet.eval()
    '''
    get one of the images for testing
    img_1 -> pixel_array of image
    img_uint8 -> img in the format of uintr8
    
    '''
    test_img = ['DATASET/Prostatex-0000/t2_tse_tra/IM-0002-0010.dcm']
    test_mask = ['DATASET/Prostatex-0000/mask/IM-0002-0010.png']
    t = transforms.Compose([transforms.ToPILImage(),
                            transforms.ToTensor()])
    _model_test = FetchImage(test_img, test_mask, t)
    img_Loader = DataLoader(_model_test, batch_size=1,
                            pin_memory=config.PIN_MEMORY)

    with torch.no_grad():
        for (x,y) in img_Loader:
            (x,y) = (x.to(config.DEVICE), y.to(config.DEVICE))
            pred = unet(x)

            #pred = torch.argmax(pred, dim=1)
            print(lossFunc(pred,y).flatten())
    pred_mask = pred.cpu().numpy()

    pred_mask = ((pred_mask> 0.5) * 255).astype(np.uint8)[0]
    output_image = list(FetchImage(test_img,test_mask,None,hist = True))
    prostate_img,gt = output_image[0][0],output_image[0][1]
    plt.figure(figsize=(15, 12))
    plt.imshow(prostate_img)
    plt.title('Test Image')
    fig, axs = plt.subplots(1, 4, figsize=(15, 6), facecolor='w', edgecolor='k')
    axs = axs.ravel()
    for i in range(4):
        axs[i].imshow(gt[...,i])
        axs[i].set_title('Group Truth Mask Region ' + str(i))

    fig, axs = plt.subplots(1, 4, figsize=(15, 6), facecolor='w', edgecolor='k')
    axs = axs.ravel()
    pred_numpy = pred.clone().detach().cpu()[0]

    for i in range(4):
        axs[i].imshow(pred_mask[i,...])
        axs[i].set_title('Predicted Mask Region ' + str(i))
    plt.show()