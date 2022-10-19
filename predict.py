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
from pydicom.tag import Tag
from torch.nn import MultiLabelMarginLoss,CrossEntropyLoss

if __name__ == '__main__':
    d = config.BASE_OUTPUT
    model = '/unet_alpha1.10.19.pth'
    unet = torch.load(d + model)
    unet = unet.to(config.DEVICE)
    lossFunc = CrossEntropyLoss()
    unet.eval()
    '''
    get one of the images for testing
    img_1 -> pixel_array of image
    img_uint8 -> img in the format of uintr8
    
    '''
    test_img = ['DATASET/Prostatex-0026/t2_tse_tra/IM-0110-0008.dcm']
    test_mask = ['DATASET/Prostatex-0026/mask/IM-0110-0008.png']
    t = transforms.Compose([transforms.ToPILImage(),
                            transforms.ToTensor()])
    _model_test = FetchImage(test_img, test_mask, t)
    #img_1 = pydicom.dcmread(test).pixel_array
    #img_uint8 = cv2.convertScaleAbs(img_1, alpha=(255.0 / 65535.0))


    #
    # #_input = t(img_uint8)
    # #_input = _input.unsqueeze(1)
    # # __input = torch.squeeze(_input,dim = 1)
    # #__input = _input.to('cpu')
    #
    img_Loader = DataLoader(_model_test, batch_size=1,
                            pin_memory=config.PIN_MEMORY)

    with torch.no_grad():
        for (x,y) in img_Loader:
            (x,y) = (x.to(config.DEVICE), y.to(config.DEVICE).long())
            y = torch.squeeze(y, dim=1)
            pred = unet(x)
            #pred = torch.argmax(pred, dim=1)
            print(lossFunc(pred,y))

    m = nn.Softmax(dim=3)
    pred_softmax = m(pred)
    p = pred.detach().cpu().numpy()
    p_softmax = pred_softmax.detach().cpu().numpy()
    plt.figure(figsize=(15, 12))
    plt.imshow(np.argmax(p_softmax, axis=1).reshape(256, 256) * 10)#, cmap='gray')
    plt.show()




    # _pred3 = (_pred3 > 0.1).astype(np.uint8)
    # # plt.figure(figsize=(15, 12))
    # # plt.imshow(img_1)
    # # plt.show()

    # _pred3 = _pred3 * 255
    # # print(_pred3)
    # plt.figure(figsize=(15, 12))
    # plt.imshow(_pred3)
    # plt.show()
