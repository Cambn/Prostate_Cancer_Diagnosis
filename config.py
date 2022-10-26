from torch.cuda import is_available
import os
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.nn import Sigmoid
'''
stores the choise of parameters
as long as some static variables
'''

def get_args():
    parser = argparse.ArgumentParser(description='unet model that detects pz and tz zones of prostates')
    parser.add_argument('--model','-m',default='MODEL.pth',metavar='FILE',
                        help = 'Specify the file in which the unet model is stored')
    parser.add_argument('--loss_plot', '-b', default='PLOT.png', metavar='FILE',
                        help='Specify the file in which the plot of the loss is stored')
    parser.add_argument('--viz',action='store_true',
                        help = 'visualize the image/gt mask/predicted mask for the testing sets')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask_threshold','-t',type =float,default=0.5,
                        help='probability value to consider a mask pixel white')
    return parser.parse_args()

MODEL_FOLDER = 'OUTPUT/Model/'
LOSS_FOLDER = 'OUTPUT/Loss Plot/'
PLOT_FOLDER = 'OUTPUT/Output/'


def plot_figure(x,pred,y,epoch,path,train = True):
    x_numpy = x[0][0].cpu().numpy()
    plt.imshow(x_numpy,cmap = 'gray')
    plt.title(f'Image on Epoch {epoch}')
    if train:
        img_p = path+' image_at_epoch_' + str(epoch)+'.png'
        mask_p = path + 'outputs_at_epoch_'+str(epoch)+'.png'
    else:
        img_p = path + 'image_for_test.png'
        mask_p = path + 'outputs_for_test.png'
    plt.savefig(img_p)
    pred_sample = pred[0][0]
    pred_numpy = pred_sample.cpu().detach().numpy()
    pred_sigmoid = Sigmoid()(pred_sample).cpu().detach().numpy()
    pred_sigmoid_th = ((pred_sigmoid> 0.5) * 255).astype(np.uint8)
    y_numpy = y[0][0].cpu().detach().numpy()

    fig,ax = plt.subplots(1,4, figsize=(15, 6), facecolor='w', edgecolor='k')
    ax[0].imshow(y_numpy,cmap='gray')
    ax[0].set_title(f'Groud Truth Mask')
    ax[1].imshow(pred_numpy,cmap='gray')
    ax[1].set_title(f'Output Mask')
    ax[2].imshow(pred_sigmoid,cmap='gray')
    ax[2].set_title(f'Sigmoid Mask')
    ax[3].imshow(pred_sigmoid_th,cmap='gray')
    ax[3].set_title(f'Sigmoid Mask with th 0.5')
    plt.savefig(mask_p)

    # for i in range(4):
    #     ax[0,i].imshow(y_numpy[i,...])
    #     ax[0,i].set_title(f'Groud Truth Mask Channel {i}')
    #     ax[1,i].imshow(pred_numpy[i,...])
    #     ax[1,i].set_title(f'Output Mask Channel {i}')
    #     ax[2,i].imshow(pred_numpy[i,...])
    #     ax[2,i].set_title(f'Sigmoid Mask Channel {i}')
    #     ax[3,i].imshow(pred_sigmoid_th[i,...])
    #     ax[3,i].set_title(f'Sigmoid Mask with threshold Channel {i}')


def plot_loss(H,path):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["test_loss"], label="test_loss")
    plt.title("Training Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(path)
    plt.show()

DATASET_MAIN_BRUNCH = 'DATASET/'
TEST_SPLIT = 0.33
DEVICE = 'cuda' if is_available else 'cpu'
PIN_MEMORY = True if DEVICE == "cuda" else False
# DEVICE = 'cpu'
# PIN_MEMORY = False
RAND_STATE = 42
INPUT_IMAGE_WIDTH = 320
INPUT_IMAGE_HEIGHT = 320
CROP_IMAGE_WIDTH = 256
CROP_IMAGE_HEIGHT = 256

THRESHOLD = 0.5

INIT_LR = 0.002
WEIGHT_DECAY = 0.0001
NUM_EPOCHS = 100
BATCH_SIZE = 160

class Binary_DiceLoss(nn.Module):

    def __init__(self):
        super(Binary_DiceLoss, self).__init__()
        self.smooth = 1e-5
    def forward(self, predict, target):
        N = target.size()[0]
        pred_flat = predict.view(N,-1)
        target_flat = target.view(N,-1)
        intersection = pred_flat * target_flat
        dice_eff = (2 * intersection.sum(1) + self.smooth)/ (pred_flat.sum(1) + target_flat.sum(1) + self.smooth)
        dice_loss = 1-dice_eff.sum()/N
        return dice_loss
