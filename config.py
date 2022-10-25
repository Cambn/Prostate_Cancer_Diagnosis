from torch.cuda import is_available
import os
import numpy as np
import argparse
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

def plot_figure(x,pred,y):
    x_numpy = x[0][0].cpu().numpy()
    plt.imshow(x_numpy,cmap = 'gray')
    pred_sample = pred[0][0]
    pred_numpy = pred_sample.cpu().detach().numpy()
    pred_sigmoid = Sigmoid()(pred_sample).cpu().detach().numpy()
    pred_sigmoid_th = ((pred_sigmoid> 0.5) * 255).astype(np.uint8)
    y_numpy = y[0][0].cpu().detach().numpy()

    fig,ax = plt.subplots(1,4, figsize=(15, 6), facecolor='w', edgecolor='k')
    ax[0].imshow(y_numpy)
    ax[0].set_title(f'Groud Truth Mask')
    ax[1].imshow(pred_numpy)
    ax[1].set_title(f'Output Mask')
    ax[2].imshow(pred_sigmoid)
    ax[2].set_title(f'Sigmoid Mask')
    ax[3].imshow(pred_sigmoid_th)
    ax[3].set_title(f'Sigmoid Mask with th 0.5')
    # for i in range(4):
    #     ax[0,i].imshow(y_numpy[i,...])
    #     ax[0,i].set_title(f'Groud Truth Mask Channel {i}')
    #     ax[1,i].imshow(pred_numpy[i,...])
    #     ax[1,i].set_title(f'Output Mask Channel {i}')
    #     ax[2,i].imshow(pred_numpy[i,...])
    #     ax[2,i].set_title(f'Sigmoid Mask Channel {i}')
    #     ax[3,i].imshow(pred_sigmoid_th[i,...])
    #     ax[3,i].set_title(f'Sigmoid Mask with threshold Channel {i}')
    plt.show()

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

THRESHOLD = 0.5

INIT_LR = 0.005
WEIGHT_DECAY = 0.0001
NUM_EPOCHS = 100
BATCH_SIZE = 20

