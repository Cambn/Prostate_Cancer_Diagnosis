import os
import numpy as np
import argparse
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn import Sigmoid
import cv2
import config
def get_args():
    parser = argparse.ArgumentParser(description='unet model that detects pz and tz zones of prostates')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the unet model is stored')
    parser.add_argument('--loss_plot', '-b', default='PLOT.png', metavar='FILE',
                        help='Specify the file in which the plot of the loss is stored')
    parser.add_argument('--viz', action='store_true',
                        help='visualize the image/gt mask/predicted mask for the testing sets')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask_threshold', '-t', type=float, default=0.5,
                        help='probability value to consider a mask pixel white')
    return parser.parse_args()


def mask_img(mask, img):
    idx = np.where(mask != 0)
    mask[idx[0], idx[1]] = 255

    weighted_img = cv2.addWeighted(img, 0.7, mask.astype(np.float32), 0.3, 0)
    weighted_img = cv2.addWeighted(weighted_img, 0.7, mask.astype(np.float32), 0.3, 0)
    return weighted_img


def mask_diff(pred, gt):
    idx = np.where(pred != 0)
    pred[idx[0], idx[1]] = 255
    weighted_img = cv2.addWeighted(gt, 0.7, pred.astype(np.int64), 0.3, 0)
    return weighted_img


def plot_figure(x, pred, y, epoch, path, train=True):
    if not os.path.isdir(path):
        os.makedirs(path)
    N = x.size()[0]
    fig, ax = plt.subplots(N, 6, figsize=(40, 20))
    fig.tight_layout()
    for i in range(N):
        x_numpy = x[i][0].cpu().numpy() * 255
        y_numpy = y[i][0].cpu().detach().numpy()
        pred_sample = pred[i][0]
        pred_numpy = pred_sample.cpu().detach().numpy()
        pred_sigmoid = Sigmoid()(pred_sample).cpu().detach().numpy()
        pred_sigmoid_th = ((pred_sigmoid > config.THRESHOLD) * 255)
        gt_mask_img = mask_img(y_numpy, x_numpy)
        pred_mask_img = mask_img(pred_sigmoid_th, x_numpy)
        mask_comp = mask_diff(pred_sigmoid_th, y_numpy)

        ax[i, 0].imshow(x_numpy, cmap='gray')
        ax[i, 0].set_title(f'Image_{i} at epoch {epoch}')
        ax[i, 1].imshow(y_numpy, cmap='gray')
        ax[i, 1].set_title(f'GT Mask_{i} at epoch {epoch}')
        ax[i, 2].imshow(pred_sigmoid_th, cmap='gray')
        ax[i, 2].set_title(f'Pred Mask_{i} at epoch {epoch}')
        ax[i, 3].imshow(gt_mask_img, cmap='gray')
        ax[i, 3].set_title(f'GT With Img_{i} at epoch {epoch}')
        ax[i, 4].imshow(pred_mask_img, cmap='gray')
        ax[i, 4].set_title(f'Pred Mask With Img_{i} at epoch {epoch}')
        ax[i, 5].imshow(mask_comp, cmap='gray')
        ax[i, 5].set_title(f'Mask_Comp_{i} at epoch {epoch}')

    if train:
        output_path = path + ' visualizations_at_epoch_' + str(epoch) + '.png'
    else:
        output_path = path + 'visualizations_test.png'

    plt.savefig(output_path)



def plot_loss(H, path):
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


class Binary_DiceLoss(nn.Module):

    def __init__(self):
        super(Binary_DiceLoss, self).__init__()
        self.smooth = 1e-5

    def forward(self, predict, target):
        N = target.size()[0]
        pred_flat = predict.view(N, -1)
        target_flat = target.view(N, -1)
        intersection = pred_flat * target_flat
        dice_eff = (2 * intersection.sum(1) + self.smooth) / (pred_flat.sum(1) + target_flat.sum(1) + self.smooth)
        dice_loss = 1 - dice_eff.sum() / N
        return dice_loss