from torch.cuda import is_available
import os
import argparse
import matplotlib.pyplot as plt
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
BASE_OUTPUT = 'OUTPUT'

MODEL_PATH = os.path.join(BASE_OUTPUT, "unet_alpha1.1.pth")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot_alpha1.1.png"])

def plot_loss(H,path):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["test_loss"], label="test_loss")
    plt.title("Training Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(PLOT_PATH)
    plt.show()

DATASET_MAIN_BRUNCH = 'DATASET/'
TEST_SPLIT = 0.33
DEVICE = 'cuda' if is_available else 'cpu'
#
# PIN_MEMORY = True if DEVICE == "cuda" else False
#DEVICE = 'cpu'
PIN_MEMORY = False
RAND_STATE = 42
INPUT_IMAGE_WIDTH = 384
INPUT_IMAGE_HEIGHT = 384

THRESHOLD = 0.5

INIT_LR = 0.001
NUM_EPOCHS = 20
BATCH_SIZE = 2

NUM_CHANNELS = 3
NUM_CLASSES = 3
NUM_LEVELS = 3



#TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])