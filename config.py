from torch.cuda import is_available
'''
stores the choise of parameters
as long as some static variables
'''
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
VERSION = 'unet_10_28_v2'
INIT_LR = 0.002
WEIGHT_DECAY = 0.0001
NUM_EPOCHS = 80
BATCH_SIZE = 16

MODEL_FOLDER = 'OUTPUT/Model/'
LOSS_FOLDER = 'OUTPUT/Loss Plot/'
PLOT_FOLDER = 'OUTPUT/Output/'


