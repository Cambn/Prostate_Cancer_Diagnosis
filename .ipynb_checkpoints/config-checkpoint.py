from torch.cuda import is_available
import os
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
INPUT_IMAGE_WIDTH = 384
INPUT_IMAGE_HEIGHT = 384

THRESHOLD = 0.5

INIT_LR = 0.001
NUM_EPOCHS = 10
BATCH_SIZE = 1

NUM_CHANNELS = 3
NUM_CLASSES = 3
NUM_LEVELS = 3

BASE_OUTPUT = 'OUTPUT'

MODEL_PATH = os.path.join(BASE_OUTPUT, "unet_alpha1.0.pth")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot_alpha1.0.png"])
#TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])