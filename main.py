from data_loader import load_path,FetchImage
from prostate_seg import U_net
from torch.cuda import is_available
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from torchvision import transforms
from torch.cuda import is_available
import matplotlib.pyplot as plt
import pydicom
import cv2
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    loader = load_path()
    d = 'DATASET/'
    train_test = 0.3
    rand_state = 42
    loader.get_path(d)
    (im_path, mask_path) = loader.load_path()

    DEVICE = 'cuda' if is_available() else 'epu'
    # define threshold to filter weak predictions
    THRESHOLD = 0.5
    lr, num_epoch, batch_size = 0.001, 40, 64
    BASE_OUTPUT = "output"
    imagePaths = sorted(list(im_path))
    maskPaths = sorted(list(mask_path))
    split = train_test_split(imagePaths,maskPaths,test_size=train_test,random_state = rand_state)
    '''
    total number of images for dicom and mask: 4104
    # of training batch: 2872
    '''
    (trainImages,testImages) = split[:2]
    (trainMasks, testMasks) = split[2:]
    transforms = transforms.Compose([transforms.ToPILImage(),
                                     transforms.ToTensor()])

    train_loader = FetchImage(trainImages,trainMasks,None)

    print(train_loader[1][1])
    plt.imshow(train_loader[1][0])
    plt.show()
    # print('Images {}'.format(len(imagePaths)))
    # print('trainImages {}'.format(len(trainImages)))
    # print('trainMasks {}'.format(len(trainMasks)))