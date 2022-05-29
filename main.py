import os
from data_loader import load_path,FetchImage
from prostate_seg import *
import torch
# import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MultiLabelMarginLoss,CrossEntropyLoss
import time
from tqdm import tqdm
from torchvision import transforms
from sklearn.model_selection import train_test_split
import config
import matplotlib.pyplot as plt


if __name__ == '__main__':
    loader = load_path()

    loader.get_path(config.DATASET_MAIN_BRUNCH)
    (im_path, mask_path) = loader.load_path()

    DEVICE = config.DEVICE
    # define threshold to filter weak predictions
    THRESHOLD = config.THRESHOLD

    imagePaths = sorted(list(im_path))[:1000]
    maskPaths = sorted(list(mask_path))[:1000]
    split = train_test_split(imagePaths,maskPaths,test_size=config.TEST_SPLIT,random_state = config.RAND_STATE)
    '''
    total number of images for dicom and mask: 4104
    # of training batch: 2749
    '''
    train_paths_Images,test_paths_Images,train_paths_Masks,test_paths_Masks = split[0],split[1],split[2],split[3]
    transforms = transforms.Compose([transforms.ToPILImage(),
                                     transforms.ToTensor()])
    _train = FetchImage(train_paths_Images,train_paths_Masks,transforms)
    _test = FetchImage(test_paths_Images, test_paths_Masks, transforms)
    train_Loader = DataLoader(_train,shuffle = True, batch_size = config.BATCH_SIZE,
                              pin_memory = config.PIN_MEMORY,num_workers = 4)
    test_Loader = DataLoader(_test, shuffle=True, batch_size=config.BATCH_SIZE,
                              pin_memory=config.PIN_MEMORY,num_workers = 4)
    unet = U_net().to(config.DEVICE)
    lossFunc = CrossEntropyLoss()
    opt = Adam(unet.parameters(), lr=config.INIT_LR)

    trainSteps = len(train_paths_Images) // config.BATCH_SIZE
    testSteps = len(test_paths_Images) // config.BATCH_SIZE

    H = {"train_loss": [], "test_loss": []}
    startTime = time.time()

    for e in tqdm(range(config.NUM_EPOCHS)):
        unet.train()

        totalTrainLoss, totalTestLoss = 0, 0

        for (i,(x,y)) in enumerate(train_Loader):

            (x,y) = (x.to(config.DEVICE), y.to(config.DEVICE).long())
            y = torch.squeeze(y,dim = 1)
            #print(y.shape)
            pred = unet(x)
            #print(r'pred before argmax: {}'.format(pred.shape))
            #pred = torch.argmax(pred,dim = 1)
            #print(r'pred after argmax: {}'.format(pred.shape))
            loss = lossFunc(pred,y)

            opt.zero_grad()
            #loss.requires_grad = True
            loss.backward()
            opt.step()

            totalTrainLoss += lossFunc(pred,y)

        ## switch off autograd
        with torch.no_grad():

            ## set the model in evaluation mode
            unet.eval()

            ## loop over the validation set
            for (x,y) in test_Loader:
                (x,y) = (x.to(config.DEVICE), y.to(config.DEVICE).long())
                y = torch.squeeze(y, dim=1)
                pred = unet(x)
                #pred = torch.argmax(pred, dim=1)
                totalTestLoss += lossFunc(pred,y)

        avgTrainLoss = totalTrainLoss / trainSteps
        avgTestLoss = totalTestLoss / testSteps

        #print(avgTrainLoss)
        #print(avgTestLoss)
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["test_loss"].append(avgTestLoss.cpu().detach().numpy())
        print()
        print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
        print("Train loss: {:.6f}, Test loss: {:.4f}".format(avgTrainLoss, avgTestLoss))
    # display the total time needed to perform the training
    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["test_loss"], label="test_loss")
    plt.title("Training Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(config.PLOT_PATH)
    torch.save(unet, config.MODEL_PATH)




    #

    #print(train_paths_Masks[0])
    
    # plt.imshow(pydicom.dcmread(train_paths_Images[0]).pixel_array,cmap = 'gray')
    # plt.show()
    # print(train_Loader)
    # _m = 'DATASET/Prostatex-0006/mask/IM-0026-0012.png'
    # mask = cv2.imread(_m)#, 0)
    # print(mask[155])
    # #mask = self.gray_2d(mask)
    # # print()
    # plt.imshow(mask)#pydicom.dcmread(test_paths_Masks[1]).pixel_array,cmap = 'gray')
    # plt.show()
    # print('Images {}'.format(len(imagePaths)))
    # print('trainImages {}'.format(len(trainImages)))
    # print('trainMasks {}'.format(len(trainMasks)))

    '''
    look through the pixels in the mask (0/1/2)
    grad_2d scale
    try skipping? / size alignment
    pre-trained model (etc. resnet, heart from ct?)
    '''

    # x = torch.randn(1,1,32,32)
    # _e = Encoder()
    # __e = _e(x)
    # y = torch.randn(1, 32, 4, 4)
    # _d = Decoder()
    # _d(__e[::-1][0], __e[::-1][1:])
    # print(torch.nn.Conv2d(16,3,1))