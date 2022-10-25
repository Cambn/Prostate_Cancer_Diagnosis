import os
from data_loader import *
from prostate_seg import *
import torch
# import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss,BCEWithLogitsLoss, L1Loss
import time
from tqdm import tqdm
from torchvision import transforms
from sklearn.model_selection import train_test_split
import config

import logging

if __name__ == '__main__':
    loader = path_loader()

    loader.get_path(config.DATASET_MAIN_BRUNCH)
    (im_path, mask_path) = loader.load_path()

    DEVICE = config.DEVICE
    # define threshold to filter weak predictions
    THRESHOLD = config.THRESHOLD

    split = train_test_split(im_path,mask_path,test_size=config.TEST_SPLIT,random_state = config.RAND_STATE)
    '''
    total number of images for dicom and mask: 4104
    # of training batch: 2749
    '''
    train_paths_Images,test_paths_Images,train_paths_Masks,test_paths_Masks = split[0],split[1],split[2],split[3]
    train = dataset_preperation(train_paths_Images, train_paths_Masks, True)
    train_dataset = train.read_preprocess_dicom_mask(True)
    test = dataset_preperation(test_paths_Images, test_paths_Masks, False)
    test_dataset = test.read_preprocess_dicom_mask(True)
    train_imgs, train_mask = train_dataset[0], train_dataset[1]
    test_imgs, test_mask = test_dataset[0], test_dataset[1]
    transformation = transforms.Compose([transforms.ToPILImage(),
                                         transforms.ToTensor()])
    _train = FetchImage(train_imgs, train_mask, transformation)
    _test = FetchImage(test_imgs, test_mask, transformation)
    train_Loader = DataLoader(_train,shuffle = True, batch_size = config.BATCH_SIZE,
                              pin_memory = config.PIN_MEMORY,num_workers = 4)
    test_Loader = DataLoader(_test, shuffle=True, batch_size=config.BATCH_SIZE,
                              pin_memory=config.PIN_MEMORY,num_workers = 4)

    unet = U_net().to(config.DEVICE)
    BEC_Loss = BCEWithLogitsLoss(reduce = 'mean').to(DEVICE)
    L1_Loss = L1Loss(reduce = 'mean').to(DEVICE)
    opt = Adam(unet.parameters(), lr=config.INIT_LR)

    trainSteps = len(train_paths_Images) // config.BATCH_SIZE
    testSteps = len(test_paths_Images) // config.BATCH_SIZE

    H = {"train_loss": [], "test_loss": []}
    print()
    print('[INFO] training the network...')
    startTime = time.time()

    for e in tqdm(range(config.NUM_EPOCHS)):
        torch.cuda.empty_cache()
        unet.train()

        totalTrainLoss, totalTestLoss = 0, 0

        for (i,(x,y)) in enumerate(train_Loader):

            (x,y) = (x.to(config.DEVICE), y.to(config.DEVICE))#.long())

            pred = unet(x)
            loss = BEC_Loss(y,pred) + L1_Loss(y,pred) * 10

            opt.zero_grad()
            #loss.requires_grad = True
            loss.backward()
            opt.step()

            totalTrainLoss += BEC_Loss(y,pred) + L1_Loss(y,pred) * 10
        if (e + 1) % 51 == 0:
            print(f'Running on epoch {e + 1}...')
            config.plot_figure(x, pred, y)
        ## switch off autograd
        with torch.no_grad():

            ## set the model in evaluation mode
            unet.eval()

            ## loop over the validation set
            for (x,y) in test_Loader:
                (x,y) = (x.to(config.DEVICE), y.to(config.DEVICE))#.long())
                pred = unet(x)
                totalTestLoss += BEC_Loss(y,pred) + L1_Loss(y,pred) * 10

        avgTrainLoss = totalTrainLoss / trainSteps
        avgTestLoss = totalTestLoss / testSteps


        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["test_loss"].append(avgTestLoss.cpu().detach().numpy())
        print()
        print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
        print("Train loss: {:.6f}, Test loss: {:.4f}".format(avgTrainLoss, avgTestLoss))
        torch.cuda.empty_cache()
    # display the total time needed to perform the training
    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))

    if not os.path.isdir(config.LOSS_FOLDER):
        os.makedirs(config.LOSS_FOLDER)
    if not os.path.isdir(config.MODEL_FOLDER):
        os.makedirs(config.MODEL_FOLDER)

    model_name = 'unet_10_24_v4.pth'
    loss_plot_name = 'unet_10_24_v4.png'
    complete_model_path = config.MODEL_FOLDER + model_name
    complete_loss_plot_path = config.LOSS_FOLDER + loss_plot_name
    config.plot_loss(H,complete_loss_plot_path)
    torch.save(unet, complete_model_path)




    '''
    look through the pixels in the mask (0/1/2)
    grad_2d scale
    try skipping? / size alignment
    pre-trained model (etc. resnet, heart from ct?)
    '''
