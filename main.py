import os
from data_loader import *
from prostate_seg import *
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss,BCEWithLogitsLoss, L1Loss
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import config
import utils
import logging

if __name__ == '__main__':

    model_name = config.VERSION + '.pth'
    loss_plot_name = config.VERSION + '.png'
    plot_folder_train = config.PLOT_FOLDER + config.VERSION + '/train/'
    plot_folder_test = config.PLOT_FOLDER + config.VERSION + '/test/'
    if not os.path.isdir(config.MODEL_FOLDER):
        os.makedirs(config.MODEL_FOLDER)
    if not os.path.isdir(config.LOSS_FOLDER):
        os.makedirs(config.LOSS_FOLDER)
    complete_model_path = config.MODEL_FOLDER + model_name
    complete_loss_plot_path = config.LOSS_FOLDER + loss_plot_name
    loader = path_loader()
    binary = True
    loader.get_path(config.DATASET_MAIN_BRUNCH)
    (im_path, mask_path) = loader.load_path()

    DEVICE = config.DEVICE
    # define threshold to filter weak predictions
    THRESHOLD = config.THRESHOLD

    split = train_test_split(im_path,mask_path,test_size=config.TEST_SPLIT,random_state = config.RAND_STATE)
    '''
    total number of images for dicom and mask: 4104
    # of training batch: 2749
    get images one per patient for training 
    '''
    train_paths_Images,test_paths_Images,train_paths_Masks,test_paths_Masks = split[0],split[1],split[2],split[3]

    train = dataset_preperation(train_paths_Images, train_paths_Masks, True)
    train_dataset = train.read_preprocess_dicom_mask('both', binary, True)

    test = dataset_preperation(test_paths_Images, test_paths_Masks, False)
    test_dataset = test.read_preprocess_dicom_mask('both', binary, True)

    train_imgs, train_mask = train_dataset[0], train_dataset[1]
    test_imgs, test_mask = test_dataset[0], test_dataset[1]

    transformation = True
    _train = FetchImage(train_imgs, train_mask, transformation, binary)
    _test = FetchImage(test_imgs, test_mask, transformation, binary)
    train_Loader = DataLoader(_train,shuffle = True, batch_size = config.BATCH_SIZE,
                              pin_memory = config.PIN_MEMORY,num_workers = 4)
    test_Loader = DataLoader(_test, shuffle=True, batch_size=config.BATCH_SIZE,
                              pin_memory=config.PIN_MEMORY,num_workers = 4)
    encChannels = [1, 16, 32, 64, 128]
    decChannels = [128, 64, 32, 16]
    unet = U_net(encChannels = encChannels,
                 decChannels = decChannels, binary= binary).to(config.DEVICE)

    BEC_Loss = BCEWithLogitsLoss().to(DEVICE)
    # L1_Loss = L1Loss(reduce = 'mean').to(DEVICE)
    # loss_func = config.Binary_DiceLoss()
    opt = Adam(unet.parameters(), lr=config.INIT_LR,weight_decay = config.WEIGHT_DECAY)

    trainSteps = len(train_paths_Images) // config.BATCH_SIZE
    testSteps = len(test_paths_Images) // config.BATCH_SIZE

    H = {"train_loss": [], "test_loss": []}
    print()
    print('[INFO] training the network...')
    startTime = time.time()

    # for e in range(config.NUM_EPOCHS):
    #     for (i,(x,y)) in enumerate(train_Loader):
    #         y_numpy = y[0][0].cpu().detach().numpy()
    #         plt.imshow(y_numpy,cmap = 'gray')
    #         plt.show()
    for e in tqdm(range(config.NUM_EPOCHS)):
        torch.cuda.empty_cache()
        #unet.train()

        totalTrainLoss, totalTestLoss = 0, 0

        for (i,(x,y)) in enumerate(train_Loader):
            (x,y) = (x.to(config.DEVICE), y.to(config.DEVICE))#.long())

            pred = unet(x)
            loss = BEC_Loss(pred,y.float())#BEC_Loss(pred,y.float())# + L1_Loss(pred,y.float()) * 10

            opt.zero_grad()
            #loss.requires_grad = True
            loss.backward()
            opt.step()

            totalTrainLoss += BEC_Loss(pred,y.float()) #BEC_Loss(pred,y.float()) + L1_Loss(pred,y.float()) * 10
        if (e+1) % 20 == 0:
            print(f'Running on epoch {e+1}...')
            print(' Plotting Figures for training ...')
            utils.plot_figure(x, pred, y, e+1, plot_folder_train,True)
        ## switch off autograd
        with torch.no_grad():

            ## set the model in evaluation mode
            unet.eval()

            ## loop over the validation set
            for (x,y) in test_Loader:
                (x,y) = (x.to(config.DEVICE), y.to(config.DEVICE))#.long())

                pred = unet(x)

                totalTestLoss += BEC_Loss(pred,y.float())#BEC_Loss(pred,y.float())# + L1_Loss(pred,y.float()) * 10
            if (e + 1) % 20 == 0:
                print(' Plotting Figures for testing ...')
                utils.plot_figure(x, pred, y, e + 1, plot_folder_test, True)
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


    utils.plot_loss(H,complete_loss_plot_path)
    torch.save(unet, complete_model_path)




    '''
    look through the pixels in the mask (0/1/2)
    grad_2d scale
    try skipping? / size alignment
    pre-trained model (etc. resnet, heart from ct?)
    '''
