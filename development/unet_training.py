# -*- coding: utf-8 -*-
"""
Created on Nov 29 2021

@author: Marlene Leimeister and Jonas Bergner

With using this script an U-Net model, which is defined in unet_architecture,
can be trained. The loss and the scores accuracy, intersection over union,
precision, recall and F1 are stored for each epoch in a json file.
The resulting trained model is stored in a .pth.tar file.
The hyperparameters for training and paths to training and test datasets
are to be set in the hyperparams_training.json file.

To run this script, write in your terminal:
python unet_training.py hyperparams_training.json
"""

import numpy as np
from tqdm import tqdm
import argparse
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from unet_architecture import UNet_small, UNet_medium, UNet_large
from unet_architecture import DataLoaderSegmentation

from torchmetrics.classification import BinaryAccuracy as BA
from torchmetrics.classification import BinaryJaccardIndex as BJI
from torchmetrics.classification import BinaryPrecision as BP
from torchmetrics.classification import BinaryRecall as BR
from torchmetrics.classification import BinaryF1Score as BF1

import albumentations as A
from albumentations.pytorch import ToTensorV2


# %% Settings

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=argparse.FileType("r"),
                        help="json file including hyperparameters")
    args = parser.parse_args()

    with args.filename as file:
        params = json.loads(file.read())

    ### Directories for training and test data ###
    TRAIN_IMG_DIR = params['TRAIN_IMG_DIR']
    TRAIN_MASK_DIR = params['TRAIN_MASK_DIR']
    TEST_IMG_DIR = params['TEST_IMG_DIR']
    TEST_MASK_DIR = params['TEST_MASK_DIR']

    ### Ratio for validation-test split
    RATIO_VAL_DATA = params['RATIO_VAL_DATA']

    ### Hyperparameters: learning rate, batch size and number of epochs ###
    LEARNING_RATE = params['LEARNING_RATE']
    BATCH_SIZE = params['BATCH_SIZE']
    NUM_EPOCHS = params['NUM_EPOCHS']

    ### Model name for storing model after training ###
    MODEL_NAME = params['MODEL_NAME']
    INDEX = params["INDEX"]
    checkpoint_name = f"{MODEL_NAME}_{INDEX}.pth.tar"

    ### Misc. settings ###
    PIN_MEMORY = params['PIN_MEMORY']
    LOAD_MODEL = params['LOAD_MODEL']
    NUM_WORKERS = params['NUM_WORKERS']
    SEED = params['SEED']

    DISPLAY_LOOP = True

else:  # for image analysis: checkpoint name needs to be chosen
    checkpoint_name = "UNet_medium_e150_bs16_lr0004.pth.tar"


# %% Functions


def load_checkpoint(checkpoint, model):
    """
    Function for loading the state of a neural network stored under checkpoint.
    This will for example change the weights of the NN to the ones stored in
    the checkpoint.

    Parameters
    ----------
    checkpoint : string
        String of the filename which was created using save_checkpoint
        function.
    model : Torch.Model
        Torch model where the information stored in checkpoint is
        to be loaded in.

    """

    print("=> Loading U-Net Model")
    model.load_state_dict(checkpoint["state_dict"])
    print("U-Net Model initialised")


def check_accuracy(loader,
                   model,
                   device):
    """
    Function for calculating metrics to evaluate the model.
    The metrics are: accuracy, intersection over union, precision, recall, F1

    Parameters
    ----------
    loader : object
        Object of the DataLoader class.
    model : object
        Object of the model class.
    device : string
        Either "CUDA" for running programm on GPU or "CPU" for running
        programm on processor.

    Returns
    -------
    scores : list
        List including float values of the scores.

    """

    model.eval()

    #  https://torchmetrics.readthedocs.io/en/stable/classification/accuracy.html
    ACC_metric = BA(threshold=0.5, multidim_average="global").to(device)
    IOU_metric = BJI(threshold=0.5, multidim_average="global").to(device)
    P_metric = BP(threshold=0.5, multidim_average="global").to(device)
    R_metric = BR(threshold=0.5, multidim_average="global").to(device)
    F1_metric = BF1(threshold=0.5, multidim_average="global").to(device)

    ACC = 0
    IOU = 0
    P = 0
    R = 0
    F1 = 0

    with torch.no_grad():
        for x, y in loader:  # x=image, y=target
            x = x.to(device)
            y = y.to(device).unsqueeze(1).int()
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float().int()

            ACC += ACC_metric(preds, y)
            IOU += IOU_metric(preds, y)
            P += P_metric(preds, y)
            R += R_metric(preds, y)
            F1 += F1_metric(preds, y)

    ACC = ACC/len(loader)
    IOU = IOU/len(loader)
    P = P/len(loader)
    R = R/len(loader)
    F1 = F1/len(loader)

    scores = [ACC.item(), IOU.item(), P.item(), R.item(), F1.item()]

    model.train()

    return scores

def train_fn(loader,
             model,
             optimizer,
             loss_fn,
             scaler,
             device,
             display_loop=False):
    """
    Function for training the model.

    Parameters
    ----------
    loader : object of get_loders function
        Used for loading in the images during training in a batch like manner.
    model : object of torch.model class
        Model used for training.
    optimiser : object of torch.optim
        Object defining the optimiser. In this instance the ADAM-algorithm is used.
    loss_fn : object
        Object defining the loss-function used for minimizing.
    scaler : object
        Object used for weights adjustment of the model.
    device : string
        String defining the training device, either 'CUDA' or 'CPU'.

    Returns
    -------
    loss_over_epoch : float
        Float number of the loss-function value for one epoch.

    """

    if display_loop:
        loop = tqdm(loader)

    loss_over_epoch = 0

    for batch_idc, (Data, targets) in enumerate(loop) if display_loop else enumerate(loader):
        Data = Data.to(device=device)
        targets = targets.float().unsqueeze(1).to(device=device)

        # forward-pass of the model
        with torch.cuda.amp.autocast():
            predictions = model(Data)
            loss = loss_fn(predictions, targets)

        # backward-pass of the model
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_over_epoch += loss.item()

        if display_loop:
            loop.set_postfix(loss=loss.item())

    return loss_over_epoch


def val_fn(loader,
             model,
             loss_fn,
             device):
    """
    Function for calculating the validation loss of the model.

    Parameters
    ----------
    loader : object of get_loders function
        Used for loading in the images during validation in a batch
        like manner.
    model : object of torch.model class
        Model used for training.
    loss_fn : object
        Object defining the loss-function used for minimizing.
    device : string
        String defining the training device, either 'CIDA' or 'CPU'.

    Returns
    -------
    loss_over_epoch : float
        Float number of the loss-function value for one epoch.

    """

    loss_over_epoch = 0

    for batch_idc, (Data, targets) in enumerate(loader):
        Data = Data.to(device=device)
        targets = targets.float().unsqueeze(1).to(device=device)

        # forward-pass of the model
        with torch.no_grad():
            predictions = model(Data)
            loss = loss_fn(predictions, targets)
        loss_over_epoch += loss.item()

    return loss_over_epoch


def main():

    """
    Function for training the chosen model as defined in the
    hyperparams.json file.

    Returns
    -------
    report_dic : dictionary
        Dictionary with scores and the loss (float numbers) for training,
        validation and scores for the test data.
        The scores for each epoch are stored in lists in the dictionary.
        Structure of the dictionary:
        report_dic =
            {'training': {'ACC':[],'IOU':[],'P':[],'R':[],'F1':[],'loss':[]},
            'validation': {'ACC':[],'IOU':[],'P':[],'R':[],'F1':[],'loss':[]},
            'test': {'ACC':[],'IOU':[],'P':[],'R':[],'F1':[]}}

    """

    # Torch RNG
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # Python RNG
    np.random.seed(SEED)

    # transformation functions for training - Image Augmentation
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Normalize(mean=0.133, std=0.133, max_pixel_value=255.0, always_apply=(True)),
            ToTensorV2(),
        ],
    )

    # transformation functions for validation and testing
    test_transform = A.Compose(
        [
            A.Normalize(mean=0.133, std=0.133, max_pixel_value=255.0, always_apply=(True)),
            ToTensorV2(),
        ],
    )

    train_ds = DataLoaderSegmentation(img_dir=TRAIN_IMG_DIR,
                            mask_dir=TRAIN_MASK_DIR,
                            transform=train_transform)

    test_ds = DataLoaderSegmentation(img_dir=TEST_IMG_DIR,
                            mask_dir=TEST_MASK_DIR,
                            transform=test_transform)

    # split and load data -> validation, test

    val_abs = int(len(test_ds) * RATIO_VAL_DATA)
    val_subset, test_subset = random_split(
                        test_ds, [val_abs, len(test_ds) - val_abs])

    train_loader = DataLoader(train_ds,
                            batch_size=BATCH_SIZE,
                            num_workers=NUM_WORKERS,
                            pin_memory=PIN_MEMORY,
                            shuffle=True,
                            )

    val_loader = DataLoader(val_subset,
                            batch_size=BATCH_SIZE,
                            num_workers=NUM_WORKERS,
                            pin_memory=PIN_MEMORY,
                            shuffle=True,)

    MODELS = {"UNet_small": UNet_small, "UNet_medium": UNet_medium, "UNet_large": UNet_large} # uses model set in json file
    try:
        MODEL = MODELS[MODEL_NAME]
    except KeyError:
        raise ValueError('Invalid MODEL_NAME! Use UNet_small, UNet_medium or UNet_large.')


    model = MODEL(1, 1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    report_dic = {'training': {'ACC':[],'IOU':[],'P':[],'R':[],'F1':[],'loss':[]},
                  'validation': {'ACC':[],'IOU':[],'P':[],'R':[],'F1':[],'loss':[]},
                  'test': {'ACC':[],'IOU':[],'P':[],'R':[],'F1':[]}}

    for epoch in range(NUM_EPOCHS):

        loss_training = train_fn(train_loader,
                                        model,
                                        optimizer,
                                        loss_fn,
                                        scaler,
                                        DEVICE,
                                        DISPLAY_LOOP)

        loss_validation = val_fn(val_loader,
                                        model,
                                        loss_fn,
                                        DEVICE)

        # save model
        checkpoint = {"state_dict": model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(),}
        print("-> Saving checkpoint")
        torch.save(checkpoint, checkpoint_name)

        scores_training = check_accuracy(train_loader,
                                            model,
                                            device=DEVICE)

        scores_validation = check_accuracy(val_loader,
                                            model,
                                            device=DEVICE)

        i = 0
        for key in report_dic['training'].keys():
            if i == 5:
                report_dic['training'][key].append(loss_training)
                report_dic['validation'][key].append(loss_validation)
            else:
                report_dic['training'][key].append(scores_training[i])
                report_dic['validation'][key].append(scores_validation[i])
            i+=1

        if (epoch+1) % 20 == 0:
            print(f"Accuracy train: {scores_training[0]}, validation: {scores_validation[0]}")
            print(f"IOU score train: {scores_training[1]}, validation: {scores_validation[1]}")
            print(f"Loss train: {loss_training}, validation: {loss_validation}")

    # test model on unseen test data
    test_loader = DataLoader(test_subset,
                            batch_size=BATCH_SIZE,
                            num_workers=NUM_WORKERS,
                            pin_memory=PIN_MEMORY,
                            shuffle=True,)

    scores_test = check_accuracy(test_loader,
                                 model,
                                 device=DEVICE)

    print('test scores:\n[acc,iou,p,r,f1] = ', scores_test)

    i = 0
    for key in report_dic['test'].keys():
        report_dic['test'][key].append(scores_test[i])
        i+=1

    return report_dic


# %%
if __name__ == "__main__":

    start_time = time.perf_counter()
    report_dic = main()
    end_time = time.perf_counter()

    duration = end_time-start_time

    print(f"Training the {MODEL_NAME} for {NUM_EPOCHS} epochs took a total of {duration} seconds!")
    print("Write resulting scores and loss to a json file!")
    try:
        with open(f"properties_{MODEL_NAME}_{INDEX}.json", 'x') as json_file:
            json.dump(report_dic, json_file, indent=4)
    except:
        print('Properties file exists already. NEW file is generated!')
        with open(f"NEW_properties_{MODEL_NAME}_{INDEX}.json", 'a') as json_file:
            json.dump(report_dic, json_file, indent=4)

# %%
