# -*- coding: utf-8 -*-
"""
Created on Jan 10 2024

@author: Marlene Leimeister

This script is implemented to tune the model, which is defined in
unet_architecture.py. Therefore, the library RayTune is used.
Parameters for tuning are to be set in the hyperparams_tuning.json file.
The hyperparameter space can be configured in the config dictionary (line 230)
and includes the space for the learning rate, batch size and model size.
The resulting models are stored in the folder ray_results, the configureation
of the best model is stored in a .txt file along with the resulting scores
for validation and test data.

To run this script, write in your terminal:
python unet_tuning.py hyperparams_tuning.json
"""

import os
import numpy as np
import argparse
import tempfile
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from unet_architecture import UNet_small, UNet_medium, UNet_large
from unet_architecture import DataLoaderSegmentation
from unet_training import check_accuracy, train_fn, val_fn

import albumentations as A
from albumentations.pytorch import ToTensorV2

from ray import train, tune
from ray.tune.schedulers import ASHAScheduler


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
    path = os.path.abspath(__file__).replace('unet_tuning.py', '')
    TRAIN_IMG_DIR = path + params['TRAIN_IMG_DIR']
    TRAIN_MASK_DIR = path + params['TRAIN_MASK_DIR']
    TEST_IMG_DIR = path + params['TEST_IMG_DIR']
    TEST_MASK_DIR = path + params['TEST_MASK_DIR']

    ### Ratio for training-validation split
    RATIO_TUNING_DATA = params['RATIO_TUNING_DATA']

    ### Tuning Parameters: Maximum number of epochs, number of trials ###
    MAX_NUM_EPOCHS = params['MAX_NUM_EPOCHS']
    NUM_TRIALS = params["NUM_TRIALS"]

    ### Index for the storage-name ###
    INDEX = params["INDEX"]

    ### Misc. settings ###
    PIN_MEMORY = params['PIN_MEMORY']
    NUM_WORKERS = params['NUM_WORKERS']
    SEED = params['SEED']


# %% Functions

def stop_nan(trial_id: str, result: dict):
    """
    Function to stop a trial if the loss is nan.

    Parameters
    ----------
    result : dictionary
        Dictionary with information about the current trial.

    Returns
    -------
    loss_condition : bool
        Boolean parameter. True, if the loss is nan, else false.

    """

    loss_condition = np.isnan(result.get("loss"))
    return loss_condition


def training(config):
    """
    Function for training a model. The parameters stored in
    the directory ray_results (checkpoint.pth.tar).

    Parameters
    ----------
    config : dictionary
        Dictionary with the parameter space of the hyperparameters to be tuned.

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


    MODEL = config['unet_size']
    model = MODEL(1, 1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    if train.get_checkpoint():
        loaded_checkpoint = train.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, 'checkpoint.pth.tar'))
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

    train_ds = DataLoaderSegmentation(img_dir=TRAIN_IMG_DIR,
                                        mask_dir=TRAIN_MASK_DIR,
                                        transform=train_transform)

    # split and load data -> training and validation

    val_abs = int(len(train_ds) * RATIO_TUNING_DATA)
    train_subset, val_subset = random_split(
                        train_ds, [val_abs, len(train_ds) - val_abs])

    train_loader = DataLoader(train_subset,
                            batch_size=int(config['batch_size']),
                            num_workers=NUM_WORKERS,
                            pin_memory=PIN_MEMORY,
                            shuffle=True,
                            )

    val_loader = DataLoader(val_subset,
                            batch_size=int(config['batch_size']),
                            num_workers=NUM_WORKERS,
                            pin_memory=PIN_MEMORY,
                            shuffle=True,
                            )

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(MAX_NUM_EPOCHS):


        loss_training = train_fn(train_loader,
                                model,
                                optimizer,
                                loss_fn,
                                scaler,
                                DEVICE,
                                display_loop=False)


        loss_validation = val_fn(val_loader,
                                model,
                                loss_fn,
                                DEVICE)

        scores_validation = check_accuracy(val_loader,
                                            model,
                                            device=DEVICE)

        print('loss-train:', '{:.2f}'.format(loss_training),'| loss-val:',
              '{:.2f}'.format(loss_validation), '|| acc-val:', '{:.2f}'.format(scores_validation[0]),
              '| iou-val:','{:.2f}'.format(scores_validation[1]))

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, 'checkpoint.pth.tar')
            torch.save((model.state_dict(), optimizer.state_dict()), path)
            checkpoint = train.Checkpoint.from_directory(temp_checkpoint_dir)
            train.report({'loss': loss_validation,
                          'acc': round(scores_validation[0],3),
                          'iou': round(scores_validation[1],3),
                          'p': round(scores_validation[2],3),
                          'r': round(scores_validation[3],3),
                          'f1': round(scores_validation[4],3)},
                          checkpoint = checkpoint,)


# %% Functions

def main():
    """
    Function for tuning the model as defined in the hyperparams.json file.

    Returns
    -------

    best_result : dictionary
        Dictionary with information about the best result. These are
        the configuration (best hyperparameters), metrics and the path of
        the trial's checkpoint.

    scores_test : list
        List with metrics from the best model performing on test data.
        Structure: [acc,iou,p,r,f1]

    """

    # Define the space for the hyperparameters to be tuned
    config = {'unet_size': tune.choice([UNet_small, UNet_medium, UNet_large]),
              'lr': tune.loguniform(1e-5, 1e-3),
              'batch_size': tune.choice([2,4,8,16])}

    scheduler = ASHAScheduler(time_attr='training_iteration',  # training result attribute for time
                                metric='loss',  # parameter for optimisation
                                mode='min',  # minimise the metric
                                max_t=MAX_NUM_EPOCHS,  # max time (here number of epochs)
                                grace_period=10,  # min time (here number of epochs)
                                reduction_factor=3,  # halving rate and amount, promote the 1/eta best trials
                                brackets=2)  # number of brackets for halving rate


    tuner = tune.Tuner(tune.with_resources(
                        tune.with_parameters(training),
                        resources={"cpu": 8, "gpu": 1}),
                        run_config=train.RunConfig(
                                name=INDEX,
                                stop=stop_nan),
                        tune_config=tune.TuneConfig(
                                scheduler=scheduler,
                                num_samples=NUM_TRIALS),
                        param_space=config)

    results = tuner.fit()

    best_result = results.get_best_result("loss", "min", scope='all', filter_nan_and_inf=True)

    print("Best trial config: {}".format(best_result.config))

    # test best_model with unseen test_data

    best_trained_model = best_result.config['unet_size'](1, 1).to(DEVICE)  # choose best model
    checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pth.tar")  # load model checkpoint
    model_state, optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)

    test_transform = A.Compose(
        [
            A.Normalize(mean=0.133, std=0.133, max_pixel_value=255.0, always_apply=(True)),
            ToTensorV2(),
        ],
    )

    test_ds = DataLoaderSegmentation(img_dir=TEST_IMG_DIR,
                            mask_dir=TEST_MASK_DIR,
                            transform=test_transform)

    test_loader = DataLoader(test_ds,
                        batch_size=int(best_result.config['batch_size']),
                        num_workers=NUM_WORKERS,
                        pin_memory=PIN_MEMORY,
                        shuffle=False,
                        )

    scores_test = check_accuracy(test_loader,
                                 best_trained_model,
                                 device=DEVICE)

    print('Test scores of the best trial:\n[acc,iou,p,r,f1] = ', scores_test)

    return best_result, scores_test


# %%
if __name__ == "__main__":

    start_time = time.perf_counter()
    print(f"The device is {DEVICE}!")
    best_result, scores_test = main()
    end_time = time.perf_counter()
    duration = end_time-start_time

    # save properties in file
    file = open(f"properties_tuning_{INDEX}.txt",'w')
    file.write(f"Tuning took a total of {duration} seconds!")
    file.write("\nBest trial config: {}".format(best_result.config))
    file.write("\nBest trial checkpoint_path: {}".format(best_result.checkpoint))
    file.write(f"\nBest trial test results: [acc,iou, p, r, f1] = {scores_test}")

    print(f"Tuning took a total of {duration} secconds!")

# %%
