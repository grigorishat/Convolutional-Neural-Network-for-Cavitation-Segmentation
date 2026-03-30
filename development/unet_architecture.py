"""
Created on Nov 29 2021

@author: Marlene Leimeister and Jonas Bergner

This script contains the architecture of the neural net called U-Net.
It is implemented in three different sizes (small, medium and large),
which differ in the number of contraction and expansion blocks.
Also 'DataLoaderSegmentation', a class for loading in the images and
masks in batches, is implemented here.

The implementation of the U-Nets is inspired by the U-Net in the GitHub
project 'Machine-Learning -Collection' by Aladdin Persson:
https://github.com/aladdinpersson/Machine-Learning-Collection

"""

import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data


#%% Define the U-Net models


class DoubleConv(nn.Module):
    """
    Class for Double Convolution [input] -> Conv -> ReLu -> Conv
                                                -> ReLu (CB Block).

    Parameters
    ----------
    in_channels : int
        Number of input planes.
    out_channels : int
        Number of output planes, this number directly defines the number of
        convolutions applied.
    mid_channels : int, deafault None
        If not None, the number of planes after the first convolution are
        defined(in this case, this most be lower the out_channels).

    """


    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Class for Downsampling [Input] -> MaxPooling -> Conv > ReLu -> Conv
                                                            -> ReLu,
    halfs [input] in both width and height, will increase planes
    to out_channels.

    Parameters
    ----------
    in_channels : int
        Number of input planes.
    out_channels : int
        Number of output planes, this number directly defines the number
        convolutions applied.

    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2),
                                          DoubleConv(in_channels, out_channels))
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Class for Upscaling [Input] -> Upscaling -> Conv > ReLu -> Conv -> ReLu,
    doubles width and heigt of the [inpout], will reduce planes
    to out_channels.

    Parameters
    ----------
    in_channels : int
        Number of input planes.
    out_channels : int
        Number of output planes, this number directly defines the number
        convolutions applied.
    bilinear: boolean
        If true, bilinear interpolation (upsampling) is applied in order to
        increase [input] size.

    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Class for Upscaling [Input] -> Upscaling -> Conv > ReLu -> Conv -> ReLu,
    doubles width and heigt of the [input], will reduce planes to out_channels.

    Parameters
    ----------
    in_channels : int
        Number of input planes.
    out_channels : int
        Number of output planes, this should be equal to number of classes.
        For foreground - background segmentation this is equal to 1
        (only 1 output plane required).

    """

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet_medium(nn.Module):
    """
    Class of the U-Net Model, this is inspired by the
    Paper (https://arxiv.org/abs/1505.04597).
    Medium refers to the number of layers compared to the
    other Unet Models defined in this script.

    Parameters
    ----------
    in_channels : int, default 1 (Gray-scale image as input)
        Number of input planes.
    n_classes : int, default 1
        Number of classes, currently 1 (Foreground-pixel).
    bilinear: boolean.
        If true, bilinear interpolation is applied in order to
        increase [input] size.

    """

    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_medium, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        factor = 2 if bilinear else 1
        self.down4 = Down(128, 256 // factor)
        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.up3 = Up(64, 32 // factor, bilinear)
        self.up4 = Up(32, 16, bilinear)
        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class UNet_large(nn.Module):
    """
    Class of the U-Net Model, this is inspired by the
    Paper (https://arxiv.org/abs/1505.04597).
    Large refers to the number of layers compared to the other
    Unet Models defined in this script.

    Parameters
    ----------
    in_channels : int, default 1 (Gray-scale image as input)
        Number of input planes.
    n_classes : int, default 1
        Number of classes, currently 1 (Foreground-pixel).
    bilinear: boolean.
        If true, bilinear interpolation is applied in order to
        increase [input] size.

    """

    def __init__(self, n_channels = 1, n_classes = 1, bilinear=True):
        super(UNet_large, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        self.down4 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down5 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32 // factor, bilinear)
        self.up5 = Up(32, 16, bilinear)
        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        logits = self.outc(x)
        return logits

class UNet_small(nn.Module):
    """
    Class of the U-Net Model, this is inspired by the
    Paper (https://arxiv.org/abs/1505.04597).
    Small refers to the number of layers compared to the other
    Unet Models defined in this script.

    Parameters
    ----------
    in_channels : int, default 1 (Gray-scale image as input)
        Number of input planes.
    n_classes : int, default 1
        Number of classes, currently 1 (Foreground-pixel).
    bilinear: boolean.
        If true, bilinear interpolation is applied in order to
        increase [input] size.

    """

    def __init__(self, n_channels = 1, n_classes = 1, bilinear=True):
        super(UNet_small, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        factor = 2 if bilinear else 1
        self.down3 = Down(64, 128 // factor)

        self.up1 = Up(128, 64 // factor, bilinear)
        self.up2 = Up(64, 32 // factor, bilinear)
        self.up3 = Up(32, 16, bilinear)
        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)

        return logits

# %% Define data loading

class DataLoaderSegmentation(data.Dataset):
    """
    Class for loading in the images and masks for training in batches.

    Parameters
    ----------
    img_dir : string
        Directory of images.
    mask_dir : string,
        Directory of masks.
    transform: list, default None
        List of transformation functions applied to images

    """

    def __init__(self, img_dir, mask_dir, transform=None):
        super(DataLoaderSegmentation, self).__init__()
        self.image_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(img_dir)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        image = np.array(Image.open(img_path), dtype=np.float32)
        mask = np.array(Image.open(mask_path), dtype=np.float32)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        return image, mask

    def __len__(self):
        return len(self.images)


#%%
