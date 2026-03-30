# -*- coding: utf-8 -*-
"""
Created on Dec 07 2022

@author: Marlene Leimeisterand Jonas Bergner

This script contains functions for image pre-processing.
It includes image-blackening, rotating of images and
the calculating of the pixel resolution.

"""

import pandas as pd
import cv2
import math


# %% Functions

def mask_apply(image, bounds):
    """
    Function applies a blackening mask to input image.

    Parameters
    ----------
    image : Array of uint8
        Input image where mask is to be applied on.
    bounds : list
        Contains the two y-coordinates setting the boundary above and below
        which the mask is applied to.

    Returns
    -------
    image : array of uint8
        Output image with the blackened region above and below values
        in bounds.

    """

    for y in range(0, bounds[0]+1):
        for x in range(len(image[0])):
            image[y][x] = 0
    for y in range(bounds[1], len(image)):
        for x in range(len(image[0])):
            image[y][x] = 0

    return image


def read_tif(file, apply_mask=True, Resize=True, rotate180=False, bounds=None):
    """
    Function for reading and preprocessing an uint 16 TIF image and preprocess
    it. The preprocessing can include blackening, resizing and a rotation.

    Parameters
    ----------
    file : string
        String containing the filename of the image.
    apply_mask : boolean, optional
        If True, enables the mask_apply function. The default is True.
    Resize : boolean, optional
        Resizes the image so that it can be used for segmentation with U-Net.
        The default is True.
    rotate180 : boolean, optional
        Rotate the image in 180°. The default is False.
    bounds : list, optional
        Needed when apply_mask == True. The list contains the two
        y-coordinates setting the boundary above and below which
        the mask is applied to. The default is None.

    Returns
    -------
    image : array of uint8
        Preprocessed image.

    """

    image = cv2.imread(file, -1)*(2**0/2**8)
    image = image.astype('uint8')
    if Resize:
        a = 2**6
        if len(image[0]) % a != 0 or len(image)%a != 0:
            cuth = (len(image[0]) % a)/2
            if cuth != int(cuth):
                cuth1 = math.floor(cuth)
                cuth2 = cuth1+1
            else:
                cuth1 = int(cuth)
                cuth2 = cuth1
            cutv = (len(image) % a)/2
            if cutv != round(cutv):
                cutv1 = math.floor(cutv)
                cutv2 = cutv1+1
            else:
                cutv1 = int(cutv)
                cutv2 = cutv1
            image = image[cutv1:len(image)-cutv2, cuth1:len(image[0])-cuth2]
    if rotate180:
        image = cv2.rotate(image, cv2.ROTATE_180)
    if apply_mask:
        image = mask_apply(image, bounds)

    return image


def calc_pixel_size(edges, profile_df, angle):
    """
    Function for calculating the pixelsize (mm/pixel) based on the identified
    edges of the hydrofoil.

    Parameters
    ----------
    edges : dictionary
        Dictonary including the edges of the hydrofoil.
    profile_df : pandas dataframe
        DataFrame including the dimensions of the hydrofoil.
    angle : integer
        Value of the tilt-angle of the hydrofoil.

    Returns
    -------
    pixel_size : float
        Value of the averaged pixel-size in [mm/pixel] and the corresponding
        standard deviation.
    pixel_df : dataframe
        DataFrame where the pixel-size is listed for each entry in edges.

    """

    pixel_lst = []
    angle_rad = angle*(math.pi/180)

    pixel_lst = [profile_df.iloc[0, 1] / (edges["lower_edge"].values[0] -
                                          edges["upper_edge"].values[0]),
                (profile_df.iloc[0, 0]*math.cos(angle_rad)) /
                (edges["trailing_edge"].values[0] - edges["leading_edge"].values[0])]

    pixel_df = pd.DataFrame({"mm/pixel": pixel_lst})
    pixel_size = (pixel_df.mean()[0], pixel_df.std()[0])

    return pixel_size, pixel_df
