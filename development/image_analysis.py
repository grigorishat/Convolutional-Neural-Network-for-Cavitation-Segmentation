# -*- coding: utf-8 -*-
"""
Created on Nov 30 2021

@author: Marlene Leimeister and Jonas Bergner

This script contains functions for image analysis.
The function 'contour_cutoff' is adapted specifically to the data
and may needs to be adjusted for your measurements.
In the notebook image_analysis.ipynb the usage of these functions
are exemplary shown.
"""

#%% Modules

import cv2
import torch
import numpy as np
import albumentations as A


#%% Functions

def segment_image(img, model, device):
    """
    Function for Segmenting in input image
    in form of an array using the UNet Model.

    Parameters
    ----------
    img : uint-8 numpy array
        Array of the input image which ought to be segmented.
    model : object
        Object created with torch.model.
    device : string
        Either 'CUDA' if cuda support available or 'CPU' if not available.
        This defines the hardware location for the segmentation process.

    Returns
    -------
    bin_img : uint-8 numpy arrayand
        Binarised image resulting from the U-Net segmentation.

    """

    norm_img = A.Compose([A.Normalize(mean=0.133, std=0.133,
                                      max_pixel_value=255.0,
                                      always_apply=(True))])(image=img)["image"]
    norm_img = torch.from_numpy(norm_img).unsqueeze(1).unsqueeze(1)
    norm_img = norm_img.transpose(0,2)

    # Run U-Net, create prediction
    with torch.no_grad():
        x = norm_img.to(device)
        with torch.cuda.amp.autocast():
            preds = torch.sigmoid(model(x))
        preds = (preds > 0.5).float()

    # Pull prediction from GPU to CPU, change into numpy-array
    torch.Tensor.ndim = property(lambda self: len(self.shape))
    preds = preds.to("cpu")
    bin_img = preds[0][0].numpy()

    return bin_img


def process_bin(bin_img):
    """
    Function which applies postprocessing steps on the binarised image
    resulting from the segmentation. The output of this function is used
    for contour identification.

    Parameters
    ----------
    bin_img :  U-int8 numpy array
        Binarised image form after U-Net segmentation.

    Returns
    -------
    bin_img_processed : Uint8 numpy array
        Returns an binarised image after processing.

    """

    kernel = np.ones((6,6),np.uint8)
    # Mophologic closing
    closing = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
    # Morphologic opening
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    opening[:,0] = 0
    opening[:,len(opening[0])-1] = 0
    opening[0,:] = 0
    opening[len(opening)-1, :] = 0

    (_, bin_img_processed) = cv2.threshold(opening, 0.5, 1, cv2.THRESH_BINARY)
    bin_img_processed = np.uint8(bin_img_processed)

    return bin_img_processed


def fix_contours(contour_lst):
    """
    Function to fix contours that intersects with itself,
    drop contours that lie within another, and ensure all
    contours are closed.

    Parameters
    ----------
    contour_lst : list
        List of numpy arrays which contain the contours identified.

    Returns
    -------
    filtered_contours : list
        List of numpy arrays which contain the filtered contours after
        a search for broken contours and its fix.

    """

    def is_contour_inside(cont1, cont2):
        """
        Check if all points of cont1 are inside the bounding box of cont2.
        """
        x_min, y_min, w, h = cv2.boundingRect(cont2)
        x_max, y_max = x_min + w, y_min + h
        for point in cont1[:, 0]:
            x, y = point
            if not (x_min <= x <= x_max and y_min <= y <= y_max):
                return False
        return True

    closed_contours = []
    for cont in contour_lst:
        if not np.array_equal(cont[0, 0], cont[-1, 0]):
            cont = np.vstack([cont, [cont[0]]])
        closed_contours.append(cont)

    filtered_contours = []
    for i, cont1 in enumerate(closed_contours):
        is_nested = False
        for j, cont2 in enumerate(closed_contours):
            if i != j and is_contour_inside(cont1, cont2):
                is_nested = True
                break
        if not is_nested:
            filtered_contours.append(cont1)

    return filtered_contours


def classify_contours(contour_lst, vert_thresh):
    """
    Classifies list of contours based on there proximity to the identified
    leading edge of the profile and the contour's length.

    Parameters
    ----------
    contour_lst : list
        List of contours.
    vert_thresh : int
        Threshold to classify a contour as sheet.

    Returns
    -------
    detachments_nocut : List
        List of contours marked as detachment.
    sheet_nocut : list
        List of contours marked as sheet.

    """
    sheet_nocut = []
    detachments_nocut = []

    for cont in contour_lst:
        if cont[:,0,0].min() < vert_thresh + 50 and cont.shape[0] > 100:
            sheet_nocut.append(cont)
        elif cont[:,0,0].min() > vert_thresh + 10 and cont.shape[0] > 100:
            detachments_nocut.append(cont)

    return detachments_nocut, sheet_nocut


def contour_cutoff(sheet_nocut, detachments_nocut,
                   bin_img_processed, bounds, threshold=0.5):
    """
    Function for applying the cut-off Algorithm and decide, whether it
    is to be used. The algorithm works in a heuristic manner and returns
    classified contours with or without the usage of the cut-off.

    Parameters
    ----------
    sheet_nocut : list
        List of sheet contours.
    detachments_nocut : list
        List of detachment contours.
    bin_img_processed : uint-8 numpy array
        Binarised and processed image.
    bounds : list
        List containing the upper and lower edge of the hydrofoil.
    threshold : float, default 0.5
        Value for cut-off criteria, e.g. 0.5 -> 50% of pixels
        between the bounds need to be classified as background, then cut.

    Returns
    -------
    sheet_cut : list
        List of sheet contours after cut-off.
    detachments_cut : list
        List of new detachment contours after cut-off.
    sheet_start : integer
        Integar value of sheet start (x-coordinate in the image).
    sheet_end : integer
        Integar value of sheet end (x-coordinate in the image).

    """

    # Ensure all sheet contours are merged into a single array
    sheet_nocut = np.concatenate(sheet_nocut) if len(sheet_nocut) > 1 else sheet_nocut[0]

    # Determine the start and end points of the sheet
    begin_end_array = np.zeros((len(bin_img_processed), 2), dtype=int)
    for point in sheet_nocut:
        y, x = int(point[0][1]), int(point[0][0])
        if begin_end_array[y][0] == 0 or begin_end_array[y][0] > x:
            begin_end_array[y][0] = x

    begin_end_array = begin_end_array[~np.all(begin_end_array == 0, axis=1)]
    sheet_start = int(begin_end_array[:, 0].mean()) if begin_end_array.size > 0 else 0

    # Identify the sheet's end by scanning horizontally from `sheet_start`
    for x in range(sheet_start+50, bin_img_processed.shape[1]):
        if bin_img_processed[bounds[0]:bounds[1], x].sum() < (bounds[1] - bounds[0]) * threshold:
            bin_img_processed[:, x:x+3] = 0
            sheet_end = x
            break

    # Detect contours and fix any broken ones
    contours, _ = cv2.findContours(bin_img_processed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    contours = fix_contours(contours)

    sheet_cut = []
    detachments_cut = []
    ratio_condition = True

    for cont in contours:
        x_min, x_max = cont[:, 0, 0].min(), cont[:, 0, 0].max()

        if x_min <= sheet_start and x_max <= sheet_end and cont.shape[0] > 100:
            sheet_cut.append(cont)
        elif x_max > sheet_end and cont.shape[0] > 100:
            detachments_cut.append(cont)

            # Check aspect ratio conditions for detachment contours
            # near sheet end
            x, y, w, h = cv2.boundingRect(cont)
            left_points = sum(1 for i in range(len(cont)) if cont[i, 0, 0] <= int(x) + 5)
            if int(x) <= sheet_end+5 and left_points > int(cont.shape[0]*0.4) and w/h <= 0.4:
                ratio_condition = False

    # Calculate areas and determine whether to apply the cut-off
    A_sheet_nocut = cv2.contourArea(sheet_nocut)
    A_sheet_cut = sum((cv2.contourArea(c) for c in sheet_cut))
    if A_sheet_cut == 0:
        A_sheet_cut = 1e9  # if no sheet is there, do not apply
                           # the cut-off algorithm!

    if A_sheet_nocut / A_sheet_cut > 1.25 and ratio_condition:
        for cont in sheet_cut:
            new_sheet_end = tuple(cont[cont[:, 0, 0].argmax()][0])[0]
            sheet_end = max(sheet_end, new_sheet_end)
    else:
        # Revert to original contours if no cut is applied
        sheet_cut = sheet_nocut
        detachments_cut = detachments_nocut
        sheet_end = tuple(sheet_nocut[sheet_nocut[:, 0, 0].argmax()][0])[0]

    return sheet_cut, detachments_cut, sheet_start, sheet_end

