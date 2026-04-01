# Image processing of cloud cavitation using semantic segmentation

This software analyses high-speed photographs of cavitation phenomena. A U-Net neural network is used for semantic segmentation, generating binary images that differentiate between cavitation regions and background. These binary images are further processed to outline cavitation contours and distinguish between attached sheet cavitation and detached clouds.

![me](example_images/example_movie.gif)

This work is based on the U-Net architecture presented in:

Ronneberger, O., Fischer, P., & Brox, T. **U-Net: Convolutional Networks for Biomedical Image Segmentation**  [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)

# Table of contents

- [Installation of required software](#installation-of-required-software)
- [Overview](#overview)
- [Citation](#citation)
- [Contact](#contact)


# Installation of required software

Clone this repository to your local machine: 
```git clone git@github.com:grigorishatConvolutional-Neural-Network-for-Cavitation-Segmentation.git```
Move to the project directory:
```cd Convolutional-Neural-Network-for-Cavitation-Segmentation```
Install the virtual environment from the ``environment.yml`` file: 
```conda env create -f environment.yml```
Activate the environment: 
```conda activate dpenv```


# Overview

- **`examples.ipynb`**  
  Jupyter notebook demonstrating the complete image segmentation workflow and serving as a usage guide.

- **`development/`**  
  Contains the source code for model architecture, training, and hyperparameter tuning, as well as image pre-processing, segmentation, and post-processing.

- **`model_parameters/`**  
  Stores trained model files (`.pth.tar`) together with their training performance metrics (`.json`).

- **`training_data/`**  
  Contains the images and corresponding labels used for training and testing the models.

- **`examples/`**  
  Contains sample images for analysis with the `examples.ipynb` notebook.


# Citation

If you use this software in your work, please cite the Zenodo record for this repository:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19354087.svg)](https://doi.org/10.5281/zenodo.19354087)


# Contact

Chair of Fluid Systems
Technische Universität Darmstadt

Grigorios Hatzissawidis
    mail  grigorios.hatzissawidis@gmail.com

Marlene Leimeister
    mail  marlene.leimeister@stud.tu-darmstadt.de

Jonas Bergner
    mail  jonast.bergner@gmail.com
