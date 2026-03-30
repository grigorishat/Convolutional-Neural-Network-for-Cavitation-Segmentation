## Image processing of cloud cavitation using semantic segmentation

This software analyses high-speed photographs of cavitation phenomena. A U-Net neural network is used for semantic segmentation, generating binary images that differentiate between cavitation regions and background. These binary images are further processed to outline cavitation contours and distinguish between attached sheet cavitation and detached clouds.

![me](example_images/example_movie.gif)

This work is based on the U-Net architecture presented in:

Ronneberger, O., Fischer, P., & Brox, T. **U-Net: Convolutional Networks for Biomedical Image Segmentation**  [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)
# Table of contents

 1. Installation of required software
 2. Overview
 3. License
 4. Contact


# 1. Installation of required software

Clone this repository to your local machine: 
```git clone git@github.com:grigorishatConvolutional-Neural-Network-for-Cavitation-Segmentation.git```
Move to the project directory:
```cd Convolutional-Neural-Network-for-Cavitation-Segmentation```
Install the virtual environment from the ``environment.yml`` file: 
```conda env create -f environment.yml```
Activate the environment: 
```conda activate dpenv```


# 2. Overview

## Repository Structure

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


# 3. License

MIT License

Copyright (c) 2024 Grigorios Hatzissawidis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.


# 4. Contact

Chair of Fluid Systems
Technische Universität Darmstadt

Grigorios Hatzissawidis
    mail  grigorios.hatzissawidis@gmail.com

Marlene Leimeister
    mail  marlene.leimeister@stud.tu-darmstadt.de

Jonas Bergner
    mail  jonast.bergner@gmail.com
