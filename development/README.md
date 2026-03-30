## Development

This folder includes all necessary python scripts. They are briefly summed up in the following list:

- Model architecture - unet_architecture.py: The U-Net model is implemented in this script. This is done for three different sizes (small, medium, large), they differ in the number of convolution and deconvolution blocks. The PyTorch library is used for implementation and the code is inspired by the github project https://github.com/aladdinpersson/Machine-Learning-Collection of Aladdin Persson.

- Model training - unet_training.py: This script is used to train the model, implemented in unet_architecture.py, with configurations specified in hyperparams_training.json. Trained model files (.pth.tar) and performance metrics (.json) are saved separately.
    - hyperparams_training.json: Config file for U-Net training. Contains hyperparameters and paths to training and test datasets.

- Hyperparameter tuning - unet_tuning.py: Fine-tunes the U-Net model implemented by unet_architecture.py. Requires hyperparams_tuning.json for configuration and allows customisation of the hyperparameter search space via a config dictionary within the script. Trial results are saved in the ray_results folder.
    - hyperparams_tuning.json: Config file for tuning the U-Net model, including paths to training and test datasets, and settings like the maximum number of epochs.

- Image pre-processing - image_enhancer.py: Pre-processes .tif images and calculates the pixel size. For blackening the insignificant areas above and below the hydrofoil, the edges of the hydrofoils must be given. This is done with the 'edges.csv' file in the example_images folder. For the classification of the cavitation regimes into sheet or detached clouds it is important to rotate the images to ensure that the flow direction is from left to right.

- Image analysis and post-processing - image_analysis.py: Contains functions for image segmentation and post-processing. The post-processing includes morphological operations, contour detection and classification.

The whole process of image analysis is shown in 'example.ipynb'.