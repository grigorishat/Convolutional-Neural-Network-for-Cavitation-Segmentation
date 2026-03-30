## Training data

The training data for image segmentation consists of pre-processed cavitation images and relating targets/labels. These labels are binary images destinguishing between cavitation (gray-value 1) and background (gray-value 0). They were created manually using the Open Data
Annotation Platform CVAT (https://app.cvat.ai). The dataset consists of 500 images and labels.

# Structure of the data set:
- train: 350 samples (each image+label)
- test: 150 samples (each image+label)

For hyperparameter tuning and training, the data is split further to create a subset of the data for hyperparameter validation.