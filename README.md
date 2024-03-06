# AirbusShipDetectionChallenge



## Overview

This project is a solution for Kaggle Airbus Ship Detection Challenge using the U-Net model architecture along with custom loss functions and metrics, specifically Dice score. The solution is divided into two main files: one for model training and another for inference. In addition to training the model on the entire dataset and performing inference on the test data, you can also use the trained model to predict segmentation masks for individual images using `predict.py`.


## Files & Instructions for Use

1. `train.py`: This file contains the code for preparing training data, creating custom loss functions and metrics (Dice score), and building and training the U-Net model.
    To run:
    1.  Navigate to the directory containing the `train.py` file
    2. Run the following command, replacing `TRAIN_IMAGE_DIR` with the path to the folder with train images, `CSV_PATH` with path fo file with masks, `NB_EPOCHS` with number of epochs to train midel.

    ```
        python train.py TRAIN_IMAGE_DIR CSV_PATH NB_EPOCHS
    ```
    3. Created model arcitecture will be saved in `seg_model.h5` and trained weights to `seg_model_weights.best.hdf5` in the same directory as `train.py` file.


2. `test.py`: This file is responsible for utilizing the already trained U-Net model to perform inference on the test data and save the results in a CSV file, containing pairs of image IDs and received RLE masks.
    To run:
    1. Navigate to the directory containing the `test.py` file
    2. Run the following command, replacing `TEST_IMAGE_DIR` with the path to the folder with test images, `MODEL_PATH` with path to `.h5` file with saved model architecture and `WEIGHTS_PATH` with path to `.hdf5` file to trained weights file.

    ```
        python test.py TEST_IMAGE_DIR MODEL_PATH WEIGHTS_PATH
    ```
    3. Resuls saved to `output.csv` file in the same directory.


3. `predict.py`: Uses already trained model to predict mask for individeual image and visualizes results. 
    To run:
    1. Navigate to the directory containing the `predict.py` file
    2. Run the following command, replacing `IMAGE_PATH` with the path to the image you want to segment:

    ```
        python predict.py IMAGE_PATH
    ```


## Model Architecture

The U-Net architecture is chosen for its effectiveness in semantic segmentation tasks. It consists of a contracting path to capture context and a symmetric expanding path to enable precise localization. This architecture is well-suited for segmenting objects of interest from images.

## Custom Loss Function and Metrics

The Dice score, also known as the Dice similarity coefficient, is utilized as a custom metric and loss function during model training. It measures the overlap between the predicted segmentation mask and the ground truth mask. Maximizing the Dice score during training helps in accurately segmenting ships from satellite images.


## Dependencies

Install all dependencies from reqirements.txt using pip install -r requirements.txt





