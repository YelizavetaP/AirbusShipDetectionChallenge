# AirbusShipDetectionChallenge

train.py - contains train data preparation code and model training
Before training model train datast is split into 'train' and 'valid'.
From train data removed corrunted images. 
Data balanced using undersampling to have enouph images with different amount of ships.

TO RUN:
You need to change path to train imgs and masks in TRAIN_IMAGE_DIR and  CSV_PATH in the beggining of main function

MAIN FUNCTIONS:

splitTrainValid(masks) - removes corupted images and returns train/valid splits

sample_ships(in_df, base_rep_val=1500) & undersampling(df) - undersamplint train data so  that it has balanced number of samples with different ship amount 

rle_decode & masks_as_image - decodes masks from scv file

buildUnetModel - creates and returns Unet model


predict.py - use to visualize model work from terminal giving path to img as an argument
TO RUN:
Change path to test image IMG_PATH
Have saved model and weights is the same folder



test.py - contains code for proccesing imgs from test dataset and saving results in sample_submission_v2.csv file in RLE
TO USE:
Change path to test images TEST_IMAGE_DIR
Have saved model and weights is the same folder

MAIN FUNCTIONS:
normilize(test_df, TEST_IMAGE_DIR) - normilizes pixels of input images so their valus are in range 0 to 1

def rle_encode_ones(x) - RLE encoding for predicted mask to include only values of '1'