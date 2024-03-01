# AirbusShipDetectionChallenge

train.py - contains train data preparation code and model training
Before training model train datast is split into 'train' and 'valid'.
From train data removed corrunted images. 
Data balanced using undersampling to have enouph images with different amount of ships.

TO RUN:
You need to change path to train imgs and masks in TRAIN_IMAGE_DIR and  CSV_PATH in the beggining of main function



predict.py - use to visualize model work from terminal giving path to img as an argument
TO RUN:
Change path to test image IMG_PATH
Have saved model and weights is the same folder



test.py - contains code for proccesing imgs from test dataset and saving results in sample_submission_v2.csv file in RLE
TO USE:
Change path to test images TEST_IMAGE_DIR
Have saved model and weights is the same folder