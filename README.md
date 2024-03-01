# AirbusShipDetectionChallenge

train.py - contains train data preparation code and model training
Before training model train datast is split into 'train' and 'valid'. From train data removed corrunted images. Also it balanced using undersampling to have enouph images with different amount of ships.

To run train.py you need to change path to train imgs and masks in TRAIN_IMAGE_DIR and  CSV_PATH in the beggining of main function

train_ship_segmentations_v2.csv"

predict.py - use to visualize model work from terminal giving path to img as an argument
test.py - contains code for proccesing imgs from test dataset and saving results in sample_submission_v2.csv file in RLE