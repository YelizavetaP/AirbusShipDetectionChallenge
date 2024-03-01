from keras.models import load_model
from matplotlib import pyplot as plt
from skimage.io import imread      
import tensorflow as tf            
import numpy as np
import os
import rle
import csv

# normilize all input imgs 
def normilize(test_df, TEST_IMAGE_DIR):
# list of all img id in test dir , test dir path
    
    all_imgs = list(test_df)                             
    out_rgb = []                                                             
                                                                                                                                     
    for img_id in all_imgs:                                
        rgb_path = os.path.join(TEST_IMAGE_DIR, img_id)               
        c_img = imread(rgb_path)                                         
        out_rgb += [c_img]  

    return np.stack([out_rgb])/255.0                                             



def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.cast(tf.keras.backend.flatten(y_true), tf.float32)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)





if __name__ == "__main__":

    TEST_IMAGE_DIR = "../data/test_v2"
    CSV_PATH = "../data/sample_submission_v2.csv"

    test = os.listdir(TEST_IMAGE_DIR)
    normilized_imgs = normilize(test, TEST_IMAGE_DIR)


    
    model = load_model("seg_model.h5", compile=False)
    model.compile(loss=dice_coef_loss, optimizer='adam', metrics=dice_coef)
    model.load_weights('seg_model_weights.best.hdf5')

    predicted_masks = []
    for img in normilized_imgs:
        prediction = np.squeeze(model.predict(img), axis=0)
        predicted_masks += prediction

    rle_predicted_masks = []
    for mask in predicted_masks:
        rle_mask = rle.encode(mask.flatten())
        rle_predicted_masks += rle_mask

    


    









