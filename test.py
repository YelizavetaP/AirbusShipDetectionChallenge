from keras.models import load_model
from matplotlib import pyplot as plt
from skimage.io import imread      
import tensorflow as tf            
import numpy as np
import os
import pandas as pd

# normilize all input imgs 
def normilize(test_df, TEST_IMAGE_DIR):
# list of all img id in test dir , test dir path
    
    all_imgs = list(test_df)                         
    out_rgb = []                                                             
                                                                                                                                     
    for img_id in all_imgs:        
        print(img_id)
        rgb_path = os.path.join(TEST_IMAGE_DIR, img_id)               
        c_img = imread(rgb_path)                                         
        out_rgb.append(np.stack([c_img])/255.0)

    return out_rgb                                           



def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.cast(tf.keras.backend.flatten(y_true), tf.float32)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)



def rle_encode_ones(x):

    rle_list = []

    prev_val = x.flatten()[0]
    count = 1
    x = x.flatten()
    for i in range (1, len(x)):
        val = x[i]
        if val == prev_val:
            count += 1
        else:
            if prev_val == 1:
                rle_list.append(i-1)
                rle_list.append(count)
            count = 1
        prev_val = val

    return rle_list



if __name__ == "__main__":

    TEST_IMAGE_DIR = "../data/test_v2"

    test = os.listdir(TEST_IMAGE_DIR)
    normilized_imgs = normilize(test[:100:1], TEST_IMAGE_DIR)
    # normilized_imgs = normilize(test, TEST_IMAGE_DIR)


    
    model = load_model("seg_model.h5", compile=False)
    model.compile(loss=dice_coef_loss, optimizer='adam', metrics=dice_coef)
    model.load_weights('seg_model_weights_16bs_5e.best.hdf5')

    df = pd.DataFrame(columns=['ImageId', 'EncodedPixels'])

    for id, img in zip(test, normilized_imgs):
        print('1')
        prediction = np.squeeze(model.predict(img), axis=0)
        print(prediction.shape, type(prediction))
        print(id)


        rle_mask = rle_encode_ones(prediction)
        rle_str = [str(elem) for elem in rle_mask]
        rle_str = ' '.join(rle_str)
        if len(rle_str) == 0: 
            rle_str = np.NaN
            print(rle_str)
        

        df = df._append({'ImageId': id, 'EncodedPixels': rle_str}, ignore_index=True)




    df.to_csv('output.csv', index=False)


    


    









