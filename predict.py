from keras.models import load_model
from matplotlib import pyplot as plt
from skimage.io import imread      
import tensorflow as tf            
import numpy as np


def preprocess_image(img):
    return np.stack([img])/255.0


def show(original, mask):

    plt.figure(figsize=(11, 6))
    plt.subplot(1, 2, 1)
    plt.title(f"Image {original.shape}")
    plt.imshow(original)
    plt.subplot(1, 2, 2)
    plt.title(f"Mask {mask.shape}")
    plt.imshow(mask, cmap = "Blues_r")
    plt.tight_layout()
    plt.show()


# Load the image
IMG_PATH = '../data/test_v2/0c0d90d8d.jpg'
originalimg = imread(IMG_PATH)                                         

image = preprocess_image(originalimg)

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.cast(tf.keras.backend.flatten(y_true), tf.float32)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


model = load_model("seg_model.h5", compile=False)
model.compile(loss=dice_coef_loss, optimizer='adam', metrics=dice_coef)
model.load_weights('seg_model_weights.best.hdf5')

prediction = np.squeeze(model.predict(image), axis=0)

print(prediction.shape)
print(originalimg.shape)

show(originalimg, prediction)

