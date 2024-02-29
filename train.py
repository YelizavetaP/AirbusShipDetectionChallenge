import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns        

from skimage.io import imread                        
from skimage.segmentation import mark_boundaries     #mark bounds around shapes
from skimage.util import montage                     #create montage od imgs
from skimage.morphology import label                 


TRAIN_IMAGE_DIR = "../data/train_v2"
CSV_PATH = "../data/train_ship_segmentations_v2.csv"

train = os.listdir(TRAIN_IMAGE_DIR)
masks = pd.read_csv(CSV_PATH)
print('Test train: \n', train[:5])
print('Test masks: \n',masks.head())


# Add extra col to mark if there is a ship
masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
print('Test ships: \n',masks.head(9))

# Sum up ship counts
unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index() 
unique_img_ids.index+=1 # Incrimenting all the index by 1
print('Test unique imgs: \n',unique_img_ids.head())

# Mark if ship exist on unique imgs
unique_img_ids['has_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x>0 else 0.0)
print('Test unique imgs ships: \n',unique_img_ids.head())

# add file size
unique_img_ids['file_size_kb'] = unique_img_ids['ImageId'].map(lambda c_img_id: os.stat(os.path.join(TRAIN_IMAGE_DIR, c_img_id)).st_size/1024)
print('Test file size: \n',unique_img_ids.head())


