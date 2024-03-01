import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns        
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator


from skimage.io import imread                        
from skimage.util import montage                     #create montage od imgs
from skimage.morphology import label                 


# Parameters
BATCH_SIZE = 4                 # Train batch size
EDGE_CROP = 16                 # While building the model
NB_EPOCHS = 5                  # Training epochs
GAUSSIAN_NOISE = 0.1           # To be used in a layer in the model
UPSAMPLE_MODE = 'SIMPLE'       # SIMPLE ==> UpSampling2D, else Conv2DTranspose
NET_SCALING = None             # Downsampling inside the network                        
IMG_SCALING = (1, 1)           # Downsampling in preprocessing
VALID_IMG_COUNT = 400          # Valid batch size
MAX_TRAIN_STEPS = 200          # Maximum number of steps_per_epoch in training



def splitTrainValid(masks):

    # Add extra col to mark if there is a ship
    masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)

    # Sum up ship counts
    unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index() 
    unique_img_ids.index+=1

    # Mark if ship exist on unique imgs
    unique_img_ids['has_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x>0 else 0.0)

    # add file size
    unique_img_ids['file_size_kb'] = unique_img_ids['ImageId'].map(lambda c_img_id: os.stat(os.path.join(TRAIN_IMAGE_DIR, c_img_id)).st_size/1024)
    
    # remove 'ships' from original masks
    unique_img_ids = unique_img_ids[unique_img_ids.file_size_kb>35]

    masks.drop(['ships'], axis=1, inplace=True)
    masks.index+=1 

    # split into train and test 
    # stratify - same proportions for train and test
    train_ids, valid_ids = train_test_split(unique_img_ids, test_size = 0.3, stratify = unique_img_ids['ships'])
    train_df = pd.merge(masks, train_ids)
    valid_df = pd.merge(masks, valid_ids)

    return  train_df, valid_df


def sample_ships(in_df, base_rep_val=1500):
    
    # in_df - dataframe for undersampling
    # base_val - random sample of this value to be taken from the data frame

    if in_df['ships'].values[0]==0:                                                 
        return in_df.sample(base_rep_val//3)  
    else:                                 
        return in_df.sample(base_rep_val)    

def undersampling(df):

    df['grouped_ship_count'] = df.ships.map(lambda x: (x+1)//2).clip(0,7)
    balanced_df = df.groupby('grouped_ship_count').apply(sample_ships)
    return balanced_df



# Decode mask for one ship on img
def rle_decode(mask_rle, shape=(768,768)):
  
    # mask_rle: Mask of one ship in the train image
    # shape: Output shape of the image array

    s = mask_rle.split()                                                               # Split the mask of each ship that is in RLE format
    starts, lengths = [np.array(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]     # Get the start pixels and lengths for which image has ship
    ends = starts + lengths - 1                                                        # Get the end pixels where we need to stop
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)                                  # A 1D vec full of zeros of size = 768*768
    for start, end in zip(starts, ends):                                                   # For each start to end pixels where ship exists
        img[start:end+1] = 1                                                               # Fill those values with 1 in the main 1D vector
 
    # Transposed array of the mask: Contains 1s and 0s. 1 for ship and 0 for background
    return img.reshape(shape).T                                                       


# Combine all masks for one img
def masks_as_image(img_mask_list):

    # in_mask_list: List of the masks of each ship in one whole training image
    
    all_masks = np.zeros((768, 768), dtype = np.int16)                                 # Creating 0s for the background
    for mask in img_mask_list:
        if isinstance(mask, str):                                                      # If the datatype is string
            # If the datatype is string
            all_masks += rle_decode(mask)                                              # Use rle_decode to create one mask for whole image

    # Full mask of the training image whose RLE data has been passed as an input
    return np.expand_dims(all_masks, -1)



# Image and Mask Generator
def make_image_gen(in_df, batch_size = BATCH_SIZE):

    # in_df - data frame 
    # batch_size - number of training examples in one iteration
 
    # mask grouped for every img
    all_batches = list(in_df.groupby('ImageId'))                             # Group ImageIds and create list of that dataframe
    out_rgb = []                                                             # Image list
    out_mask = []                                                            # Mask list
    while True:                                                              # Loop for every data
        np.random.shuffle(all_batches)                                       # Shuffling the data
        for img_id, mask_rle in all_batches:                                # For img_id and msk_rle in all_batches
            rgb_path = os.path.join(TRAIN_IMAGE_DIR, img_id)               # Get the img path
            c_img = imread(rgb_path)                                         # img array
            c_mask = masks_as_image(mask_rle['EncodedPixels'].values)         # Create mask of rle data for each ship in an img
            out_rgb += [c_img]                                               # Append the current img in the out_rgb / img list
            out_mask += [c_mask]                                             # Append the current mask in the out_mask / mask list
            if len(out_rgb)>=batch_size:                                     # If length of list is more or equal to batch size then
                yield np.stack(out_rgb)/255.0, np.stack(out_mask)            # Yeild the scaled img array (b/w 0 and 1) and mask array (0 for bg and 1 for ship)
                out_rgb, out_mask=[], []                                     # Empty the lists to create another batch


def create_aug_gen(in_gen, seed = None):
  
    # in_gen - train data generator, seed value

    np.random.seed(seed if seed is not None else np.random.choice(range(9999)))  # Randomly assign seed value if not provided
    for in_x, in_y in in_gen:                                                    # For imgs and msks in train data generator
        seed = 12                                                                # Seed value for imgs and msks must be same else augmentation won't be same
        
        # Create augmented imgs
        g_x = image_gen.flow(255*in_x,                                           # Inverse scaling on imgs for augmentation                                       
                             batch_size = in_x.shape[0],                         # batch_size = 3
                             seed = seed,                                        # Seed
                             shuffle=True)                                       # Shuffle the data
        
        # Create augmented masks
        g_y = label_gen.flow(in_y,
                             batch_size = in_x.shape[0],                       
                             seed = seed,                                         
                             shuffle=True)                                       
        
        # Yeilds - augmented scaled imgs and msks array
        yield next(g_x)/255.0, next(g_y)




if __name__ == "__main__":

    TRAIN_IMAGE_DIR = "../data/train_v2"
    CSV_PATH = "../data/train_ship_segmentations_v2.csv"



    train = os.listdir(TRAIN_IMAGE_DIR)
    masks = pd.read_csv(CSV_PATH)


    train_df, valid_df = splitTrainValid(masks)


    # prepare train data 
    balanced_train_df = undersampling(train_df)
    train_gen = make_image_gen(balanced_train_df)

    # Image and Mask - x img pixels values in range 0 - 1, y - mask 0 or 1
    t_x, t_y = next(train_gen)
 

    # Augmenting Data
    # Preparing image data generator arguments
    dg_args = dict(rotation_range = 15,         # Degree range for random rotations
                horizontal_flip = True,         # Randomly flips the inputs horizontally
                vertical_flip = True,           # Randomly flips the inputs vertically
                data_format = 'channels_last')  # channels_last refer to (batch, height, width, channels)
    
    image_gen = ImageDataGenerator(**dg_args)
    label_gen = ImageDataGenerator(**dg_args)
    # Augment the train data
    cur_gen = create_aug_gen(train_gen, seed = 42)
    train_x, train_y = next(cur_gen)
    print(f"train_x ~\nShape: {train_x.shape}\nMin value: {train_x.min()}\nMax value: {train_x.max()}")
    print(f"\ntrain_y ~\nShape: {train_y.shape}\nMin value: {train_y.min()}\nMax value: {train_y.max()}")
    


    # Prepare validation data
    valid_x, valid_y = next(make_image_gen(valid_df, VALID_IMG_COUNT))
    print(f"valid_x ~\nShape: {valid_x.shape}\nMin value: {valid_x.min()}\nMax value: {valid_x.max()}")
    print(f"\nvalid_y ~\nShape: {valid_y.shape}\nMin value: {valid_y.min()}\nMax value: {valid_y.max()}")
















