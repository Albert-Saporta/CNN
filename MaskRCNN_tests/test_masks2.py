import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import json
import pandas as pd
#from datasets import *
from tqdm import tqdm
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from scipy.ndimage import label
TRAIN_PATH="C:/Users/alber/Bureau/Development/Data/Images_data/cell_nucleus/stage1_test/"
outpath="C:/Users/alber/Bureau/Development/Data/Images_data/cell_nucleus/test/mask/"

seed = 42
random.seed = seed
np.random.seed = seed

# Get train and test IDs
train_ids = next(os.walk(TRAIN_PATH))[1]

# Get and resize train images and masks

IMG_CHANNELS = 3
print('Getting and resizing train images and masks ... ')
#sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    #print(id_)
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')
    #img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    mask = np.zeros((img.shape[0], img.shape[1], IMG_CHANNELS), dtype=np.bool_)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_=cv2.cvtColor(mask_,cv2.COLOR_GRAY2RGB)


  
        mask,ref_num_features = label(np.maximum(mask, mask_))
        #print(outpath+str(id_))
    cv2.imwrite(outpath+str(id_)+".jpg", mask)
print(mask.shape)
plt.imshow(mask)
plt.show()