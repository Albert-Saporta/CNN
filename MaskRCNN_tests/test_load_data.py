# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 10:36:52 2023

@author: alber
"""
#!pip install numpy 
#https://bjornkhansen95.medium.com/mask-r-cnn-for-segmentation-using-pytorch-8bbfa8511883
#https://colab.research.google.com/drive/11FN5yQh1X7x-0olAOx7EJbwEey5jamKl?usp=sharing
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models
import torchvision.transforms as T
import skimage
from torchsummary import summary

import torch.optim as optim
from torch.autograd import Variable
from time import time
from natsort import natsorted
import os
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline 
import matplotlib.patches as mpatches
from matplotlib import patches
from tqdm import tqdm
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
#%%
local_train = "C:/Users/alber/Bureau/Development/Data/Images_data/Warwick_QU/train"
nucleus_path = "C:/Users/alber/Bureau/Development/Data/Images_data/cell_nucleus/small_subset"




# Read mask files from .png image


# Return mask, and array of class IDs of each instance. Since we have
# one class ID, we return an array of ones



test_walk=list(natsorted(os.walk(local_train, topdown=True)))
test=list((os.walk(os.path.join(nucleus_path, "images"))))
image = []
final_masks=[]
for (dirpath, dirnames, filenames) in os.walk(nucleus_path):
    mask = []

    for dirname in dirnames:
        if "image" in dirname:
            #print(os.listdir(dirpath+'/'+dirname))
            image_path=os.listdir(dirpath+'/'+dirname)[0]
            m = skimage.io.imread(os.path.join(dirpath+'/'+dirname, image_path))
            image.append(m)
        elif "mask" in dirname:
            mask_path=os.listdir(dirpath+'/'+dirname)
            #print(mask_path)
            for m_path in mask_path:
                #print(m_path)
                ma = skimage.io.imread(os.path.join(dirpath+'/'+dirname, m_path))
                mask.append(ma)
                mas = mask[:, :, np.where(class_ids == 1)[0]]
                mas = np.sum(m * np.arange(1, m.shape[-1] + 1), -1)
            #mask=np.array(mask)
            masks = np.stack(mas, axis=-1)
        final_masks.append(masks)         
print(final_masks[0].shape)          
print(image[0].shape) 

plt.imshow(final_masks[0])         

"""    
print(image)  
plt.imshow(image[1])    
"""
"""
print(test_walk)
print(test)

#plt.imshow(image_slices[0])
image_ids = next(os.walk(nucleus_path))[1]
print(image_ids)
"""
