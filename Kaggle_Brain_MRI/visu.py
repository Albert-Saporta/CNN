# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 13:47:25 2022

@author: alber
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 14:58:06 2022

@author: alber
"""


from modules import *
import numpy as np
import pandas as pd 
import seaborn as sns
from numpy import random
import matplotlib.pyplot as plt 
from os import listdir, mkdir
from os.path import isfile, join
from scipy import ndimage
from scipy.stats import mannwhitneyu

from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.utils import resample

# for evaluating the model
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix,recall_score
from sklearn.metrics import roc_auc_score, auc, roc_curve, precision_recall_curve
from scipy import interp

from random import randrange#bootstrapping
import SimpleITK as sitk

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam, SGD
from torch.autograd import Variable


#%% Path
pth_name="radiomics3dCNN_1212_added_norm_recall_batch1d_init"
path_local='C:/Users/alber/Bureau/Development/Data/Images_data/Radiomics_McMedHacks/'
pth_path_local="C:/Users/alber/Bureau/Development/DeepLearning/training_results/"
device = torch.device("cuda")

pth_file_name=pth_path_local+"radiomics3dCNN_0712"
path=path_local

#%% CT



n_patients = 2

dim1 = 185 - 70
dim2 = 170 - 30
dim3 = 230 - 40



# Get patient's ID
p_id = list(range(1, n_patients+1))
p_id = [format(idx, '03d') for idx in p_id]  

X_cts = np.zeros((len(p_id), dim1, dim2, dim3))      # CT scans - 3D


#%%% Read image data
for ip, p in enumerate(p_id):
    
    if ip%5 == 0:
        print(f" Extracting data ... {ip:>3d}/{n_patients} patients")
    
    # CT scanes
    image = sitk.ReadImage(path+f'warpedCT/warped_{p}.mha')
    image = sitk.GetArrayFromImage(image)
    image = image[70:185, 30:170, 40:230]
    X_cts[ip, :, :, :] = image
    
    # Let us diplay some slices
slc = 60
fig, axs = plt.subplots(1,2, figsize=(20, 20))
fig.subplots_adjust(hspace = .1, wspace=.001)
axs = axs.ravel()

for i in range(2):
    axs[i].imshow(X_cts[i, slc,:,:], cmap=plt.cm.Greys_r)
    axs[i].set_title(f"patient {i+1}")


