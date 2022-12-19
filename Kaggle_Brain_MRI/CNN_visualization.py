# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 13:47:25 2022

@author: alber
"""
#https://ravivaishnav20.medium.com/visualizing-feature-maps-using-pytorch-12a48cd1e573


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
pth_name="radiomics3dCNN_1912_resnet_moe_layers" #"radiomics3dCNN_1212_added_norm_recall_batch1d_init"#

path_local='C:/Users/alber/Bureau/Development/Data/Images_data/Radiomics_McMedHacks/'
device = torch.device("cuda")

path=path_local

#%% CT



n_patients = 3

dim1 = 185 - 70
dim2 = 170 - 30
dim3 = 230 - 40
df_cln = pd.read_excel(path+'Clinical_data_modified_2.xlsx', sheet_name = 'CHUM')

features = ['Sex', 'Age', 'Stage']
n_cln = len(features)

# Get patient's ID
p_id = list(range(1, n_patients+1))
p_id = [format(idx, '03d') for idx in p_id]  

X_cts = np.zeros((len(p_id), dim1, dim2, dim3))      # CT scans - 3D
X_dos = np.zeros((len(p_id), dim1, dim2, dim3))      # dose maps - 3D
X_cln = np.zeros((len(p_id), n_cln))                 # clinical variables - 1D


#%%% Read image data

for ip, p in enumerate(p_id):
    
    if ip%5 == 0:
        print(f" Extracting data ... {ip:>3d}/{n_patients} patients")
    
    # CT scanes
    image = sitk.ReadImage(path+f'warpedCT/warped_{p}.mha')
    image = sitk.GetArrayFromImage(image)
    image = image[70:185, 30:170, 40:230]
    X_cts[ip, :, :, :] = image
    
    # Dose maps
    image = sitk.ReadImage(path+f'warpedDose/HN-CHUM-{p}-dose-refct.mha') 
    image = sitk.GetArrayFromImage(image)
    image = image[70:185, 30:170, 40:230]
    X_dos[ip, :, :, :] = image
    
    # Clinical
    for ix, x in enumerate(features):
        X_cln[ip, ix] = df_cln[x][ip]
    
    # Let us diplay some slices
slc = 60
fig, axs = plt.subplots(1,2, figsize=(20, 20))
fig.subplots_adjust(hspace = .1, wspace=.001)
axs = axs.ravel()

for i in range(2):
    axs[i].imshow(X_cts[i, slc,:,:], cmap=plt.cm.Greys_r)
    axs[i].set_title(f"patient {i+1}")
#%%%
var = 1 # index of 'Age'
mean = X_cln[:,var].mean()
std  = X_cln[:,var].std()
#X_cln[:,var] = (X_cln[:,var]-mean)/std
X_cln=X_cln/X_cln.max()
print("norm_test cln",X_cln.min(),X_cln.max())


X_dos=X_dos/X_dos.max()
print("norm_test dos",X_dos.min(),X_dos.max())

X_cts=normalize_CT(X_cts)#X_cts/X_cts.max()
print("norm_test CT",X_cts.min(),X_cts.max())
#%%% train and test set
#Split data
train = 0

X_cln=X_cln/X_cln.max()
X_dos=X_dos/X_dos.max()
X_cts=normalize_CT(X_cts)#X_cts/X_cts.max()

X_dos_train = X_dos[train,:,:,:]
X_cts_train = X_cts[train,:,:,:]
X_cln_train = X_cln[train,:]

# convert data to tensors
X_cts_train  = torch.tensor(X_cts_train).float().unsqueeze(0)
X_dos_train  = torch.tensor(X_dos_train).float().unsqueeze(0)
X_cln_train  = torch.tensor(X_cln_train).float().unsqueeze(0)
print(X_cts_train.shape,X_cts_train.shape)
image_test=torch.cat((X_cts_train, X_dos_train), dim=0)
image = image_test.unsqueeze(0)
print(image.shape)

#%% Visu
pth="C:/Users/alber/Bureau/Development/DeepLearning/training_results/cluster/"+pth_name+"/"+pth_name+".pth"
print(pth)
#model = RadiomicsCNN(dim1,dim2,dim3,3)
model=ResNet(dim1,dim2,dim3,3,ResidualBlock, [3, 4, 6, 3])
#model= nn.DataParallel(model)#,device_ids=[0, 1])


state_dict = torch.load(pth)#, map_location=device)
model.load_state_dict(state_dict)
model.eval()
#model.to(device)
#print(model)
"""
layer = model.conv1[0]
print(layer)

filter1 = layer.weight.clone()
print(filter1.shape)
image=filter1.detach().numpy()*255
"""

#
model_weights =[]
#we will save the  conv layers in this list
conv_layers = []
# get all the model children as list
model_children = list(model.children())

#print(list(model_children[2][0].children())[0][0])
#counter to keep count of the conv layers
counter = 0
#append all the conv layers and their respective wights to the list
for i in range(len(model_children)):
    if type(model_children[i]) == nn.Conv3d:
        #print(model_children[i])
        counter+=1
        model_weights.append(model_children[i].weight)
        conv_layers.append(model_children[i])
    elif type(model_children[i]) == nn.Sequential:
        if type(model_children[i][0]) == nn.Conv3d:
            counter+=1
            model_weights.append(model_children[i][0].weight)
            conv_layers.append(model_children[i][0])
        else:
            for j in range(len(model_children[i])):
                for child in list(model_children[i][j].children()):
                    print("bug",child)
                    try:
                        for k in range(len(child)):
                            if type(child[k]) == nn.Conv3d:
                                #print(child[k])
        
                                counter+=1
                                model_weights.append(child[k].weight)
                                conv_layers.append(child[k])
                    except:
                        TypeError

print(f"Total convolution layers: {counter}")
#print("conv_layers")

print(len(conv_layers[0:]))

#%%% generate feature maps

outputs = []
names = []
#https://discuss.pytorch.org/t/downsampling-at-resnet/39038/5 
# issue with downsample 64 128...
#conv_layers=conv_layers[0:6]
conv_layers.pop(8) # [0,1,2,3,4,5,6,8,9,10,11,13,14,15,16,18,19]
conv_layers.pop(14)
conv_layers.pop(26)
conv_layers.pop(17)
conv_layers.pop(8)
conv_layers.pop(15)
conv_layers.pop(25)
print(conv_layers)
i=0
for layer in conv_layers[0:]:
    
    print(i,layer)
    i+=1
    print("")
    image = layer(image)
    outputs.append(image)
    names.append(str(layer))
print(len(outputs))
#print feature_maps
for feature_map in outputs:
    print(feature_map.shape)
    
#%%%

processed = []
for feature_map in outputs:
    feature_map = feature_map.squeeze(0)
    gray_scale = torch.sum(feature_map,0)
    gray_scale = gray_scale/feature_map.shape[0]
    processed.append(gray_scale.data.cpu().numpy())
for fm in processed:
    print(fm.shape)

#%%%
slice_number=7
fig = plt.figure(figsize=(30, 50))
for i in range(len(processed)):
    a = fig.add_subplot(5, 6, i+1)
    imgplot = plt.imshow(processed[i][slice_number,:,:])
    a.axis("off")
    a.set_title(names[i].split('(')[0], fontsize=30)#ajouter num layer
#plt.savefig(str('feature_maps.jpg'), bbox_inches='tight')

#%%% old code for 3dcnn

"""
for i in range(5):
    if type(model_children[i][0]) == nn.Conv3d:
        counter+=1
        model_weights.append(model_children[i][0].weight)
        conv_layers.append(model_children[i][0])
print(f"Total convolution layers: {counter}")
print("conv_layers")

print(conv_layers[0:])
"""