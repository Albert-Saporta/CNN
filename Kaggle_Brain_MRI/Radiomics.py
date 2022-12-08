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
from os import listdir
from os.path import isfile, join
from scipy import ndimage
from scipy.stats import mannwhitneyu

from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.utils import resample

# for evaluating the model
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.metrics import roc_auc_score, auc, roc_curve, precision_recall_curve
from scipy import interp

from random import randrange#bootstrapping
import SimpleITK as sitk

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam, SGD
from torch.autograd import Variable

#%% Hyperparameters
bs = 4
n_epochs =1000
learning_rate = 0.001 #0.01
loss_fn = nn.BCELoss()


#%% Extract clinical data and outcome
#%%% Path
path_cluster='/bigdata/casus/optima/data/Radiomics_McMedHacks/'
path_local='C:/Users/alber/Bureau/Development/Data/Images_data/Radiomics_McMedHacks/'
pth_path_cluster="/bigdata/casus/optima/"
pth_path_local="C:/Users/alber/Bureau/Development/DeepLearning/training_results/"


pth_file_name=pth_path_cluster+"radiomics3dCNN_0812"
path=path_cluster
device = torch.device("cuda")

"""
pth_file_name=pth_path_local+"radiomics3dCNN_0712"
path=path_local
device = torch.device("cuda")
"""

#%%% Clinical data
df_cln = pd.read_excel(path+'Clinical_data_modified_2.xlsx', sheet_name = 'CHUM')


#%%% P

n_patients = 56
"""
dim1 = 194
dim2 = 256
dim3 = 256
"""

dim1 = 185 - 70
dim2 = 170 - 30
dim3 = 230 - 40




# Clinical data and outcomes
features = ['Sex', 'Age', 'Stage']
n_cln = len(features)

# Get patient's ID
p_id = list(range(1, n_patients+1))
p_id = [format(idx, '03d') for idx in p_id]  

X_cts = np.zeros( (len(p_id), dim1, dim2, dim3) )      # CT scans - 3D
X_dos = np.zeros( (len(p_id), dim1, dim2, dim3) )      # dose maps - 3D
X_cln = np.zeros( (len(p_id), n_cln) )                 # clinical variables - 1D
X_gtv = np.zeros( (len(p_id), dim1, dim2, dim3) )      # GTV contours  - 3D

y     = np.zeros( (len(p_id) ) ) # outcomes - binary

# Read data
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
        
    # Outcomes
    y[ip] = df_cln['Death'][ip]

    """
    # Also GTV contours for later use
    image = sitk.ReadImage(path+f'warpedGtv/HN-CHUM-{p}-gtv-refct.mha') 
    image = sitk.GetArrayFromImage(image)
    image = image[70:185, 30:170, 40:230]
    X_gtv[ip, :, :, :] = image
    """
 
print()
        
print(f"{sum(y)}/{len(y)} patients are positive")

#%%% Diplay some slices
"""
slc = 60
fig, axs = plt.subplots(5,5, figsize=(20, 20))
fig.subplots_adjust(hspace = .1, wspace=.001)
axs = axs.ravel()

for i in range(25):
    axs[i].imshow(X_cts[i, slc,:,:], cmap=plt.cm.Greys_r)
    axs[i].set_title(f"patient {i+1}")
"""   

#%% Statistical significance and p value
"""
U    = np.zeros((dim1, dim2, dim3))
pval = np.ones((dim1, dim2, dim3))

# For each voxel
for ii in range(dim1):
    for jj in range(dim2):
        for kk in range(dim3):
            
            # Combine voxel doses of neg/pos patients together
            group_0 = X_dos[:,ii,jj,kk] [y == 0]
            group_1 = X_dos[:,ii,jj,kk] [y == 1]
            
            if sum(group_1) == 0:
                continue
            # Run statistical test to determine if they are significantly different
            # This is a 2 sample test
            # We have not checked the data distrubtion 
            # So we will use a non-parametric test: Mann–Whitney U test
            # Other tests include Student's t-test, Wilcoxon test, Karuskal-Wallis, etc
            # Each test has its own adjavntages and disadvantages
            
            U[ii,jj,kk], pval[ii,jj,kk]= mannwhitneyu(group_0, group_1)
    
    if ii%10 == 0: 
        print(f"Slice {ii}")
        
sl = 25
sns.heatmap(-np.log( pval[sl,:,:] ), cbar_kws = {'label': '-log(p)'}, cmap = 'gray_r')
            
#%%% significance
alpha = 0.05 # significance level
hits = np.zeros((dim1, dim2, dim3))
hits[ pval < alpha ] = 1

sns.heatmap(hits[sl,:,:], cmap = 'gray_r')
print(f"Total number of hits : {hits.sum().astype(int)}/{dim1*dim2*dim3}")

#%%%

# Bonferroni correction
m = (dim1*dim2*dim3)
alpha = 0.05
alpha = alpha/m
hits = np.zeros((dim1, dim2, dim3))
hits[ pval < alpha ] = 1

sns.heatmap(hits[sl,:,:], cmap = 'gray_r')
plt.figure(figsize=(4, 4))
plt.title("After Correction")
plt.title(f"After correction:      hits : {hits.sum().astype(int)}")
"""
# Reduce the images further
"""
RF = 0.2

x_cts = ndimage.zoom(X_cts, (1, RF, RF, RF))
x_dos = ndimage.zoom(X_dos, (1, RF, RF, RF))

dim1, dim2, dim3 = np.shape(x_cts)[1], np.shape(x_cts)[2], np.shape(x_cts)[3]

print(f"number of voxels after resampling = {dim1*dim2*dim3}")

# Let's check now how it looks like
fig, axs = plt.subplots(5,5, figsize=(20, 20))
fig.subplots_adjust(hspace = .1, wspace=.001)
axs = axs.ravel()
slc = 15

for i in range(25):
    axs[i].imshow(x_cts[i, slc,:,:], cmap=plt.cm.Greys_r)
    axs[i].set_title(f"patient {i+1}")
"""  
#%% Outcome Modeling
#Use a  CNN to predict the outcome from CT scans, dose maps and patient-specific clinical variables
#The model is composed on 2 paths; one to extract features from the images and the other process clinical variables
#concatenate both paths before prediction
#read https://aapm.onlinelibrary.wiley.com/doi/10.1002/mp.13122

 
#%% model
model1 = RadiomicsCNN(dim1,dim2,dim3,n_cln)
print(model1)

optimizer = Adam(model1.parameters(), lr = learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, \
                        factor=0.5, patience=5, min_lr=0.00001*learning_rate,\
                            verbose=False)
# mettre avant optimizer?
def weights_init(m):
   if isinstance(m, nn.Conv3d):
       torch.nn.init.normal_(m.weight, 0.0, 0.02)
   if isinstance(m, nn.BatchNorm3d):
       torch.nn.init.normal_(m.weight, 0.0, 0.02)
       torch.nn.init.constant_(m.bias, 0)
   if isinstance(m, nn.Linear):
       torch.nn.init.normal_(m.weight, 0.0, 0.02)
       torch.nn.init.constant_(m.bias, 0)
            
model1 = model1.apply(weights_init)
#%% Data preprocessing

#%%% Normalization


""" use scikit learn standard scaler"""
# Normalize continuous clinical variables
#X_cln=X_cln/X_cln.max()

var = 1 # index of 'Age'
mean = X_cln_train[:,var].mean()
std  = X_cln_train[:,var].std()
X_cln_train[:,var] = ( X_cln_train[:,var] - mean ) / std
X_cln_test [:,var] = ( X_cln_test [:,var] - mean ) / std


X_dos=X_dos/X_dos.max()
X_cts=X_cts/X_cts.max()

#%%% train and test set
#Split data
train = [int(x) for x in range(int(0.7*n_patients))]
test  = [x for x in range(n_patients-1) if x not in train] 

X_dos_train, X_dos_test = X_dos[train,:,:,:], X_dos[test,:,:,:]
X_cts_train, X_cts_test = X_cts[train,:,:],   X_cts[test,:,:,:]
X_cln_train, X_cln_test = X_cln[train,:],     X_cln[test,:]
y_train, y_test         = y[train],           y[test]




# convert data to tensors
X_cts_train  = torch.tensor(X_cts_train).float().unsqueeze(1)
X_dos_train  = torch.tensor(X_dos_train).float().unsqueeze(1)
X_cln_train  = torch.tensor(X_cln_train).float().unsqueeze(1)
# X_dvh_train  = torch.tensor(X_dvh_train).float().unsqueeze(1)
y_train      = torch.tensor(y_train).float()

X_cts_test  = torch.tensor(X_cts_test).float().unsqueeze(1)
X_dos_test  = torch.tensor(X_dos_test).float().unsqueeze(1)
X_cln_test  = torch.tensor(X_cln_test).float().unsqueeze(1)
# X_dvh_test  = torch.tensor(X_dvh_test).float().unsqueeze(1)
y_test      = torch.tensor(y_test).float()


# Combine datasets


train_set = TensorDataset(X_cts_train, X_dos_train, X_cln_train, y_train)
train_loader = DataLoader(train_set, batch_size = bs,pin_memory=True,shuffle=True) 

test_set = TensorDataset(X_cts_test, X_dos_test, X_cln_test, y_test)
test_loader = DataLoader(test_set, batch_size = bs,pin_memory=True,shuffle=True)

#%% training


model1.to(device)

train_losses = []
test_losses = []
plt.figure(figsize = (4,4))
for epoch in range(n_epochs):
    train_loss = 0

    # Training 
    model1.train()
    train_loop=tqdm(train_loader)
    for batch, (X_cts_train, X_dos_train, X_cln_train, y_train) in enumerate(train_loop):
        train_loop.set_description(f'Epoch {epoch+1}/{n_epochs}')
        # print(y_train)
        X_cts_train, X_dos_train, X_cln_train, y_train=X_cts_train.to(device), X_dos_train.to(device), X_cln_train.to(device), y_train.to(device)
        # loss function
        #loss_fn = nn.BCELoss()

        # zero the parameter gradients
        optimizer.zero_grad()
    
        # forward + backward + optimize
        pred_train = model1(X_cts_train, X_dos_train, X_cln_train)
        
        try:
            auc = roc_auc_score(y_train.cpu().detach().numpy().astype(int), pred_train.cpu().detach().numpy())
            #auc = roc_auc_score(y_train.detach().numpy().astype(int), pred_train.detach().numpy())

            #print(f"AUC : {auc}")
        except ValueError:
            pass
        
        Tloss = loss_fn(pred_train, y_train)
        train_loop.set_postfix(AUC=auc,train_loss=Tloss.item())

        Tloss.backward()
        optimizer.step()
        
        train_loss += Tloss.item()
        
    train_loss = train_loss/(batch+1)        
    #scheduler.step(train_loss)
    train_losses.append(train_loss)
    
    # Model validation
    model1.eval()
    Y = torch.Tensor([])
    Y_hat = torch.Tensor([])
    batch_accuracy = 0
    test_loss = 0
    test_loop=tqdm(test_loader)
    for batch, (X_cts_test, X_dos_test, X_cln_test, y_test) in enumerate(test_loop):
        X_cts_test, X_dos_test, X_cln_test, y_test=X_cts_test.to(device), X_dos_test.to(device), X_cln_test.to(device), y_test.to(device)
        pred_test  = model1(X_cts_test, X_dos_test, X_cln_test)        
        Vloss = loss_fn(pred_test, y_test)
        
        try:
            auc_test = roc_auc_score(y_test.cpu().detach().numpy().astype(int), pred_test.cpu().detach().numpy())
            #auc_test = roc_auc_score(y_test.detach().numpy().astype(int), pred_test.detach().numpy())

            #print(f"AUC : {auc}")
        except ValueError:
            pass
        test_loop.set_postfix(AUC=auc_test,test_loss=Vloss.item())
        test_loss += Vloss.item()
    
    test_loss = test_loss/(batch+1)  
    test_losses.append(test_loss)
    
    torch.save(model1.state_dict(), pth_file_name+'.pth')

    # Plot results
    if epoch%20 == 0 and epoch != 0:
      print(f"[{epoch:>3d}] \t Train : {train_loss:0.5f} \t Test : {test_loss:0.5f}")


plt.plot(list(range(epoch+1)), train_losses, label = 'Training')
plt.plot(list(range(epoch+1)), test_losses,  label = 'Validation')
plt.legend()
plt.savefig(pth_path_cluster+'Learning_Curves.pdf',format='pdf')
plt.xlabel("Epoch")