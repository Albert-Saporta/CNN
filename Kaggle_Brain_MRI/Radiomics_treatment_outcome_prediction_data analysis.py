# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 14:58:06 2022

@author: alber
"""
#!pip install --no-cache-dir smt
#from smt.sampling_methods import FullFactorial 
# !pip install medpy

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

from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam, SGD
from torch.autograd import Variable

#%% Extract clinical data and outcome
path='C:/Users/alber/Bureau/Development/Data/Images_data/Radiomics_McMedHacks/'
df_cln = pd.read_excel(f'{path}Clinical_data_modified_2.xlsx', sheet_name = 'CHUM')
print(df_cln.head())
device = torch.device("cuda")
pth_file_name="radiomics3dCNN"


#%%% P

n_patients = 56
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


    # Also GTV contours for later use
    image = sitk.ReadImage(path+f'warpedGtv/HN-CHUM-{p}-gtv-refct.mha') 
    image = sitk.GetArrayFromImage(image)
    image = image[70:185, 30:170, 40:230]
    X_gtv[ip, :, :, :] = image
    
 
print()
        
print(f"{sum(y)}/{len(y)} patients are positive")

#%%% Diplay some slices
slc = 60
fig, axs = plt.subplots(5,5, figsize=(20, 20))
fig.subplots_adjust(hspace = .1, wspace=.001)
axs = axs.ravel()

for i in range(25):
    axs[i].imshow(X_cts[i, slc,:,:], cmap=plt.cm.Greys_r)
    axs[i].set_title(f"patient {i+1}")
    
#reduce the size of the images to speed up
"""
RF = 0.3

print(f"number of voxels before resampling= {dim1*dim2*dim3}")

x_cts = ndimage.zoom(X_cts, (1, RF, RF, RF))
x_dos = ndimage.zoom(X_dos, (1, RF, RF, RF))

dim1, dim2, dim3 = np.shape(x_cts)[1], np.shape(x_cts)[2], np.shape(x_cts)[3]

print(f"number of voxels after resampling = {dim1*dim2*dim3}")

fig, axs = plt.subplots(5,5, figsize=(20, 20))
fig.subplots_adjust(hspace = .1, wspace=.001)
axs = axs.ravel()
slc = 15

for i in range(25):
    axs[i].imshow(x_cts[i, slc,:,:], cmap=plt.cm.Greys_r)
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
#%% Outcome Modeling
#Use a  CNN to predict the outcome from CT scans, dose maps and patient-specific clinical variables
#The model is composed on 2 paths; one to extract features from the images and the other process clinical variables
#concatenate both paths before prediction
#read https://aapm.onlinelibrary.wiley.com/doi/10.1002/mp.13122

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
#%%% model
model1 = RadiomicsCNN(dim1,dim2,dim3,n_cln)
print(model1)

#%%% Data preprocessing

#Split data
train = [int(x) for x in range(int(0.7*n_patients))]
test  = [x for x in range(n_patients-1) if x not in train] 

X_dos_train, X_dos_test = X_dos[train,:,:,:], X_dos[test,:,:,:]
X_cts_train, X_cts_test = X_cts[train,:,:],   X_cts[test,:,:,:]
X_cln_train, X_cln_test = X_cln[train,:],     X_cln[test,:]
y_train, y_test         = y[train],           y[test]


""" use scikit learn standard scaler"""
# Normalize continuous clinical variables
var = 1 # index of 'Age'
mean = X_cln_train[:,var].mean()
std  = X_cln_train[:,var].std()
X_cln_train[:,var] = ( X_cln_train[:,var] - mean ) / std
X_cln_test [:,var] = ( X_cln_test [:,var] - mean ) / std


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
bs = 10

train_set = TensorDataset(X_cts_train, X_dos_train, X_cln_train, y_train)
train_loader = DataLoader(train_set, batch_size = bs ) 

test_set = TensorDataset(X_cts_test, X_dos_test, X_cln_test, y_test)
test_loader = DataLoader(test_set, batch_size = bs )

#%% training

#%%% Hyperparameters

n_epochs =1# 100
learning_rate = 0.01
optimizer = Adam(model1.parameters(), lr = learning_rate)
loss_fn = nn.BCELoss()

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, \
                        factor=0.5, patience=5, min_lr=0.00001*learning_rate,\
                            verbose=False)

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

        Tloss = loss_fn(pred_train, y_train)
        train_loop.set_postfix(training_loss=Tloss.item())

        Tloss.backward()
        optimizer.step()
        
        train_loss += Tloss.item()
        
    train_loss = train_loss/(batch+1)        
    scheduler.step(train_loss)
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
        test_loop.set_postfix(test_loss=Vloss.item())
        test_loss += Vloss.item()
    
    test_loss = test_loss/(batch+1)  
    test_losses.append(test_loss)
    
    torch.save(model1.state_dict(), f'C:/Users/alber/Bureau/Development/DeepLearning/training_results/{pth_file_name}.pth')
    # Plot results
    if epoch%20 == 0 and epoch != 0:
      print(f"[{epoch:>3d}] \t Train : {train_loss:0.5f} \t Test : {test_loss:0.5f}")


plt.plot(list(range(epoch+1)), train_losses, label = 'Training')
plt.plot(list(range(epoch+1)), test_losses,  label = 'Validation')
plt.legend()
plt.xlabel("Epoch")

#%% validation. to do save model and use another code
"""
# Make prediction using the testing dataset
model1.eval()
y_true = torch.Tensor([]).to(device)
y_pred = torch.Tensor([]).to(device)
batch_accuracy = 0
for batch, (x_test, x_ct_test, x_clinical_test, y_test) in enumerate(test_loader):
    pred   = model1(x_test.to(device), x_ct_test.to(device), x_clinical_test.to(device))        
    y_true = torch.cat((y_true,y_test))
    y_pred = torch.cat((y_pred,pred))


# Calculate AUC
auc = roc_auc_score(y_true.detach().numpy().astype(int), y_pred.detach().numpy())
print(f"AUC : {auc}")

# Plot ROC Curve
fpr, tpr, thr = roc_curve(y_true.detach().numpy(), y_pred.detach().numpy())
plt.plot(fpr, tpr, 'b', label = f"AUC = {auc:0.3f}")
plt.plot([0, 1], [0, 1],'r--', label = 'Chance')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.legend()
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

def to_labels(pos_probs, threshold):
        return (pos_probs >= threshold).astype('int')
        
thr = .2
CM = confusion_matrix(y_true.detach().numpy().astype(int), to_labels(y_pred.detach().numpy(), thr).astype(int))

df_cm = pd.DataFrame(CM, index = [i for i in "PN"],
                  columns = [i for i in "PN"])
sns.heatmap(df_cm, annot=True, cmap = 'Blues')

#%%% Precision recall curve


prec, recall, thr2 = precision_recall_curve(y_true.detach().numpy().astype(int), to_labels(y_pred.detach().numpy(), thr).astype(int))

plt.plot(recall, prec, 'orange', lw=2)
no_skill = len(y_true[y_true==1]) / len(y_true)
plt.plot([0, 1], [no_skill, no_skill], '--')
plt.xlabel('Recall')
plt.ylabel('Precision')
"""
#%% prediction using DVH
#The dose-volume histogram (DVH) is a 2-D representation of the 3D dose maps
#It is used to evaluate plans and assess toxicity in organs-at-risk
#Although it ignores spatial information, the use of DVH saves significant computational cost, which could allow for better tuning and cross-validation of the model
#The DVH can be processed by the CNN as a signal/1D array
#Typically, the GTV dose is removed since DVH is calculated for different organs-at-risk (OAR)
#OARs are site-dependent


#In practice, specific OARs are tied to clinical endpoints, for which countours should be used

plt.imshow(X_gtv[0,30,:,:], cmap='gray_r')

dGy = 1
bin_edges = [i*dGy for i in range(76)]
X_dvh = np.zeros( (len(p_id), len(bin_edges)-1) )   # DVH - 1D

for ip, p in enumerate(p_id):
    
    if ip%5 == 0:
        print(f" Extracting DVH ... {ip:>3d}/{n_patients} patients")
    
    # Remove GTV
    dosemap  = X_dos[ip,:,:,:].copy()
    dosemap -= dosemap*X_gtv[ip,:,:,:]
    
    # Calculate DVH
    dvh, bins = np.histogram(dosemap, bins=bin_edges)

    # Noramlize the area under the diff DVH to 1
    dvh = dvh / sum(dvh*np.diff(bin_edges))

    
    X_dvh[ip, :] = dvh
    
# Look at how the DVH looks like
plt.plot(bins[0:-1],X_dvh[4, :])

#%%% Model  DVH

kernel_size  = 5
pool_size   = 3
model2 = RadiomicsDVH(kernel_size, pool_size,X_cln,X_dvh)
print(model2)

def train(n_epochs, train_loader, model2, loss_fn, optimizer):
    
    model2.train()

    for epoch in range(n_epochs):
        train_loss = 0

        # Training 

        for batch, (X_dvh_train, X_cln_train, y_train) in enumerate(train_loader):

            # Loss function
            loss_fn = nn.BCELoss()
            
            # zero the parameter gradients
            optimizer.zero_grad()
        
            # forward + backward + optimize
            pred_train = model2(X_dvh_train, X_cln_train)
            Tloss = loss_fn(pred_train, y_train)
            Tloss.backward()
            optimizer.step()
            
            train_loss += Tloss.item()
            
        train_loss = train_loss/(batch+1) 
#%%% Cros validate the model

bs = 14

# Cross-validation parameters
n_cv = 4
cv = StratifiedKFold(n_splits=n_cv)

# Training parameters
optimizer = Adam(model2.parameters(), lr = learning_rate)
loss_fn = nn.BCELoss()  # Check Train modeule. The loss function may be customized
n_epochs = 100 



# Initialize metrics
aucs, tprs, fprs            = [], [], np.linspace(0, 1, 100)
f1s, precs, recalls         = [], [], np.linspace(0, 1, 100)


# Start CV
for k, (train_id, test_id) in enumerate(cv.split(X_dvh, y)):

    print('------------------------------------------')
    print(f"CV \t {k}")


    # PREPARE DATASETS
    X_dvh_train, X_dvh_test = X_dvh[train_id,:], X_dvh[test_id,:]
    X_cln_train, X_cln_test = X_cln[train_id,:], X_cln[test_id,:]
    y_train, y_test         = y[train_id],       y[test_id]

    # --- Normalize continuous clinical variables
    var = 1 # index of 'Age'
    mean = X_cln_train[:,var].mean()
    std  = X_cln_train[:,var].std()
    X_cln_train[:,var] = ( X_cln_train[:,var] - mean ) / std
    X_cln_test [:,var] = ( X_cln_test [:,var] - mean ) / std

    # --- Convert data to tensors
    X_dvh_train  = torch.tensor(X_dvh_train).float().unsqueeze(1)
    X_cln_train  = torch.tensor(X_cln_train).float().unsqueeze(1)
    y_train      = torch.tensor(y_train).float()

    X_dvh_test  = torch.tensor(X_dvh_test).float().unsqueeze(1)
    X_cln_test  = torch.tensor(X_cln_test).float().unsqueeze(1)
    y_test      = torch.tensor(y_test).float()


    # --- Combine datasets
    train_set = TensorDataset(X_dvh_train, X_cln_train, y_train)
    train_loader = DataLoader(train_set, batch_size = bs ) 

    test_set = TensorDataset(X_dvh_test, X_cln_test, y_test)
    test_loader = DataLoader(test_set, batch_size = bs ) 





    # Train the model of train data k
    train(n_epochs, train_loader, model2, loss_fn, optimizer)
    


    # Validate the model of val data k
    model2.eval()
    y_true = torch.Tensor([])
    y_pred = torch.Tensor([])
    batch_accuracy = 0
    for batch, (X_dvh_test, X_cln_test, y_test) in enumerate(test_loader):
        pred  = model2(X_dvh_test, X_cln_test)        
        y_true = torch.cat((y_true,y_test))
        y_pred = torch.cat((y_pred,pred))
    
    
    # ROC
    auc_k = roc_auc_score(y_true.detach().numpy().astype(int), y_pred.detach().numpy())
    aucs.append(auc_k)
    fpr, tpr, _ = roc_curve(y_true.detach().numpy(), y_pred.detach().numpy())
    tpr = np.interp(fprs, fpr, tpr)
    tpr[0] = 0.0
    tprs.append(tpr)

    # TP, FP, TN, FN
    CM = confusion_matrix(y_true.detach().numpy().astype(int), np.round(y_pred.detach().numpy()).astype(int))
    TN = CM[0][0]
    FN = CM[1][0]
    FP = CM[0][1]
    TP = CM[1][1]
    print(f"\t TN   : {TN:>3d} \t \t FN  : {FN:>3d}")
    print(f"\t FP   : {FP:>3d} \t \t TP  : {TP:>3d} ")
    print(f"\t AUC  : {auc_k:0.3f}") 

#%%% Plot Roc Curve        
tprs = np.array(tprs)
mean_tprs = tprs.mean(axis=0)
tprs_up = np.percentile(tprs, 97.5, axis = 0)
tprs_lo = np.percentile(tprs, 2.5, axis = 0)
auc_up = np.percentile(aucs, 97.5, axis = 0)
auc_lo = np.percentile(aucs, 2.5, axis = 0)
plt.fill_between(fprs, tprs_lo, tprs_up, color='grey', alpha=0.3)
plt.plot(fprs, mean_tprs, 'k')

plt.plot([0, 1], [0, 1],'k--')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

plt.title(f"AUC {np.mean(aucs):0.2f} [{auc_lo:0.2f}, {auc_up:0.2f}]")

#%%% Grid search

def hp_score(kernel_size, pool_size, learning_rate,X_cln,X_dvh):  
  
  model3 = RadiomicsDVH(kernel_size, pool_size,X_cln,X_dvh)

  # Cross-validation parameters
  n_cv = 4
  cv = StratifiedKFold(n_splits=n_cv)

  # Training parameters
  optimizer = Adam(model3.parameters(), lr = learning_rate)
  loss_fn = nn.BCELoss()  
  n_epochs = 100 



  # Initialize metrics
  aucs, tprs, fprs            = [], [], np.linspace(0, 1, 100)
  f1s, precs, recalls         = [], [], np.linspace(0, 1, 100)


  # Start CV
  for k, (train_id, test_id) in enumerate(cv.split(X_dvh, y)):

      # print('------------------------------------------')
      # print(f"CV \t {k}")


      # PREPARE DATASETS
      X_dvh_train, X_dvh_test = X_dvh[train_id,:], X_dvh[test_id,:]
      X_cln_train, X_cln_test = X_cln[train_id,:], X_cln[test_id,:]
      y_train, y_test         = y[train_id],       y[test_id]

      # --- Normalize continuous clinical variables
      var = 1 # index of 'Age'
      mean = X_cln_train[:,var].mean()
      std  = X_cln_train[:,var].std()
      X_cln_train[:,var] = ( X_cln_train[:,var] - mean ) / std
      X_cln_test [:,var] = ( X_cln_test [:,var] - mean ) / std

      # --- Convert data to tensors
      X_dvh_train  = torch.tensor(X_dvh_train).float().unsqueeze(1)
      X_cln_train  = torch.tensor(X_cln_train).float().unsqueeze(1)
      y_train      = torch.tensor(y_train).float()

      X_dvh_test  = torch.tensor(X_dvh_test).float().unsqueeze(1)
      X_cln_test  = torch.tensor(X_cln_test).float().unsqueeze(1)
      y_test      = torch.tensor(y_test).float()


      # --- Combine datasets
      train_set = TensorDataset(X_dvh_train, X_cln_train, y_train)
      train_loader = DataLoader(train_set, batch_size = bs ) 

      test_set = TensorDataset(X_dvh_test, X_cln_test, y_test)
      test_loader = DataLoader(test_set, batch_size = bs ) 


      # Train the model of train data k
      train(n_epochs, train_loader, model3, loss_fn, optimizer)
      


      # Validate the model of val data k
      model3.eval()
      y_true = torch.Tensor([])
      y_pred = torch.Tensor([])
      batch_accuracy = 0
      for batch, (X_dvh_test, X_cln_test, y_test) in enumerate(test_loader):
          pred  = model3(X_dvh_test, X_cln_test)        
          y_true = torch.cat((y_true,y_test))
          y_pred = torch.cat((y_pred,pred))
      
      
      # ROC
      auc_k = roc_auc_score(y_true.detach().numpy().astype(int), y_pred.detach().numpy())
      aucs.append(auc_k)
      fpr, tpr, _ = roc_curve(y_true.detach().numpy(), y_pred.detach().numpy())
      tpr = np.interp(fprs, fpr, tpr)
      tpr[0] = 0.0
      tprs.append(tpr)

      # TP, FP, TN, FN
      CM = confusion_matrix(y_true.detach().numpy().astype(int), np.round(y_pred.detach().numpy()).astype(int))
      TN = CM[0][0]
      FN = CM[1][0]
      FP = CM[0][1]
      TP = CM[1][1]
      # print(f"\t TN   : {TN:>3d} \t \t FN  : {FN:>3d}")
      # print(f"\t FP   : {FP:>3d} \t \t TP  : {TP:>3d} ")
      # print(f"\t AUC  : {auc:0.3f}") 
      
  
  return np.mean( np.array(aucs) )

# Other sampling methods include Random; and Latin Hyper Cube 
#%% smt does not wprk (pip install fail)
# Define the upper and lower bounds of the hyperparams
"""
kernel_lims = [1., 10.]
pool_lims   = [1., 10.]
lr_lims     = [-4., 0] # 1e-4 - 0

limits = np.array([kernel_lims, pool_lims, lr_lims])
    
sampling = FullFactorial(xlimits=limits)

num = 10
samples = sampling(num)


best_AUC, best_KS, best_PS, best_LR = 0, 0, 0, 0

for iteration, sample in enumerate(samples):
  
  KS = int(sample[0])
  PS = int(sample[1])
  LR = 10**(sample[2])

  print("-------------------------------------------------------\n")
  print(f"HP Eval \t [{(iteration+1):>5d}/{num:>5d}]\n")
  print("Hyperparameters")
  print(f"\t Kernel Size     : {KS}")
  print(f"\t Pool Size       : {PS}")
  print(f"\t Learning Rate   : {LR:0.4f}")


  # Evaluate the HP
  AUC = hp_score(KS, PS, LR)
  print(f"\n AUC                    : {AUC:0.5f}")


  # Update best HPO
  if best_AUC<AUC:
      best_AUC = AUC
      best_KS = KS
      best_PS = PS
      best_LR = LR


print("-------------------------------------------------------\n")
print("End of HPO\n")
print(f"Best Hyperparameters")
print(f"\t Kernel Size     : {best_KS}")
print(f"\t Pool Size       : {best_PS}")
print(f"\t Learning Rate   : {best_LR}")
print(f"Best AUC  : {best_AUC:0.5f}\n")
"""
#%% Refinement : Always look at the precision-recall curve