# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 13:43:21 2022

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

from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam, SGD
from torch.autograd import Variable
#%% Hyperparameters
bs = 14
n_epochs =100
learning_rate = 0.001 #0.01
loss_fn = nn.BCELoss()


#%% Extract clinical data and outcome
#%%% Path
path_cluster='/bigdata/casus/optima/data/Radiomics_McMedHacks/'
path_local='C:/Users/alber/Bureau/Development/Data/Images_data/Radiomics_McMedHacks/'
pth_path_cluster="/bigdata/casus/optima/"
pth_path_local="C:/Users/alber/Bureau/Development/DeepLearning/training_results/"


pth_file_name=pth_path_local+"radiomics_DVH"
path=path_local

#%%% Clinical data
df_cln = pd.read_excel(path+'Clinical_data_modified_2.xlsx', sheet_name = 'CHUM')
device = torch.device("cuda")


#%%% P

n_patients = 56
dim1 = 194#185 - 70
dim2 = 256#170 - 30
dim3 = 256#230 - 40


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
    #image = image[70:185, 30:170, 40:230]
    X_cts[ip, :, :, :] = image
    
    # Dose maps
    image = sitk.ReadImage(path+f'warpedDose/HN-CHUM-{p}-dose-refct.mha') 
    image = sitk.GetArrayFromImage(image)
    #image = image[70:185, 30:170, 40:230]
    X_dos[ip, :, :, :] = image
    
    # Clinical
    for ix, x in enumerate(features):
        X_cln[ip, ix] = df_cln[x][ip]
        
    # Outcomes
    y[ip] = df_cln['Death'][ip]


    # Also GTV contours for later use
    image = sitk.ReadImage(path+f'warpedGtv/HN-CHUM-{p}-gtv-refct.mha') 
    image = sitk.GetArrayFromImage(image)
    #image = image[70:185, 30:170, 40:230]
    X_gtv[ip, :, :, :] = image
    
 
print()
        
print(f"{sum(y)}/{len(y)} patients are positive")
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
optimizer = Adam(model2.parameters(), lr = learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, \
                        factor=0.5, patience=5, min_lr=0.00001*learning_rate,\
                            verbose=False)


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



# Cross-validation parameters
n_cv = 4
cv = StratifiedKFold(n_splits=n_cv)

# Training parameters




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