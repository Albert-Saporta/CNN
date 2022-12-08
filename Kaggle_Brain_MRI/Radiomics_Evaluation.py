# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 16:05:07 2022

@author: alber
"""

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
import tqdm
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.utils import resample

# for evaluating the model
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.metrics import roc_auc_score, auc, roc_curve, precision_recall_curve,plot_roc_curve
from scipy import interp

from random import randrange#bootstrapping
import SimpleITK as sitk

from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam, SGD
from torch.autograd import Variable

#%% Extract clinical data and outcome
path='C:/Users/alber/Bureau/Development/Data/Images_data/Radiomics_McMedHacks/'

df_cln = pd.read_excel(path+'Clinical_data_modified_2.xlsx', sheet_name = 'CHUM')
device = torch.device("cpu")
pth_file_name="C:/Users/alber/Bureau/Development/DeepLearning/training_results/cluster/radiomics3dCNN_0712.pth"
pth=pth_file_name

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

    """
    # Also GTV contours for later use
    image = sitk.ReadImage(path+f'warpedGtv/HN-CHUM-{p}-gtv-refct.mha') 
    image = sitk.GetArrayFromImage(image)
    #image = image[70:185, 30:170, 40:230]
    X_gtv[ip, :, :, :] = image
    """
 
print()
        
print(f"{sum(y)}/{len(y)} patients are positive")



#%%%
  




#%%% Data preprocessing
X_cln=X_cln/X_cln.max()
X_dos=X_dos/X_dos.max()
X_cts=X_cts/X_cts.max()
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
bs = 4



test_set = TensorDataset(X_cts_test, X_dos_test, X_cln_test, y_test)
test_loader = DataLoader(test_set, batch_size = bs )



CNN3D = RadiomicsCNN(dim1,dim2,dim3,n_cln)
state_dict = torch.load(pth, map_location=device)
CNN3D.load_state_dict(state_dict)
CNN3D.eval()
CNN3D.to(device)



#%% validation. to do save model and use another code


y_true = torch.Tensor([]).to(device)
y_pred = torch.Tensor([]).to(device)
batch_accuracy = 0
for batch, (x_test, x_ct_test, x_clinical_test, y_test) in enumerate(test_loader):
    pred   = CNN3D(x_test.to(device), x_ct_test.to(device), x_clinical_test.to(device))        
    y_true = torch.cat((y_true,y_test.to(device)))
    y_pred = torch.cat((y_pred,pred))


# Calculate AUC
auc = roc_auc_score(y_true.detach().numpy().astype(int), y_pred.detach().numpy())
#auc = roc_auc_score(y_true.cpu().detach().numpy().astype(int), y_pred.cpu().detach().numpy())

print(f"AUC : {auc}")

print(y_true,y_pred)
# Plot ROC Curve
fpr, tpr, thr = roc_curve(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
plt.plot(fpr,tpr, 'b', label = f"AUC = {auc:0.3f}")
plt.plot([0, 1], [0, 1],'r--', label = 'Chance')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.legend()
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

def to_labels(pos_probs, threshold):
        return (pos_probs >= threshold).astype('int')
        
thr = .2
CM = confusion_matrix(y_true.detach().numpy().astype(int), to_labels(y_pred.detach().numpy(), thr).astype(int))
#CM = confusion_matrix(y_true.cpu().detach().numpy().astype(int), to_labels(y_pred.cpu().detach().numpy(), thr).astype(int))

df_cm = pd.DataFrame(CM, index = [i for i in "PN"],
                  columns = [i for i in "PN"])
sns.heatmap(df_cm, annot=True, cmap = 'Blues')
plt.show()
#%%% Precision recall curve


prec, recall, thr2 = precision_recall_curve(y_true.detach().numpy().astype(int), to_labels(y_pred.detach().numpy(), thr).astype(int))
#prec, recall, thr2 = precision_recall_curve(y_true.cpu().detach().numpy().astype(int), to_labels(y_pred.cpu().detach().numpy(), thr).astype(int))

plt.plot(recall, prec, 'orange', lw=2)
no_skill = len(y_true[y_true==1]) / len(y_true)
plt.plot([0, 1], [no_skill, no_skill], '--')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

#%% prediction using DVH
#The dose-volume histogram (DVH) is a 2-D representation of the 3D dose maps
#It is used to evaluate plans and assess toxicity in organs-at-risk
#Although it ignores spatial information, the use of DVH saves significant computational cost, which could allow for better tuning and cross-validation of the model
#The DVH can be processed by the CNN as a signal/1D array
#Typically, the GTV dose is removed since DVH is calculated for different organs-at-risk (OAR)
#OARs are site-dependent


