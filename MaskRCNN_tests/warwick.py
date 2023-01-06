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
cluster_train="/bigdata/casus/optima/Warwick_QU/train"

pth_name="maskrcnn"
pth_path_cluster="/bigdata/casus/optima/hemera_results/"+pth_name+"/"

root_train=cluster_train
device = torch.device('cuda')
#%% function
class WarwickCellDataset(object):
    def __init__(self, root, transforms=None): # transforms
        self.root = root
        # self.transforms = transforms
        self.transforms=[]
        if transforms!=None:
          self.transforms.append(transforms)
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(natsorted(os.listdir(os.path.join(root, "image"))))
        print('imgs file names:', self.imgs)
        self.masks = list(natsorted(os.listdir(os.path.join(root, "mask"))))
        #self.masks = list(natsorted(os.listdir(root+"/mask")))
        print('masks file names:', self.masks)

    def __getitem__(self, idx):
        # idx sometimes goes over the nr of training images, add logic to keep it lower
        if idx >= 80:
          idx = np.random.randint(80, size=1)[0]
        # print(idx)
        # load images ad masks
        # print('idx:', idx)
        img_path = os.path.join(self.root, "image", self.imgs[idx])
        # print('img_path', self.imgs[idx])
        mask_path = os.path.join(self.root, "mask", self.masks[idx])
        print(mask_path,idx)
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        # print(num_objs)
        boxes = []
        for i in range(num_objs):
          pos = np.where(masks[i])
          xmin = np.min(pos[1])
          xmax = np.max(pos[1])
          ymin = np.min(pos[0])
          ymax = np.max(pos[0])
          # Check if area is larger than a threshold
          A = abs((xmax-xmin) * (ymax-ymin)) 
          # print(A)
          if A < 5:
            print('Nr before deletion:', num_objs)
            obj_ids=np.delete(obj_ids, [i])
            # print('Area smaller than 5! Box coordinates:', [xmin, ymin, xmax, ymax])
            print('Nr after deletion:', len(obj_ids))
            continue
            # xmax=xmax+5 
            # ymax=ymax+5

          boxes.append([xmin, ymin, xmax, ymax])

        # print('nr boxes is equal to nr ids:', len(boxes)==len(obj_ids))
        num_objs = len(obj_ids)
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        for i in self.transforms:
          img = i(img)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        #target["labels"] = labels # Not sure if this is needed

        return img.double(), target

    def __len__(self):
        return len(self.imgs)
    
def view(images,labels,n=2,std=1,mean=0):
    figure = plt.figure(figsize=(15,10))
    images=list(images)
    labels=list(labels)
    for i in range(n):
        out=torchvision.utils.make_grid(images[i])
        inp=out.cpu().numpy().transpose((1,2,0))
        inp=np.array(std)*inp+np.array(mean)
        inp=np.clip(inp,0,1)  
        ax = figure.add_subplot(2,2, i + 1)
        ax.imshow(images[i].cpu().numpy().transpose((1,2,0)))
        l=labels[i]['boxes'].cpu().numpy()
        l[:,2]=l[:,2]-l[:,0]
        l[:,3]=l[:,3]-l[:,1]
        for j in range(len(l)):
            ax.add_patch(patches.Rectangle((l[j][0],l[j][1]),l[j][2],l[j][3],linewidth=1.5,edgecolor='r',facecolor='none')) 
#%%
dataset_train = WarwickCellDataset(root_train, transforms=torchvision.transforms.ToTensor()) # get_transform(train=True)
data_loader_train = DataLoader(dataset_train, batch_size=8, shuffle=True,collate_fn=lambda x:list(zip(*x)))
print(len(data_loader_train))
#%% visu
#images,labels=next(iter(data_loader_train))
#view(images=images,labels=labels,n=2,std=1,mean=0)
#%%
num_classes = 2
# load an instance segmentation model pre-trained pre-trained on COCO
model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="MaskRCNN_ResNet50_FPN_Weights.COCO_V1")

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# now get the number of input features for the mask classifier
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 2#256
# and replace the mask predictor with a new one
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                    hidden_layer,
                                                    num_classes)

model=model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

#%% train

# Perform training loop for n epochs
# Perform training loop for n epochs
loss_list = []
n_epochs = 1#10
model.train()
for epoch in range(n_epochs):
    loss_epoch = []
    iteration=1
    for images,targets in tqdm(data_loader_train):

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        #print(images,targets)
        optimizer.zero_grad()
        model=model.double()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()       
        optimizer.step()
        # print('loss:', losses.item())
        # loss_epoch.append(losses.item())
        loss_epoch.append(losses.item())
        # Plot loss every 10th iteration

        plt.plot(list(range(iteration)), loss_epoch)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.show()
        iteration+=1
    loss_epoch_mean = np.mean(loss_epoch) 
    loss_list.append(loss_epoch_mean) 
    # loss_list.append(loss_epoch_mean)    
    print("Average loss for epoch = {:.4f} ".format(loss_epoch_mean))

#model_nr = latest_model() + 1
save_path = pth_path_cluster+"maskrcnn.pth"#"C:/Users/alber/Bureau/Development/"
torch.save(model.state_dict(), save_path)