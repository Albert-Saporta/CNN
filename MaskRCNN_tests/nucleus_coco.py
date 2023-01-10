# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 09:43:25 2023

@author: alber
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 16:19:51 2023

@author: alber
"""
#https://www.kaggle.com/code/tjac718/semantic-segmentation-of-nuclei-pytorch
import torch.optim as optim
from torch.autograd import Variable
from time import time
from natsort import natsorted
import os
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models
import torchvision.transforms as T
from scipy.ndimage import label,binary_closing,find_objects
from torchsummary import summary
import skimage.io
from pycocotools.coco import COCO

import torch.optim as optim
from torch.autograd import Variable
from time import time
from natsort import natsorted
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
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline 
import matplotlib.patches as mpatches
from matplotlib import patches
from tqdm import tqdm

import random
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from scipy import ndimage
from skimage.io import imread, imshow
from skimage.transform import resize
#from skimage.morphology import label

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


#%% functions

n_epochs = 2#00
device = torch.device('cuda')


class myOwnDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))
    def __getitem__(self, index):
        # Own coco file
        
        coco = self.coco

        # Image ID
        img_id = self.ids[index]

        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = Image.open(os.path.join(self.root, path))

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Labels (In my case, I only one class: target class or background)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        # Tensorise img_id
        img_id = torch.tensor([img_id])
        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        masks = [self.coco.annToMask(ann) for ann in ann_ids]
        print(masks.shape)

        masks = torch.as_tensor(masks, dtype=torch.uint8)

        # Annotation is in dictionary format
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        #target["image_id"] = img_id
        #target["area"] = areas
        #target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.ids)
    
def get_transform():
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(custom_transforms)

train_data_dir = "C:/Users/alber/Bureau/Development/Data/Images_data/cell_nucleus/train/"
train_coco = "C:/Users/alber/Bureau/Development/Data/Images_data/cell_nucleus/nucleus_cocoformat.json"

# create own Dataset
my_dataset = myOwnDataset(root=train_data_dir,
                          annotation=train_coco,
                          transforms=get_transform()
                          )

# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))


# own DataLoader
data_loader_train = torch.utils.data.DataLoader(my_dataset,
                                          batch_size=2,
                                          shuffle=True,
                                          #num_workers=1,
                                          collate_fn=collate_fn)



num_classes = 2
# load an instance segmentation model pre-trained pre-trained on COCO
model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights="MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1")
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
# now get the number of input features for the mask classifier
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 20#256
# and replace the mask predictor with a new one
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                    hidden_layer,
                                                    num_classes)
model=model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

loss_list = []

#%% Training

model.train()
for epoch in range(n_epochs):
    loss_epoch = []
    iteration=1
    train_loop=tqdm(data_loader_train)
    for images,targets in train_loop:
        train_loop.set_description(f'Epoch {epoch+1}/{n_epochs}')
        #print(images[0].shape,targets[0]["labels"].shape)

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        #print(targets)
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

        iteration+=1
    loss_epoch_mean = np.mean(loss_epoch) 
    loss_list.append(loss_epoch_mean) 
    # loss_list.append(loss_epoch_mean)    
    print("Average loss for epoch = {:.4f} ".format(loss_epoch_mean))
