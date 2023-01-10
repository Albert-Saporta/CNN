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

#%% param
n_epochs = 2#00
device = torch.device('cuda')
image_size=1024#imgage_size
#%% path

local_train = "C:/Users/alber/Bureau/Development/Data/Images_data/cell_nucleus/stage1_train/"
cluster_train="/bigdata/casus/optima/data/cell_nucleus/stage1_train/"
local_test = "C:/Users/alber/Bureau/Development/Data/Images_data/cell_nucleus/stage1_test/"
cluster_test="/bigdata/casus/optima/data/cell_nucleus/stage1_test/"

pth_name="maskrcnn_nucleus"
pth_path_cluster="/bigdata/casus/optima/hemera_results/"+pth_name+"/"


TRAIN_PATH=local_train

train_files = next(os.walk(TRAIN_PATH))[1]
print(train_files)

#X_train = np.zeros((len(train_files), image_size, image_size, 3), dtype = np.uint8)
#Y_train = np.zeros((len(train_files), image_size, image_size, 3), dtype = np.int32)



print('Getting training data...')

def load_image_mask():
    for n, id_ in tqdm(enumerate(train_files), total = len(train_files)):
        img_path = TRAIN_PATH + id_ + '/images/' + id_ + '.png'
        img = imread(img_path)#[:,:,:3]
        #img = resize(img, (image_size, image_size), mode='constant', preserve_range=True)
        print(img.shape)

        #X_train[n] = img
        
        masks_path = TRAIN_PATH + id_ + '/masks/'
        mask = np.zeros((image_size, image_size, 3))
        mask_images = next(os.walk(masks_path))[2]
        mask_ = []
        for mask_id in mask_images:
            mask_path = masks_path + mask_id
            mask_test = skimage.io.imread(mask_path).astype(np.bool_)
            #print(mask_test.shape)
            mask_.append(mask_test)
            """
            obj_ids = np.unique(mask_)
            #m = mask_[:, :, np.where(255)[0]]
            m = np.sum(mask_ * np.arange(1, mask_.shape[-1] + 1), -1)
     
            mask_ = np.expand_dims(resize(mask_, (image_size, image_size), mode='constant', preserve_range=True), axis=-1)
            mask = np.maximum(mask, mask_)
            """
        mask_ = np.stack(mask_, axis=-1)
        print(mask.shape)
        _idx = np.sum(mask_, axis=(0, 1)) > 0
        masktt = mask_[:, :, _idx]
        #print("test3",img.shape)
    
        Labeled_mask,ref_num_features = ndimage.label(masktt)
        #print("test4",Labeled_mask.shape)
        return img,Labeled_mask
X_train,Y_train=load_image_mask()     
X_train=X_train.reshape(X_train.shape[2],X_train.shape[0],X_train.shape[1])
Y_train=Y_train.reshape(Y_train.shape[2],Y_train.shape[0],Y_train.shape[1])

#%% Functions
    
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
        plt.savefig(pth_path_cluster+'figure/images_test.pdf',format='pdf')
        
def view_mask(targets, output, n=2, cmap='Greys'):
    figure = plt.figure(figsize=(15,10))
    for i in range(n):
      # plot target (true) masks
      target_im = targets[i]['masks'][0].cpu().detach().numpy()
      for k in range(len(targets[i]['masks'])):
        target_im2 = targets[i]['masks'][k].cpu().detach().numpy()
        target_im2[target_im2>0.5] = 1
        target_im2[target_im2<0.5] = 0
        target_im = target_im+target_im2

      target_im[target_im>0.5] = 1
      target_im[target_im<0.5] = 0
      ax = figure.add_subplot(2,2, i+1)
      ax.imshow(target_im, cmap=cmap)
      # Plot output (predicted) masks
      output_im = output[i]['masks'][0][0, :, :].cpu().detach().numpy()
      for k in range(len(output[i]['masks'])):
        output_im2 = output[i]['masks'][k][0, :, :].cpu().detach().numpy()
        output_im2[output_im2>0.5] = 1
        output_im2[output_im2<0.5] = 0
        output_im = output_im+output_im2

      output_im[output_im>0.5] = 1
      output_im[output_im<0.5] = 0
      ax = figure.add_subplot(2,2, i+3)
      ax.imshow(output_im, cmap=cmap)
    plt.savefig(pth_path_cluster+'figure/mask_test.pdf',format='pdf')

def IoU(y_real, y_pred):
  # Intersection over Union loss function
  intersection = y_real*y_pred
  #not_real = 1 - y_real
  #union = y_real + (not_real*y_pred)
  union = (y_real+y_pred)-(y_real*y_pred)
  return np.sum(intersection)/np.sum(union)

def dice_coef(y_real, y_pred, smooth=1):
  intersection = y_real*y_pred
  union = (y_real+y_pred)-(y_real*y_pred)
  return np.mean((2*intersection+smooth)/(union+smooth))

def confusion_matrix(y_true, y_pred):
  y_true= y_true.flatten()
  y_pred = y_pred.flatten()*2
  cm = y_true+y_pred
  cm = np.bincount(cm, minlength=4)
  tn, fp, fn, tp = cm
  return tp, fp, tn, fn

def get_f1_score(y_true, y_pred):
    """Return f1 score covering edge cases"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred)
    f1_score = (2 * tp) / ((2 * tp) + fp + fn)

    return f1_score   

def get_outputs(image, model, threshold):
    with torch.no_grad():
        # forward pass of the image through the modle
        outputs = model(image)
    
    # get all the scores
    scores = list(outputs[0]['scores'].detach().cpu().numpy())
    # index of those scores which are above a certain threshold
    thresholded_preds_inidices = [scores.index(i) for i in scores if i > threshold]
    thresholded_preds_count = len(thresholded_preds_inidices)
    # get the masks
    masks = (outputs[0]['masks']>0.5).squeeze().detach().cpu().numpy()
    # discard masks for objects which are below threshold
    masks = masks[:thresholded_preds_count]
    # get the bounding boxes, in (x1, y1), (x2, y2) format
    boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]  for i in outputs[0]['boxes'].detach().cpu()]
    # discard bounding boxes below threshold value
    boxes = boxes[:thresholded_preds_count]
    # get the classes labels
    labels = ''#[coco_names[i] for i in outputs[0]['labels']]
    return masks, boxes, labels

def draw_segmentation_map(image, masks, boxes, labels):
    alpha = 1 
    beta = 0.6 # transparency for the segmentation map
    gamma = 0 # scalar added to each sum
    for i in range(len(masks)):
        red_map = np.zeros_like(masks[i]).astype(np.uint8)
        green_map = np.zeros_like(masks[i]).astype(np.uint8)
        blue_map = np.zeros_like(masks[i]).astype(np.uint8)
        # apply a randon color mask to each object
        color = COLORS[random.randrange(0, len(COLORS))]
        red_map[masks[i] == 1], green_map[masks[i] == 1], blue_map[masks[i] == 1]  = color
        # combine all the masks into a single image
        segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
        #convert the original PIL image into NumPy format
        image = np.array(image)
        # convert from RGN to OpenCV BGR format
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # apply mask on the image
        cv2.addWeighted(image, alpha, segmentation_map, beta, gamma, image)
        # draw the bounding boxes around the objects
        cv2.rectangle(image, boxes[i][0], boxes[i][1], color=color, 
                      thickness=2)
        # put the label text above the objects
        cv2.putText(image , labels[i], (boxes[i][0][0], boxes[i][0][1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 
                    thickness=2, lineType=cv2.LINE_AA)
    
    return image


class Nuc_Seg():
    def __init__(self, images_np, masks_np):
        self.images_np = images_np
        self.masks_np = masks_np
    """
    def transform(self, image_np, mask_np):
        ToPILImage = transforms.ToPILImage()
        image = ToPILImage(image_np)
        mask = ToPILImage(mask_np.astype(np.int32))
        
        image = TF.pad(image, padding = 20, padding_mode = 'reflect')
        mask = TF.pad(mask, padding = 20, padding_mode = 'reflect')
        
        angle = random.uniform(-10, 10)
        width, height = image.size
        max_dx = 0.1 * width
        max_dy = 0.1 * height
        translations = (np.round(random.uniform(-max_dx, max_dx)), np.round(random.uniform(-max_dy, max_dy)))
        scale = random.uniform(0.8, 1.2)
        shear = random.uniform(-0.5, 0.5)
        image = TF.affine(image, angle = angle, translate = translations, scale = scale, shear = shear)
        mask = TF.affine(mask, angle = angle, translate = translations, scale = scale, shear = shear)
        
        image = TF.center_crop(image, (128, 128))
        mask = TF.center_crop(mask, (128, 128))
        
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        #return image, mask
    """
        
    def __len__(self):
        return len(self.images_np)
    
    def __getitem__(self,idx):
        #print("idx",idx)
        image_np = self.images_np#[idx]
        mask_np = self.masks_np#[idx]
        #plt.imshow(mask_np)
        #plt.show()
        #print("mask_np",mask_np)
        obj_ids = np.unique(mask_np)
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)
        print("num_objs",num_objs)
        masks = mask_np#== obj_ids[:, None, None]
        #print("num_objs",masks.shape)

        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            print("pos",pos[1].shape)
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


        #image, mask = self.transform(image_np, mask_np)
        image= image_np
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image = torch.as_tensor(image, dtype=torch.float64)
        #print(masks.shape,image.shape)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        return image, target    









def iou(pred, target, n_classes = 2):
    
    iou = []
    pred = pred.view(-1)
    target = target.view(-1)

    # Ignore IoU for background class ("0")
    for cls in range(1, n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
      pred_inds = pred == cls
      target_inds = target == cls
      intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()  # Cast to long to prevent overflows
      union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
    
      if union == 0:
        iou.append(float('nan'))  # If there is no ground truth, do not include in evaluation
      else:
        iou.append(float(intersection) / float(max(union, 1)))
     
    return sum(iou)

#%% model and dataloader



dataset_train = Nuc_Seg(X_train, Y_train)
#train_loader = DataLoader(train_dataset, batch_size = 16, shuffle = True)
"""
fig, axis = plt.subplots(2, 2)
axis[0][0].imshow(X_train[0].astype(np.uint8))
axis[0][1].imshow(np.squeeze(Y_train[0]).astype(np.uint8))
axis[1][0].imshow(X_val[0].astype(np.uint8))
axis[1][1].imshow(np.squeeze(Y_val[0]).astype(np.uint8))
"""

data_loader_train = DataLoader(dataset_train, batch_size=4, shuffle=True,collate_fn=lambda x:list(zip(*x)))


num_classes = 2
# load an instance segmentation model pre-trained pre-trained on COCO
model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights="MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1")
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
# now get the number of input features for the mask classifier
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
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
        """
        plt.figure()
        plt.plot(list(range(iteration)), loss_epoch)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.savefig(pth_path_cluster+f'Learning_Curves_.pdf',format='pdf')
        #plt.show()
        """
        iteration+=1
    loss_epoch_mean = np.mean(loss_epoch) 
    loss_list.append(loss_epoch_mean) 
    # loss_list.append(loss_epoch_mean)    
    print("Average loss for epoch = {:.4f} ".format(loss_epoch_mean))
    
#%% Evaluation
# Plot training loss
plt.figure()
plt.plot(list(range(n_epochs)), loss_list, label='traning loss')
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(pth_path_cluster+'figure/Learning_Curves.pdf',format='pdf')

dataset_test = WarwickCellDataset(root_test, transforms=torchvision.transforms.ToTensor()) # get_transform(train=True)
data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=2, shuffle=False,collate_fn=lambda x:list(zip(*x)))

images, targets=next(iter(data_loader_test))
images = list(image.to(device) for image in images)
targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

model=model.double()
model.eval()
output = model(images)

with torch.no_grad():
    view(images, output, 2)
#%%% scores
print(len(output[0]['boxes']))
print(len(output[0]['scores']))

print(output[0]['boxes'][0])

# torchvision.utils.make_grid(images[i])
for i in range(2):
  # out = output[i]['scores'].to('cpu')
  # out = out.detach().numpy()
  for j in range(len(output[i]['scores'])):
    if j < 0.7:
      output[i]['boxes'][j] = torch.Tensor([0,0,0,0])
      
view_mask(targets, output, n=2)

# Get IoU score for whole test set
IoU_scores_list = []
dice_coef_scores_list = []
f1_scores_list = []
skipped = 0
for images,targets in tqdm(data_loader_test):
  images = list(image.to(device) for image in images)
  targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

  model=model.double()
  model.eval()
  output = model(images)
  # print(targets)
  target_im = targets[0]['masks'][0].cpu().detach().numpy()
  for k in range(len(targets[0]['masks'])):
    target_im2 = targets[0]['masks'][k].cpu().detach().numpy()
    target_im2[target_im2>0.5] = 1
    target_im2[target_im2<0.5] = 0
    target_im = target_im+target_im2

  target_im[target_im>0.5] = 1
  target_im[target_im<0.5] = 0
  target_im = target_im.astype('int64')
  
  # Plot output (predicted) masks
  output_im = output[0]['masks'][0][0, :, :].cpu().detach().numpy()
  for k in range(len(output[0]['masks'])):
    output_im2 = output[0]['masks'][k][0, :, :].cpu().detach().numpy()
    output_im2[output_im2>0.5] = 1
    output_im2[output_im2<0.5] = 0
    output_im = output_im+output_im2

  output_im[output_im>0.5] = 1
  output_im[output_im<0.5] = 0
  output_im = output_im.astype('int64')

  if target_im.shape != output_im.shape:
    skipped+=1
    continue
  
  dice_coef_score = dice_coef(y_real=target_im, y_pred=output_im)
  dice_coef_scores_list.append(dice_coef_score)
  IoU_score = IoU(y_real=target_im, y_pred=output_im) 
  IoU_scores_list.append(IoU_score)
  f1_score = get_f1_score(target_im, output_im)
  f1_scores_list.append(f1_score)

print('mean IoU score for test set:', np.mean(IoU_scores_list))
print('mean Dice Coefficient score for test set:', np.mean(dice_coef_scores_list))
print('mean f1 score for test set:', np.mean(f1_scores_list))

print('mean IoU score for test set:', np.mean(IoU_scores_list))

for i in range(2):
  # plot target (true) masks
  target_im = targets[i]['masks'][0].cpu().detach().numpy()
  for k in range(len(targets[i]['masks'])):
    target_im2 = targets[i]['masks'][k].cpu().detach().numpy()
    target_im2[target_im2>0.5] = 1
    target_im2[target_im2<0.5] = 0
    target_im = target_im+target_im2

  target_im[target_im>0.5] = 1
  target_im[target_im<0.5] = 0
  # Plot output (predicted) masks
  output_im = output[i]['masks'][0][0, :, :].cpu().detach().numpy()
  for k in range(len(output[i]['masks'])):
    output_im2 = output[i]['masks'][k][0, :, :].cpu().detach().numpy()
    output_im2[output_im2>0.5] = 1
    output_im2[output_im2<0.5] = 0
    output_im = output_im+output_im2

  output_im[output_im>0.5] = 1
  output_im[output_im<0.5] = 0

IoU(y_real=target_im, y_pred=output_im) 

len(output[0]['masks'])
plt.figure()
im = output[0]['masks'][0][0, :, :].cpu().detach().numpy()
# im2 = outputs[0]['masks'][1][0, :, :].cpu().detach().numpy()
it = 0
for i in range(len(output[0]['masks'])):
  im2 = output[0]['masks'][i][0, :, :].cpu().detach().numpy()
  im2[im2>0.5] = 1
  im2[im2<0.5] = 0
  im = im+im2
  it+=1
print(it)
# im_new = np.concatenate((im, im2)) 
im[im>0.5] = 1
im[im<0.5] = 0
plt.savefig(pth_path_cluster+'figure/masks_finaux.pdf',format='pdf')
#plt.imshow(im, cmap='Greys')
# outputs[0]['masks']