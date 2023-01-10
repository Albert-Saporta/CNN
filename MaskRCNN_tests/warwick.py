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
local_test = "C:/Users/alber/Bureau/Development/Data/Images_data/Warwick_QU/test"
cluster_test="/bigdata/casus/optima/Warwick_QU/test"

pth_name="maskrcnn"
pth_path_cluster="/bigdata/casus/optima/hemera_results/"+pth_name+"/"

root_train=local_train
root_test=cluster_test
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
        # print('img_path', img_path)
        mask_path = os.path.join(self.root, "mask", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        print("type",type(mask),mask)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        print("shapeeee",mask)

        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        #print("shapeeee",obj_ids)

        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        plt.imshow(mask)
        plt.show()
        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]
        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)

        # print(num_objs)
        boxes = []
        for i in range(num_objs):
          pos = np.where(masks[i])
          #print(num_objs)
          #print("pos",pos[1].shape)

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
          #print("boxes",boxes)

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
        #print(img.shape,masks.shape)
        target = {}
        print(masks.shape,img.shape)

        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        #target["labels"] = labels # Not sure if this is needed
        #print("labels",target["labels"].shape)

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
#%%
dataset_train = WarwickCellDataset(root_train, transforms=torchvision.transforms.ToTensor()) # get_transform(train=True)
#print(dataset_train[3][1])
data_loader_train = DataLoader(dataset_train, batch_size=4, shuffle=True,collate_fn=lambda x:list(zip(*x)))
#%% visu
#images,labels=next(iter(data_loader_train))
#view(images=images,labels=labels,n=2,std=1,mean=0)
#%%
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

#%% train
loss_list = []
n_epochs = 100
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

#model_nr = latest_model() + 1
save_path = pth_path_cluster+"maskrcnn.pth"#"C:/Users/alber/Bureau/Development/"
torch.save(model.state_dict(), save_path)

#%% evaluation

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