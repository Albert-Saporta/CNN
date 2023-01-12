
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as Ft

import torchvision
from torchvision import models
import torchvision.transforms as T
from torchvision.utils import draw_bounding_boxes,draw_segmentation_masks

from torchvision.ops import masks_to_boxes
from torchvision.io import read_image

from d2l import torch as d2l
import cv2
import colorsys
from skimage import morphology

from torchsummary import summary
import random
from scipy.ndimage import label,binary_closing,find_objects
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
from matplotlib.patches import Polygon
from skimage.measure import find_contours

import matplotlib.patches as mpatches
from matplotlib import patches
from tqdm import tqdm
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
#%% paths
local_train = "C:/Users/alber/Bureau/Development/Data/Images_data/cell_nucleus/train"
#local_train = "C:/Users/alber/Bureau/Development/Data/Images_data/Warwick_QU/train"

pth_name="maskrcnn_nucleus"
pth_path_cluster="/bigdata/casus/optima/hemera_results/"+pth_name+"/"

root_train=local_train
#%% functions
#%%% visu
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

def draw_segmentation_map(im, boxes, labels):
    color=(255, 0, 0)
    image = np.array(im)
    print(boxes.shape)

    alpha = 1 
    beta = 0.6 # transparency for the segmentation map
    gamma = 0 # scalar added to each sum
    for i in range(boxes.shape[0]):
        start_point=(boxes[i][0],boxes[i][1])
        end_point =(boxes[i][2],boxes[i][3])
        #print(start_point,end_point)
        # draw the bounding boxes around the objects
        cv2.rectangle(image,start_point ,end_point , color, thickness=2)
        # put the label text above the objects
        """
        cv2.putText(image , labels, (boxes[i][0][0], boxes[i][0][1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 
                    thickness=2, lineType=cv2.LINE_AA)
        """

    return image


#%%% dataset     
class NucleusCellDataset(object):
    def __init__(self, root, transforms=None): # transforms
        self.root = root
        # self.transforms = transforms
        self.transforms=[]
        if transforms!=None:
          self.transforms.append(transforms)
        self.imgs = list(natsorted(os.listdir(os.path.join(root, "image"))))
        
        self.masks = list(natsorted(os.listdir(os.path.join(root, "mask"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "image", self.imgs[idx])
        mask_path = os.path.join(self.root, "mask", self.masks[idx])
        print(img_path,mask_path)
        img = Image.open(img_path)#.convert("RGB")
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask)

        mask[mask>0]=1

        mask=morphology.remove_small_objects(mask==1, min_size=10, connectivity=True)
        #♠mask=binary_closing(mask,iterations=3)
        mask,ref_num_features = label(mask)

        plt.imshow(mask)
        plt.show()
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]
        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        #print(obj_ids)
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
            boxes.append([xmin, ymin, xmax, ymax])

        maskk_test=draw_segmentation_map(mask,  np.array(boxes), "nucleus")
        plt.imshow(maskk_test)
        plt.show()
        img_test=draw_segmentation_map(img,  np.array(boxes), "nucleus")
        plt.imshow(img_test)
        plt.show()
        #target["labels"] = labels # Not sure if this is needed
        #print("labels",target["labels"].shape)
        return boxes
    def __len__(self):
        return len(self.imgs)
#%% test dataset
dataset_train = NucleusCellDataset(root_train, transforms=torchvision.transforms.ToTensor()) # get_transform(train=True)
data_loader_train = DataLoader(dataset_train, batch_size=4, shuffle=True,collate_fn=lambda x:list(zip(*x)),pin_memory=True)
images,labels=next(iter(data_loader_train))


#%% tests 1 image
"""
local_train = "C:/Users/alber/Bureau/Development/Data/Images_data/cell_nucleus/train"
image_path=local_train+"/image/"+"00ae65c1c6631ae6f2be1a449902976e6eb8483bf6b0740d00530220832c6d3e.png"
mask_path=local_train+"/mask/"+"00ae65c1c6631ae6f2be1a449902976e6eb8483bf6b0740d00530220832c6d3e.jpg"
local_train = "C:/Users/alber/Bureau/"
image_path=local_train+"train_1.png"
mask_path=local_train+"train_1_anno.png"

image = read_image(image_path)#[[0,1,3],:,:]
mask = read_image(mask_path)
#mask=torchvision.transforms.transforms.Grayscale(num_output_channels=1)(mask)
print(mask.size(),image.size())

obj_ids = torch.unique(mask)

# first id is the background, so remove it.
obj_ids = obj_ids[1:]

# split the color-encoded mask into a set of boolean masks.
# Note that this snippet would work as well if the masks were float values instead of ints.
masks = mask == obj_ids[:, None, None]
print(masks.size())

boxes=masks_to_boxes(masks)

imgtest=draw_bounding_boxes(image,boxes)
imgtest = torchvision.transforms.ToPILImage()(imgtest)
plt.imshow(imgtest)
plt.show()
"""