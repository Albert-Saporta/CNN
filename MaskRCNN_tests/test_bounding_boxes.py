
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
import torch
import torchvision
import cv2
import argparse
from PIL import Image
from torchvision.transforms import transforms as transforms
import numpy as np
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
from torchvision.transforms import transforms as transforms

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
def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors
def draw_segmentation_map(im, boxes, labels):
    
    image = np.array(im)
    N=boxes.shape[0]

    alpha = 1 
    beta = 0.6 # transparency for the segmentation map
    gamma = 0 # scalar added to each sum
    colors=random_colors(N)
    print(colors)
    for i in range(N):
        color=colors[i]
        start_point=(boxes[i][0],boxes[i][1])
        end_point =(boxes[i][2],boxes[i][3])
        #print(start_point,end_point)
        # draw the bounding boxes around the objects
        cv2.rectangle(image,start_point ,end_point , color, thickness=2)
        # put the label text above the objects
        cv2.putText(image , labels, (boxes[i][0], boxes[i][1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 
                    thickness=2, lineType=cv2.LINE_AA)

    return image

def draw_mask_box_labels(image, masks, boxes, labels):
    alpha = 1 
    beta = 0.6 # transparency for the segmentation map
    gamma = 0 # scalar added to each sum
    COLORS = np.random.uniform(0, 255, size=(2, 3))

    for i in range(len(masks)):
        red_map = np.zeros_like(masks[i]).astype(np.uint8)
        green_map = np.zeros_like(masks[i]).astype(np.uint8)
        blue_map = np.zeros_like(masks[i]).astype(np.uint8)
        # apply a randon color mask to each object
        color = COLORS[random.randrange(0, len(COLORS))]
        red_map[masks[i] == 1], green_map[masks[i] == 1], blue_map[masks[i] == 1]  = color
        # combine all the masks into a single image
        segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
        print(segmentation_map.shape)
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
        cv2.putText(image , labels, (boxes[i][0][0], boxes[i][0][1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 
                    thickness=2, lineType=cv2.LINE_AA)
    
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
"""
dataset_train = NucleusCellDataset(root_train, transforms=torchvision.transforms.ToTensor()) # get_transform(train=True)
data_loader_train = DataLoader(dataset_train, batch_size=4, shuffle=True,collate_fn=lambda x:list(zip(*x)),pin_memory=True)
images,labels=next(iter(data_loader_train))
"""

#%% tests 1 image

local_train = "C:/Users/alber/Bureau/Development/Data/Images_data/cell_nucleus/train"
image_path=local_train+"/image/"+"00ae65c1c6631ae6f2be1a449902976e6eb8483bf6b0740d00530220832c6d3e.png"
mask_path=local_train+"/mask/"+"00ae65c1c6631ae6f2be1a449902976e6eb8483bf6b0740d00530220832c6d3e.jpg"
local_train = "C:/Users/alber/Bureau/"
#image_path=local_train+"train_1.png"
#mask_path=local_train+"train_1_anno.png"

pth_path="C:/Users/alber/Bureau/Development/DeepLearning/training_results/cluster/maskrcnn/nucleus/12_01/test/test.pth"
device = torch.device('cuda')

#%%% Model
num_classes = 2
# load an instance segmentation model pre-trained pre-trained on COCO
model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=None)

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
# now get the number of input features for the mask classifier
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
# and replace the mask predictor with a new one
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,256, num_classes)

#model=model.to(device)
# load the modle on to the computation device and set to eval mode
model.load_state_dict(torch.load(pth_path))
#model=model.double()
model.to(device).eval()

# transform to convert the image to tensor
transform = transforms.Compose([transforms.ToTensor()])

image = Image.open(image_path).convert('RGB')
orig_image = image.copy()
# transform the image
image = transform(image)
image = image.unsqueeze(0).to(device)
# add a batch dimension
masks, boxes, labels = get_outputs(image, model, 0.1)
print("test",labels)#res=draw_segmentation_map(orig_image,boxes , "nucleus")
result = draw_mask_box_labels(orig_image, masks, boxes, "nucleus")
# visualize the image
cv2.imshow('Segmented image', result)
cv2.waitKey(0)