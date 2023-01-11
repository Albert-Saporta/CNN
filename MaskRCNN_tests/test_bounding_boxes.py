# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 16:53:18 2023

@author: alber
"""

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

from torchsummary import summary
import random

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