# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 09:08:27 2023

@author: alber
"""

from os import listdir
from os.path import isfile, isdir, join
import shutil
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt

local_path="C:/Users/alber/Bureau/Development/Data/Images_data/cell_nucleus/"
cluster_path="/bigdata/casus/optima/data/cell_nucleus/"
path=local_path

inpath = path+"subset/"  # the train folder download from kaggle
outpath = path+"train/"  # the folder putting all nuclei image
images_name = listdir(inpath)
for f in images_name:
    masks = listdir(inpath + f + "/masks/")
    #shutil.copyfile(inpath + f + "/images/" + masks[0], outpath + masks[0])
    #print(masks)
    #shutil.copyfile(inpath + f + "/images/" + image[0], outpath + image[0])

for i, im_name in enumerate(images_name):
    mask_folder = listdir(inpath + im_name + "/masks/")
    mask_list=[]
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool_)
    for mask in mask_folder:
        print(mask)
        t_image = cv2.imread(inpath + im_name + "/masks/" + mask, 0)
        t_image=cv2.cvtColor(t_image,cv2.COLOR_GRAY2RGB)
        mask = np.maximum(mask, t_image)
        plt.imshow(mask)

"""
print(len(mask_list))
plt.imshow(cv2.cvtColor(mask_list))
    for k in range(len(mask_list)):
        test=mask_list[0]
        array=np.add(test, mask_list[k])
        test
        #Labeled_mask,ref_num_features = ndimage.label(masktt)

    print(mask_list.shape)
"""