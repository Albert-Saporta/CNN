# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 13:16:04 2022

@author: alber
"""

import SimpleITK as sitk
import glob
import os
import json
from skimage.morphology import disk, dilation
import warnings
import matplotlib.pyplot as plt 

#%%

def DicomRead(Input_path):
    """
    Reading Dicom files from Input path to dicom image series
    :param Input_path: path to dicom folder which contains dicom series
    :return: 3D Array of the dicom image
    """
    print("Reading Dicom file from:", Input_path )
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(Input_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    array=sitk.GetArrayFromImage(image)
    return array

Input_path="C:/Users/alber/Bureau/Development/Data/Images_data/CT/manifest-1599750808610/Pancreas-CT/PANCREAS_0001/11-24-2015-PANCREAS0001-Pancreas-18957/Pancreas-99667/"
image=DicomRead(Input_path)
#print(image.shape)
#sitk.WriteImage(image, os.path.join("C:/Users/alber/Bureau/Development/Data", "sample" + ".mhd") )
plt.imshow(image[60,:,:])