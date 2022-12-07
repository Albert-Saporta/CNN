# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 16:15:50 2022

@author: alber
"""

import pydicom
from pydicom.tag import Tag

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='browser'


#%% Load data

data_path='C:/Users/alber/Bureau/Development/Data/Images_data/manifest-1599750808610/Pancreas-CT/PANCREAS_0001/11-24-2015-PANCREAS0001-Pancreas-18957/Pancreas-99667/'#1-001.dcm'


list_dcm_path = glob.glob(f"{data_path}*.dcm")
print("My list contains {} elements. These are the 5 first: ".format(len(list_dcm_path)))
print(list_dcm_path[:5])

my_dcm_file = pydicom.dcmread(list_dcm_path[0])
print(my_dcm_file)

#%%% Getting Metadata by reading tags

# all of these are equivalent
t1 = Tag(0x00100010) 
t2 = Tag(0x10,0x10)
t3 = Tag((0x10, 0x10))
t4 = Tag("PatientName")
print(t1)
print(type(t1))
print(t1==t2, t1==t3, t1==t4)

print(my_dcm_file[0x10,0x10]) # Returns the whole DataElement
print(my_dcm_file.PatientName) # Returns the value
print(my_dcm_file[0x10,0x10].value) # Returns the value as well

print("As 'my_dcm_file[0x10,0x10]' is a DataElement, we can look at all its components:")
print("Value Multiplicity : ", my_dcm_file[0x10,0x10].VM)
print("Value Representation : ", my_dcm_file[0x10,0x10].VR)
print("Tag : ", my_dcm_file[0x10,0x10].tag)
print("Value : ", my_dcm_file[0x10,0x10].value)

print(type(my_dcm_file[0x10,0x10]))
print("================")
print(my_dcm_file[0x10,0x10])





print("Storage object:", my_dcm_file.SOPClassUID.name)
print("Image modality :", my_dcm_file.Modality)
print("Image ID :",my_dcm_file.SOPInstanceUID.name)
print("Treatment site :",my_dcm_file.BodyPartExamined)

print("Image position :",my_dcm_file.ImagePositionPatient)
print("Image height :",my_dcm_file.Rows)                
print("Image width :",my_dcm_file.Columns)                       
print("Image spacings :",my_dcm_file.PixelSpacing)  

keywords = my_dcm_file.dir()

dic = dict()

for key in keywords:
    dic[key] = my_dcm_file[key]
    
print(keywords)

#%%% Images!!

print(my_dcm_file.PixelData)
print("================")
print(type(my_dcm_file.PixelData))
print("================")
print(my_dcm_file[0x7fe0, 0x0010])

print(type(my_dcm_file.pixel_array))
print("================")
print(my_dcm_file.pixel_array)
print("================")
print(np.all(my_dcm_file.pixel_array==my_dcm_file[0x7fe0, 0x0010].value))

print(my_dcm_file[0x7fe0, 0x0010].value)

print("To rescale your image from pixel values to Hounsfield Unit you can use the intercept {inter} and the slope {slo} present in the DICOM file.".format(
    inter=my_dcm_file.RescaleIntercept,slo=my_dcm_file.RescaleSlope))

#%%% read image

plt.figure(figsize=(15,10))
plt.imshow(my_dcm_file.pixel_array, cmap=plt.cm.bone)
plt.colorbar()
plt.title("My first opened CT image ")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.show()

temp = my_dcm_file.pixel_array.copy()
temp[:200] = 18
plt.imshow(temp, cmap=plt.cm.bone)
plt.colorbar()
plt.title("My first opened CT image ")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.show()

#%% CT volumes

#%%% Functions

class CT_volume():

    """
    DICOM CT dataset class.

    Takes as input the folder where dicom images of a patient are stored and creates a volume information.

    """

    def __init__(self, patient_folder):
        """Constructor."""
        self.ct_folder = patient_folder
        self._gather_data()

    def _get_file_list(self):
        file_list = os.listdir(self.ct_folder)
        return file_list

    def _gather_data(self):

        slice_ordering = []
        for ct_filename in self._get_file_list():
          ct_filepath = os.path.join(self.ct_folder, ct_filename)
          
          if ct_filepath[-4:]==".dcm":
            ct_file = pydicom.read_file(ct_filepath, force=True)
            if ct_file.SOPClassUID.name == "CT Image Storage": # dcm files can contains other things than CT images.
                slice_ordering.append((ct_filename, ct_file.ImagePositionPatient[2], ct_file.SOPInstanceUID))
          else:
            continue


        slice_ordering.sort(key=lambda x: x[1])
        self.files = [x[0] for x in slice_ordering]
        self.uids = [x[2] for x in slice_ordering]

        self.sample_image = pydicom.read_file(os.path.join(self.ct_folder,
                                      slice_ordering[0][0]), force=True)
        
        first_z = float(slice_ordering[0][1])
        slice_thickness = slice_ordering[1][1] - first_z

        # With the volume corner coordinates and the spacing, you know the position of every single voxel of the volume.
        self.volume_position = [float(self.sample_image.ImagePositionPatient[0]),
                                float(self.sample_image.ImagePositionPatient[1]),
                                first_z]

        self.volume_spacing = [float(self.sample_image.PixelSpacing[0]),
                              float(self.sample_image.PixelSpacing[1]),
                              slice_thickness]

        self.volume_shape = [self.sample_image.Columns,
                            self.sample_image.Rows,
                            len(slice_ordering)]

        self.slice_coordinates = [round(float(z[1]), 4) for z in slice_ordering]

    def get_patient_volume(self):
        """Builds a volume from the CT images."""
        ct_grid = np.zeros((self.volume_shape[2], 
                            self.volume_shape[1], 
                            self.volume_shape[0]) 
        , dtype=np.int16) # Data Type is really important here. How do you want to represent your data.

        for slice_num in range(self.volume_shape[2]):
            ctfile_path = os.path.join(self.ct_folder, self.files[slice_num])
            ct_dicom = pydicom.read_file(ctfile_path, force=True)
            ct_grid[slice_num] = ct_dicom.pixel_array


        return ct_grid
    
#%%%

ct_folder = os.path.dirname(list_dcm_path[0])
print(" Creating CT object from {} ".format(ct_folder))
my_ct_obj = CT_volume(ct_folder)
print("My CT volume is of size : ", my_ct_obj.volume_shape)

my_ct_volume = my_ct_obj.get_patient_volume()

rescale_slope = my_ct_obj.sample_image.RescaleSlope
rescale_intercept = my_ct_obj.sample_image.RescaleIntercept

scale_ct_volume = my_ct_volume * rescale_slope + rescale_intercept
print("Base shape : " , scale_ct_volume.shape)
cropped_ct_vol = scale_ct_volume[5:30,50:450,100:350 ]
print("Cropped shape : " , cropped_ct_vol.shape)

new_array = zoom(cropped_ct_vol, (0.5, 0.5, 0.5))
print("Resampled array for plotting has shape {} ". format(new_array.shape))

X, Y, Z = np.mgrid[0:new_array.shape[0]:12j,
                   0:new_array.shape[1]:200j,
                   0:new_array.shape[2]:125j
                   ]

fig = go.Figure(data=go.Volume(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value= new_array.flatten(),
    isomin=np.ndarray.min(new_array),
    isomax=np.ndarray.max(new_array),
    cmin=np.ndarray.min(new_array),
    cmax=np.ndarray.max(new_array),
    opacity=0.1,  
    surface_count=30,
    ))
fig.show()
