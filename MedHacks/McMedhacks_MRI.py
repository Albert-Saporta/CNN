# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 09:59:44 2022

@author: alber
"""

#https://colab.research.google.com/drive/1rObj9O6CCbzf3rpTFnE1Vp5BBxoIfP_2?usp=drive_open
#%% Download 
"""
!pip install torchio==0.18.70 --quiet
!pip install pandas --quiet
!pip install matplotlib --quiet
!pip install seaborn --quiet
!pip install scikit-image --quiet
!curl -s -o colormap.txt https://raw.githubusercontent.com/thenineteen/Semiology-Visualisation-Tool/master/slicer/Resources/Color/BrainAnatomyLabelsV3_0.txt
!curl -s -o slice_7t.jpg https://www.statnews.com/wp-content/uploads/2019/08/x961_unsmoothed_cropped-copy-768x553.jpg
!curl -s -o slice_histo.jpg https://bcf.technion.ac.il/wp-content/uploads/2018/05/Neta-histology-slice-626.jpg
!curl -s -o vhp.zip https://data.lhncbc.nlm.nih.gov/public/Visible-Human/Sample-Data/Six%20slices%20from%20the%20Visible%20Male.zip
!unzip -o vhp.zip > /dev/null
"""
import torch
import torchio as tio
import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import copy
import time
import pprint

import torch
import torchio as tio
import seaborn as sns; sns.set() # statistical data visualization


#%% Functions

#Function to display row of image slices
def show_slices(slices):
    fig, axes = plt.subplots(1, len(slices), figsize=(15,15))
    for idx, slice in enumerate(slices):
        axes[idx].imshow(slice.T, cmap="gray", origin="lower")

    axes[0].set_xlabel('Second dim voxel coords.', fontsize=12)
    axes[0].set_ylabel('Third dim voxel coords', fontsize=12)
    axes[0].set_title('First dimension (i), slice {}'.format(i), fontsize=15)

    axes[1].set_xlabel('First dim voxel coords.', fontsize=12)
    axes[1].set_ylabel('Third dim voxel coords', fontsize=12)
    axes[1].set_title('Second dimension (j), slice {}'.format(j), fontsize=15)
  
    axes[2].set_xlabel('First dim voxel coords.', fontsize=12)
    axes[2].set_ylabel('Second dim voxel coords', fontsize=12)
    axes[2].set_title('Third dimension (k), slice {}'.format(k), fontsize=15)

# modify above show_slices() fnc to include visualization of coordinate location
def add_coords(slices):
  fig, axes = plt.subplots(1, len(slices), figsize=(15,15))
  for idx, slice in enumerate(slices):
    axes[idx].imshow(slice.T, cmap="gray", origin="lower")

    axes[0].set_xlabel('Second dim voxel coords.', fontsize=12)
    axes[0].set_ylabel('Third dim voxel coords', fontsize=12)
    axes[0].set_title('First dimension (i), slice {}'.format(i), fontsize=15)

    axes[1].set_xlabel('First dim voxel coords.', fontsize=12)
    axes[1].set_ylabel('Third dim voxel coords', fontsize=12)
    axes[1].set_title('Second dimension (j), slice {}'.format(j), fontsize=15)
  
    axes[2].set_xlabel('First dim voxel coords.', fontsize=12)
    axes[2].set_ylabel('Second dim voxel coords', fontsize=12)
    axes[2].set_title('Third dimension (k), slice {}'.format(k), fontsize=15)  
    
    # plot the 3D coordinate in red
    axes[0].add_patch((patches.Rectangle((j, k), 3, 3, linewidth=2, edgecolor='r', facecolor='none')))
    axes[1].add_patch((patches.Rectangle((i, k), 3, 3, linewidth=2, edgecolor='r', facecolor='none')))
    axes[2].add_patch((patches.Rectangle((i, j), 3, 3, linewidth=2, edgecolor='r', facecolor='none')))

# Functions required for TorchIO 
def get_bounds(self):
    """Get image bounds in mm.

    Returns:
        np.ndarray: [description]
    """
    first_index = 3 * (-0.5,)
    last_index = np.array(self.spatial_shape) - 0.5
    first_point = nib.affines.apply_affine(self.affine, first_index)
    last_point = nib.affines.apply_affine(self.affine, last_index)
    array = np.array((first_point, last_point))
    bounds_x, bounds_y, bounds_z = array.T.tolist()
    return bounds_x, bounds_y, bounds_z

def to_pil(image):
    from PIL import Image
    from IPython.display import display
    data = image.numpy().squeeze().T
    data = data.astype(np.uint8)
    image = Image.fromarray(data)
    w, h = image.size
    display(image)
    print()  # in case multiple images are being displayed

def stretch(img):
    p1, p99 = np.percentile(img, (1, 99))
    from skimage import exposure
    img_rescale = exposure.rescale_intensity(img, in_range=(p1, p99))
    return img_rescale

def show_fpg(
        subject,
        to_ras=False,
        stretch_slices=True,
        indices=None,
        parcellation=False,
        ):
    subject = tio.ToCanonical()(subject) if to_ras else subject
    def flip(x):
        return np.rot90(x) # flip 90 degrees
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    if indices is None:
        t1_half_shape = torch.Tensor(subject.t1.spatial_shape) // 2
        t1_i, t1_j, t1_k = t1_half_shape.long()
        t1_i -= 5  # use a better slice

        t2_half_shape = torch.Tensor(subject.t2.spatial_shape) // 2
        t2_i, t2_j, t2_k = t2_half_shape.long()
        t2_i -= 5  # use a better slice
    else:
        t1_i, t1_j, t1_k = indices
    t1bounds_x, t1bounds_y, t1bounds_z = get_bounds(subject.t1)  ###
    t2bounds_x, t2bounds_y, t2bounds_z = get_bounds(subject.t2)

    orientation = ''.join(subject.t1.orientation)
    if orientation != 'RAS':
        import warnings
        warnings.warn(f'Image orientation should be RAS+, not {orientation}+')
    
    kwargs = dict(cmap='gray', interpolation='none')
    data = subject['t1'].data
    slices = data[0, t1_i], data[0, :, t1_j], data[0, ..., t1_k]
    if stretch_slices:
        slices = [stretch(s.numpy()) for s in slices]
    sag, cor, axi = slices
    
    axes[0, 0].imshow(flip(sag), extent=t1bounds_y + t1bounds_z, **kwargs)
    axes[0, 1].imshow(flip(cor), extent=t1bounds_x + t1bounds_z, **kwargs)
    axes[0, 2].imshow(flip(axi), extent=t1bounds_x + t1bounds_y, **kwargs)
    axes[0, 0].set_title('T1', fontsize=15)
    axes[0, 1].set_title('T1', fontsize=15)
    axes[0, 2].set_title('T1', fontsize=15)
    

    kwargs = dict(cmap='gray', interpolation='none')
    data2 = subject['t2'].data
    slicest2 = data2[0, t2_i], data2[0, :, t2_j], data2[0, ..., t2_k]
    if stretch_slices:
        slicest2 = [stretch(s.numpy()) for s in slicest2]
    sagt2, cort2, axit2 = slicest2
    
    axes[1, 0].imshow(flip(sagt2), extent=t2bounds_y + t2bounds_z, **kwargs)
    axes[1, 1].imshow(flip(cort2), extent=t2bounds_x + t2bounds_z, **kwargs)
    axes[1, 2].imshow(flip(axit2), extent=t2bounds_x + t2bounds_y, **kwargs)
    axes[1, 0].set_title('T2', fontsize=15)
    axes[1, 1].set_title('T2', fontsize=15)
    axes[1, 2].set_title('T2', fontsize=15)

def show_rbf(
        subject,
        to_ras=False,
        stretch_slices=True,
        indices=None,
        intensity_name='t1',
        parcellation=True,
        ):
    subject = tio.ToCanonical()(subject) if to_ras else subject
    def flip(x):
        return np.rot90(x)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    if indices is None:
        half_shape = torch.Tensor(subject.spatial_shape) // 2
        i, j, k = half_shape.long()
        i -= 5  # use a better slice
    else:
        i, j, k = indices
    bounds_x, bounds_y, bounds_z = get_bounds(subject.t1)  ###

    orientation = ''.join(subject.t1.orientation)
    if orientation != 'RAS':
        import warnings
        warnings.warn(f'Image orientation should be RAS+, not {orientation}+')
    
    kwargs = dict(cmap='gray', interpolation='none')
    data = subject[intensity_name].data
    slices = data[0, i], data[0, :, j], data[0, ..., k]
    if stretch_slices:
        slices = [stretch(s.numpy()) for s in slices]
    sag, cor, axi = slices
    
    axes[0, 0].imshow(flip(sag), extent=bounds_y + bounds_z, **kwargs)
    axes[0, 1].imshow(flip(cor), extent=bounds_x + bounds_z, **kwargs)
    axes[0, 2].imshow(flip(axi), extent=bounds_x + bounds_y, **kwargs)

    kwargs = dict(interpolation='none')
    data = subject.seg.data
    slices = data[0, i], data[0, :, j], data[0, ..., k]
    if parcellation:
        sag, cor, axi = [color_table.colorize(s.long()) if s.max() > 1 else s for s in slices]
    else:
        sag, cor, axi = slices
    axes[1, 0].imshow(flip(sag), extent=bounds_y + bounds_z, **kwargs)
    axes[1, 1].imshow(flip(cor), extent=bounds_x + bounds_z, **kwargs)
    axes[1, 2].imshow(flip(axi), extent=bounds_x + bounds_y, **kwargs)
    
    plt.tight_layout()


class ColorTable:
    def __init__(self, colors_path):
        self.df = self.read_color_table(colors_path)

    @staticmethod
    def read_color_table(colors_path):
        df = pd.read_csv(
            colors_path,
            sep=' ',
            header=None,
            names=['Label', 'Name', 'R', 'G', 'B', 'A'],
            index_col='Label',
        )
        return df

    def get_color(self, label: int):
        """
        There must be nicer ways of doing this
        """
        try:
            rgb = (
                self.df.loc[label].R,
                self.df.loc[label].G,
                self.df.loc[label].B,
            )
        except KeyError:
            rgb = 0, 0, 0
        return rgb

    def colorize(self, label_map: np.ndarray) -> np.ndarray:
        rgb = np.stack(3 * [label_map], axis=-1)
        for label in np.unique(label_map):
            mask = label_map == label
            color = self.get_color(label)
            rgb[mask] = color
        return rgb


#%% Load image
# load in our downloaded T2w MRI scan from the current directory
img = nib.load('C:/Users/alber/Bureau/Development/Data/Images_data/MRI/BraTS20_Training_001_t2.nii')

print(type(img))
print(img.shape)

#%%% the 3 part of a image
hdr = img.header
print(type(hdr))
hdr.get_zooms() # it's a 1 mm isotropic resolution MRI file!

img_data = img.get_fdata()
print(type(img_data)) # it's a numpy memory-map 
print(img_data.shape)

print(img_data)

mid_vox = img_data[118:121, 118:121, 108:111]
print(mid_vox)

mid_slice_x = img_data[80, : , :] # Change the index then run the next cell to see the changes in slices!
print(mid_slice_x.shape)

# Note that we transpose the slice (using the .T attribute).
# This is because imshow plots the first dimension on the y-axis and the
# second on the x-axis, but we'd like to plot the first on the x-axis and the
# second on the y-axis. Also, the origin to "lower", as the data was saved in
# "cartesian" coordinates.
plt.imshow(mid_slice_x.T, cmap='gray', origin='lower')
plt.xlabel('First axis')
plt.ylabel('Second axis')
plt.colorbar(label='Signal intensity')
plt.show()

n_i, n_j, n_k = img_data.shape
i = (n_i - 1) // 2  # // for integer division
j = (n_j - 1) // 2
k = (n_k - 1) // 2
print(i, j, k)

slice_0 = img_data[i, :, :]
slice_1 = img_data[:, j, :]
slice_2 = img_data[:, :, k]
# display slices calling our function
show_slices([slice_0, slice_1, slice_2])

np.set_printoptions(suppress=True, precision=3)  # Set numpy to print 3 decimal points and suppress small values
A = img.affine
print(A)

i = 80
j = 119
k = 77
# specify the indices for each slice
slice_0 = img_data[i, :, :]
slice_1 = img_data[:, j, :]
slice_2 = img_data[:, :, k]
# display slices calling our function
add_coords([slice_0, slice_1, slice_2])

xyz1 = A.dot([i, j, k, 1])
print(xyz1)

#%% Preprocess

sns.set_style("whitegrid", {'axes.grid' : False})
%config InlineBackend.figure_format = 'retina'
torch.manual_seed(14041931)

print('TorchIO version:', tio.__version__)
print('Last run on', time.ctime())

#%%% Spatial transformation

fpg = tio.datasets.FPG(load_all = True)
print('Sample subject:', fpg)
show_fpg(fpg)

print(fpg.t1)
print(fpg.t2)

to_ras = tio.ToCanonical()
fpg_ras = to_ras(fpg)
print('T1w old orientation:', fpg.t1.orientation)
print('T1w new orientation:', fpg_ras.t1.orientation)
print('T2w old orientation:', fpg.t2.orientation)
print('T2w new orientation:', fpg_ras.t2.orientation)
show_fpg(fpg_ras)

# print affine matrix
np.set_printoptions(precision=2, suppress=True)
print(fpg.t1['affine_matrix'].numpy())

# Resample to standard MNI space using the downloaded reference image at the beginning of the tutorial... this will take about a minute
to_mni = tio.Resample(mni.t1.path, pre_affine_name='affine_matrix')
fpg_mni = to_mni(fpg_ras) 
show_fpg(fpg_mni)

print(fpg_mni.t1)
print(fpg_mni.t2)

#@title No need to run. This is just another way to transform: using [torchio.Subject](https://torchio.readthedocs.io/data/subject.html) data struct

# Associate imgs to a single subject
subj = tio.Subject(T1w=fpg.t1, T2w=fpg.t2) 

preprocess_transforms = (
    tio.ToCanonical(), # reorient to RAS+
    tio.Resample(1, image_interpolation='bspline'), # resample to 1 mm isotropic spacing
    tio.Resample('T1w', image_interpolation='nearest'), # target output space (ie. match T2w to the T1w space) 
)
preprocess = tio.Compose(preprocess_transforms)
fpg_prep = preprocess(subj)

fpg_prep.plot()
print(fpg_prep.T1w)
print(fpg_prep.T2w)

target_shape = 150, 190, 170
crop_pad = tio.CropOrPad(target_shape)
fpg_crop = crop_pad(fpg_mni)
show_fpg(fpg_crop)

downsampling_factor = 3
original_spacing = 1
target_spacing = downsampling_factor / original_spacing  # in mm
downsample = tio.Resample(target_spacing)
downsampled = downsample(fpg_crop)
print('Original image:', fpg_ras.t1)
print('Downsampled image:', downsampled.t1)
print(f'The downsampled image takes {fpg_ras.t1.memory / downsampled.t1.memory:.1f} times less memory!')
show_fpg(downsampled)

#@title ### this takes about 3 minutes to run! (let's skip for now and feel free to come back to this later) 

original_spacing = 1
std = tio.Resample.get_sigma(downsampling_factor, original_spacing)
antialiasing = tio.Blur(std)  # we need don't need a random transform here
blurry = antialiasing(fpg_crop)
show_fpg(downsample(blurry)) 

#%%% Data augmentation


#%%% Intensity, normalization
#### Rescale intensity
#We can change the intensities range of our images so that it lies within e.g. 0 and 1, or -1 and 1, 
#using [`RescaleIntensity`](https://torchio.readthedocs.io/transforms/preprocessing.html#rescaleintensity).

rescale = tio.RescaleIntensity((-1, 1))
rescaled1 = rescale(fpg_ras)
fig, axes = plt.subplots(2, 1)
sns.distplot(fpg.t1.data, ax=axes[0], kde=False)
sns.distplot(rescaled1.t1.data, ax=axes[1], kde=False)
axes[0].set_title('Original histogram')
axes[1].set_title('Intensity rescaling')
axes[0].set_ylim((0, 1e6))
axes[1].set_ylim((0, 1e6))
plt.tight_layout()

#
rescale = tio.RescaleIntensity((-1, 1), percentiles=(1, 99))
rescaled2 = rescale(fpg_ras)
fig, axes = plt.subplots(2, 1)
sns.distplot(rescaled1.t1.data, ax=axes[0], kde=False)
sns.distplot(rescaled2.t1.data, ax=axes[1], kde=False)
axes[0].set_title('Original histogram')
axes[1].set_title('Intensity rescaling with percentiles 1 and 99')
axes[0].set_ylim((0, 1e6))
axes[1].set_ylim((0, 1e6))
plt.tight_layout()
show_fpg(rescaled2)

#Another common approach for normalization is forcing data points to have zero-mean and unit variance. 
#We can use [`ZNormalization`](https://torchio.readthedocs.io/transforms/preprocessing.html#znormalization) for this.

standardize = tio.ZNormalization()
standardized = standardize(fpg_ras)
fig, axes = plt.subplots(2, 1)
sns.distplot(fpg.t1.data, ax=axes[0], kde=False)
sns.distplot(standardized.t1.data, ax=axes[1], kde=False)
axes[0].set_title('Original histogram')
axes[1].set_title('Z-normalization')
axes[0].set_ylim((0, 1e6))
axes[1].set_ylim((0, 1e6))
plt.tight_layout()

fpg_thresholded = copy.deepcopy(standardized)
data_t1, data_t2 = fpg_thresholded.t1.data, fpg_thresholded.t2.data
data_t1[data_t1 > data_t1.float().mean()] = data_t1.max()
data_t2[data_t2 > data_t2.float().mean()] = data_t2.max()
show_fpg(fpg_thresholded)


standardize_foreground = tio.ZNormalization(masking_method=lambda x: x > x.float().mean()) # threshold values above mean only, for binary mask
standardized_foreground = standardize_foreground(fpg_ras)
fig, axes = plt.subplots(2, 1)
sns.distplot(fpg.t1.data, ax=axes[0], kde=False)
sns.distplot(standardized_foreground.t1.data, ax=axes[1], kde=False)
axes[0].set_title('Original histogram')
axes[1].set_title('Z-normalization using foreground stats')
axes[0].set_ylim((0, 1e6))
axes[1].set_ylim((0, 1e6))
plt.tight_layout()
show_fpg(standardized_foreground)

add_spike = tio.RandomSpike()
with_spike = add_spike(fpg_ras)
show_fpg(with_spike)

add_ghosts = tio.RandomGhosting(intensity=1.5)
show_fpg(add_ghosts(fpg_ras))

add_motion = tio.RandomMotion(num_transforms=6, image_interpolation='nearest') # this takes about a minute...
show_fpg(add_motion(fpg_ras))

#%% Transform and compose

get_foreground = tio.ZNormalization.mean

training_transform = tio.Compose([
     tio.Resample(
         mni.t1.path,
         pre_affine_name='affine_matrix'),      # to MNI space (which is RAS+)
     tio.RandomAnisotropy(p=0.25),              # make images look anisotropic 25% of times
     tio.CropOrPad((75, 100, 80)),              # tight crop to save RAM
     tio.ZNormalization(
         masking_method=get_foreground),        # zero mean, unit variance of foreground
     tio.RandomBlur(p=0.25),                    # blur 25% of times
     tio.RandomNoise(p=0.25),                   # Gaussian noise 25% of times
     tio.OneOf({                                # either
         tio.RandomAffine(): 0.8,               # random affine
         tio.RandomElasticDeformation(): 0.2,   # or random elastic deformation
     }, p=0.8),                                 # applied to 80% of images
     tio.RandomBiasField(p=0.3),                # magnetic field inhomogeneity 30% of times
     tio.OneOf({                                # either
         tio.RandomMotion(): 1,                 # random motion artifact
         tio.RandomSpike(): 2,                  # or spikes
         tio.RandomGhosting(): 2,               # or ghosts
     }, p=0.5),                                 # applied to 50% of images
])

fpg_training = copy.deepcopy(fpg_ras)
fpg_augmented = training_transform(fpg_training) # apply the transform
show_fpg(fpg_augmented)   

#
pprint.pprint(downsampled.history)

testing_transform = tio.Compose([
     tio.Resample(mni.t1.path, pre_affine_name='affine_matrix'),                   # to MNI space (which is RAS+)
     tio.ZNormalization(masking_method=get_foreground),                            # zero mean and unit std
])

fpg_testing = testing_transform(fpg)
show_fpg(fpg_testing)