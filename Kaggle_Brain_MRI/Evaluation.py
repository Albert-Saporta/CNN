# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 11:43:05 2022

@author: alber
"""

from modules import *
from skimage.io import imsave
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
device = torch.device("cuda")

phase = "valid"
with torch.set_grad_enabled(False):
    unet = UNet(in_channels=BrainSegmentationDataset.in_channels, out_channels=BrainSegmentationDataset.out_channels)
    state_dict = torch.load("unet.pt", map_location=device)
    unet.load_state_dict(state_dict)
    unet.eval()
    unet.to(device)

    input_list = []
    pred_list = []
    true_list = []

    for i, data in tqdm(enumerate(loaders[phase])):
        x, y_true = data
        x, y_true = x.to(device), y_true.to(device)

        y_pred = unet(x)
        y_pred_np = y_pred.detach().cpu().numpy()
        pred_list.extend([y_pred_np[s] for s in range(y_pred_np.shape[0])])

        y_true_np = y_true.detach().cpu().numpy()
        true_list.extend([y_true_np[s] for s in range(y_true_np.shape[0])])

        x_np = x.detach().cpu().numpy()
        input_list.extend([x_np[s] for s in range(x_np.shape[0])])
        
#%%
volumes = postprocess_per_volume(
    input_list,
    pred_list,
    true_list,
    loaders[phase].dataset.patient_slice_index,
    loaders[phase].dataset.patients,
)

dsc_dist = dsc_distribution(volumes)

dsc_dist_plot = plot_dsc(dsc_dist)
imsave("./dsc.png", dsc_dist_plot)

for p in volumes:
    x = volumes[p][0]
    y_pred = volumes[p][1]
    y_true = volumes[p][2]
    for s in range(x.shape[0]):
        image = gray2rgb(x[s, 1])  # channel 1 is for FLAIR
        image = outline(image, y_pred[s, 0], color=[255, 0, 0])
        image = outline(image, y_true[s, 0], color=[0, 255, 0])
        filename = "{}-{}.png".format(p, str(s).zfill(2))
        filepath = os.path.join("./predictions", filename)
        imsave(filepath, image)