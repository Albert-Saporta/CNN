# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 11:18:44 2022

@author: alber
"""

from modules import *
#%% Initialization

#%%% Hyperparameters

n_epochs= 2#100
vis_freq = 10
vis_images = 2
device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")
dsc_loss = DiceLoss()

#%%% Data
loader_train, loader_valid = data_loaders()
loaders = {"train": loader_train, "valid": loader_valid}

#%%% models
unet = UNet2D(in_channels=BrainSegmentationDataset.in_channels, out_channels=BrainSegmentationDataset.out_channels)
unet.to(device)
optimizer = optim.Adam(unet.parameters(), lr=0.001)

#%% Training



best_validation_dsc = 0.0


loss_train = []
loss_valid = []

step = 0

for epoch in range(n_epochs):
    for phase in ["train", "valid"]:
        if phase == "train":
            unet.train()
        else:
            unet.eval()

        validation_pred = []
        validation_true = []
        loop = tqdm(loaders[phase], leave=True)
        for i, data in enumerate(loop):
            loop.set_description(f'Epoch {epoch+1}/{n_epochs}')
            if phase == "train":
                step += 1

            x, y_true = data
            x, y_true = x.to(device), y_true.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == "train"):
                y_pred = unet(x)

                loss = dsc_loss(y_pred, y_true)

                if phase == "valid":
                    loss_valid.append(loss.item())
                    y_pred_np = y_pred.detach().cpu().numpy()
                    validation_pred.extend(
                        [y_pred_np[s] for s in range(y_pred_np.shape[0])]
                    )
                    y_true_np = y_true.detach().cpu().numpy()
                    validation_true.extend(
                        [y_true_np[s] for s in range(y_true_np.shape[0])]
                    )
                    if (epoch % vis_freq == 0) or (epoch == epochs - 1):
                        if i * batch_size < vis_images:
                            tag = "image/{}".format(i)
                            num_images = vis_images - i * batch_size

                if phase == "train":
                    loss_train.append(loss.item())
                    loss.backward()
                    optimizer.step()

            if phase == "train" and (step + 1) % 10 == 0:
                print(f"step : {step}  ---  loss train : {loss_train}")
                loss_train = []

        if phase == "valid":
            print(f"step : {step}  ---  loss train : {loss_valid}")
            mean_dsc = np.mean(
                dsc_per_volume(
                    validation_pred,
                    validation_true,
                    loader_valid.dataset.patient_slice_index,
                )
            )
            print(f"step : {step}  ---  mean_dsc : {mean_dsc}")
            if mean_dsc > best_validation_dsc:
                best_validation_dsc = mean_dsc
                torch.save(unet.state_dict(), "unet.pt")
            loss_valid = []

print("Best validation mean DSC: {:4f}".format(best_validation_dsc))