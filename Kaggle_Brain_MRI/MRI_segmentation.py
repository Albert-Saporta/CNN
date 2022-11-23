# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 11:18:44 2022

@author: alber
"""

from modules import *
#%% Initialization
pth_file_name="unet_kaggle_MRI"
#%%% Hyperparameters

n_epochs= 100
vis_freq = 10
vis_images = 2
device = torch.device("cuda")
dsc_loss = DiceLoss()
batch_size=2

#%%% Data
images_path="C:/Users/alber/Bureau/Development/Data/Images_data/MRI/subset"#"kaggle_3m"
loader_train, loader_valid = data_loaders(images_path,batch_size)
loaders = {"train": loader_train, "valid": loader_valid}
print('data loaded')

#%%% models
#if __name__ == '__main__':
unet = UNet2D(in_channels=BrainSegmentationDataset.in_channels, out_channels=BrainSegmentationDataset.out_channels)
unet.to(device)
optimizer = optim.Adam(unet.parameters(), lr=0.001)

#%% Training




print(' ')
print('Training')

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
            if phase == "train":
                loop.set_description(f'Epoch {epoch+1}/{n_epochs}')
                #loop.set_postfix(loss=loss.item())

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
                    if (epoch % vis_freq == 0) or (epoch == n_epochs - 1):
                        if i * batch_size < vis_images:
                            tag = "image/{}".format(i)
                            num_images = vis_images - i * batch_size

                if phase == "train":
                    loss_train.append(loss.item())
                    loss.backward()
                    optimizer.step()

            if phase == "train" and (step + 1) % 10 == 0:
                #print(f"step : {step}  ---  loss train : {loss_train}")
                loss_train = []

        if phase == "valid":
            #print(f"step : {step}  ---  loss train : {loss_valid}")
            mean_dsc = np.mean(
                dsc_per_volume(
                    validation_pred,
                    validation_true,
                    loader_valid.dataset.patient_slice_index,
                )
            )
            #print(f"step : {step}  ---  mean_dsc : {mean_dsc}")
            #torch.save(unet.state_dict(), f'C:/Users/alber/Bureau/Development/DeepLearning/training_results/{pth_file_name}.pth')

            if mean_dsc > best_validation_dsc:
                best_validation_dsc = mean_dsc
                torch.save(unet.state_dict(), f'C:/Users/alber/Bureau/Development/DeepLearning/training_results/{pth_file_name}.pth')
            loss_valid = []
            


print("Best validation mean DSC: {:4f}".format(best_validation_dsc))