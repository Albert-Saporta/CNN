# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 11:16:33 2022

@author: alber
"""

from collections import OrderedDict
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F

#%% 2D Unet (kaglle brain MRI)

class UNet2D(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet2D, self).__init__()

        features = init_features
        self.encoder1 = UNet2D._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet2D._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet2D._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet2D._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet2D._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet2D._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet2D._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet2D._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet2D._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

#%% CNN for radiomics (McMedHacks 7.2)


#%%% CT+dose plan

class RadiomicsCNN(nn.Module):
    
    # RadiomicsCNN Architecture
    # 
    # dosemap and CT scans --> conv --> normalization --> relu --> max pooling --> 
    #       flattening --> 
    # Adding clinical path
    #       FC reduce X to the size of X_clinical --> normalize --> rrelu -->
    #       X + X_clinic --> 
    #       FC --> normalize --> rrelu --> 
    #       FC --> normalize --> sigmoid
    



    def __init__(self,dim1,dim2,dim3,n_cln):

        super(RadiomicsCNN, self).__init__()
        
        ks    = 3
        pool  = 2
        
        # Convolution layers
        n_in  = 2
        n_out = 1
        self.conv1 = RadiomicsCNN.block(n_in,32,ks,pool)
        self.conv2 = RadiomicsCNN.block(32,64,ks,pool)
        self.conv3 = RadiomicsCNN.block(64,128,ks,pool)
        self.conv4 = RadiomicsCNN.block(128,256,ks,pool)
        self.conv5 = RadiomicsCNN.block(256,512,ks,pool)
        # Flattening
        self.flat = nn.Flatten()
        
        # Fully-conneceted layers 
        # FC 1: make the size of x equal to the size of the clinical path
        
        self.fc1 = nn.Linear(30720, n_cln)#506880
        self.fc1_bn = nn.BatchNorm1d(n_cln)
        
        # FC 2: expand the features after concatination
        L_in = int(2*n_cln) 
        L_out = int(L_in)
        self.fc2 = nn.Linear(L_in, L_out)
        self.fc2_bn = nn.BatchNorm1d(L_out)
        
        # FC 3: make prediction
        self.fc3 = nn.Linear(L_out, 1)
        self.fc3_bn = nn.BatchNorm1d(1)
        


    def forward(self, x_dos, x_cts, x_cln):
        
        
        # Concatenate dosimetric and clinical features
        x = torch.cat((x_dos, x_cts), dim=1)
        print("memory cat",torch.cuda.memory_allocated()/torch.cuda.max_memory_allocated())

        x = self.conv1(x)
        print("memory conv 1",torch.cuda.memory_allocated()/torch.cuda.max_memory_allocated())

        x = self.conv2(x)
        print("memory 2",torch.cuda.memory_allocated()/torch.cuda.max_memory_allocated())

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.flat(x)
        print("memory flat",torch.cuda.memory_allocated()/torch.cuda.max_memory_allocated())

        #print("flat shape",x.shape)
        # FC layer to reduce x to the size of the clinical path
        x = F.rrelu(self.fc1_bn( self.fc1(x)))
        # Concatinate clinical variables
        x_cln = x_cln.squeeze(1)
        #print(x_cln)
        x_final = torch.cat((x, x_cln), dim=1)
        # Last FCs
        x_final = F.rrelu(self.fc2_bn(self.fc2(x_final)))
        x_final = torch.sigmoid(self.fc3_bn(self.fc3(x_final)))

        return x_final.squeeze()    
    
    @staticmethod
    def block(n_in, n_out, ks,pool):
        return nn.Sequential(nn.Conv3d(in_channels = n_in, out_channels = n_out, kernel_size = ks,padding=1),
            nn.BatchNorm3d(n_out),
            nn.LeakyReLU(0.2),
            nn.MaxPool3d(kernel_size=pool))
            #,
            #nn.Dropout(p=0.2))
    #check the order: https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout
            
#%%% Ct +dose renet like
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv3d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm3d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv3d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm3d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        #print("res shape",residual.shape)
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        #print("res and out shape",residual.shape,out.shape)
        out += residual
        #print("final out shape",residual.shape,out.shape)
        out = self.relu(out)
        #print("memory res out",torch.cuda.memory_allocated()/torch.cuda.max_memory_allocated())
        return out

class ResNet(nn.Module):
    def __init__(self,dim1,dim2,dim3,n_cln,block, layers, num_classes = 2):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv3d(2, 64, kernel_size = 7, stride = 2, padding = 3),#•ks 3
                        nn.BatchNorm3d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool3d(kernel_size = 3, stride = 2, padding = 1)
        #self.layer0 = self._make_layer(block, 32, layers[0], stride = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool3d(7, stride=1)
        #self.flat = nn.Flatten()
        # FC 1: make the size of x equal to the size of the clinical path
        self.fc1 = nn.Linear(512, n_cln)#506880
        self.fc1_bn = nn.BatchNorm1d(n_cln)
        
        # FC 2: expand the features after concatination
        L_in = int(2*n_cln) 
        L_out = int(L_in)
        self.fc2 = nn.Linear(L_in, L_out)
        self.fc2_bn = nn.BatchNorm1d(L_out)
        
        # FC 3: make prediction
        self.fc3 = nn.Linear(L_out, 1)
        self.fc3_bn = nn.BatchNorm1d(1)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes, kernel_size=1, stride=stride),#kernel size 1?
                nn.BatchNorm3d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        
        return nn.Sequential(*layers)
    
    
    def forward(self, x_dos, x_cts, x_cln):
        x = torch.cat((x_dos, x_cts), dim=1)

        x = self.conv1(x)
        x = self.maxpool(x)
        #print("memory pool",torch.cuda.memory_allocated()/torch.cuda.max_memory_allocated())

        x = self.layer0(x)
        #print("memory layer 0",torch.cuda.memory_allocated()/torch.cuda.max_memory_allocated())
        print("")
        x = self.layer1(x)
        print("memory layer 1",torch.cuda.memory_allocated()/torch.cuda.max_memory_allocated())
        x = self.layer2(x)
        print("memory layer 2",torch.cuda.memory_allocated()/torch.cuda.max_memory_allocated())
        x = self.layer3(x)
        #x = self.layer4(x)

        #x = self.flat(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = F.rrelu(self.fc1_bn( self.fc1(x)))
        # Concatinate clinical variables
        x_cln = x_cln.squeeze(1)
        #print(x_cln)
        x_final = torch.cat((x, x_cln), dim=1)
        # Last FCs
        x_final = F.rrelu(self.fc2_bn(self.fc2(x_final)))
        x_final = torch.sigmoid(self.fc3_bn(self.fc3(x_final)))
        #x = self.avgpool(x)
  

        return x_final.squeeze()    
                

#%%% DVH

class RadiomicsDVH(nn.Module):
    # Hyperparameters:
    #   kernel size of the concolution layer     (1)
    #   size of the average pooling              (1)

    def __init__(self, ks, pool,X_cln,X_dvh):
        super(RadiomicsDVH, self).__init__()
        
        
        # Convolution layers
        n_in  = 1
        n_out = 1 
        self.conv1 = nn.Conv1d(in_channels = n_in, out_channels = n_out, kernel_size = ks)
        self.conv1_bn = nn.BatchNorm1d(n_out)
    
        
        # Pooling layers
        self.avgpool1 = nn.AvgPool1d(kernel_size = pool)
        
        # Flattening
        self.flat = nn.Flatten()
        
        
        # Fully-conneceted layers 
        
        # --- FC 1: make the size of x equal to the of the clinical path
        n_cln = X_cln.shape[-1]
    
        I, P, K, S = X_dvh.shape[1], 0, ks, 1
        O = (I - K + 2*P) / S + 1 
        O = (O - pool)/pool + 1
        
        O = int(O)

        self.fc1 = nn.Linear(O, n_cln)
        self.fc1_bn = nn.BatchNorm1d(n_cln)
        
        # --- FC 2: Allow features to interact
        L_in = int ( 2 * n_cln ) # since the dvh and clinical paths are now equal --> 2 X
        L_out = int ( L_in )
        self.fc2 = nn.Linear(L_in, L_out)
        self.fc2_bn = nn.BatchNorm1d(L_out)
        
        # --- FC 3: make prediction
        self.fc3 = nn.Linear(L_out, 1)
        self.fc3_bn = nn.BatchNorm1d(1)
        

    def forward(self, X_dvh, X_cln):
        
        # average pooling
        x = self.avgpool1(  F.rrelu(  self.conv1_bn(  self.conv1(X_dvh)  )  )  )  
        
        x = self.flat(x)

 
        # FC layer to reduce x to the size of the clinical path
        x = F.relu ( self.fc1_bn( self.fc1(x) ) )
       

        # Concatinate clinical variables
        X_cln = X_cln.squeeze(1)
        x_fin = torch.cat((x, X_cln), dim=1)

        # Last FCs
        x_fin = F.rrelu  ( self.fc2_bn( self.fc2(x_fin) ) )
        x_fin = torch.sigmoid( self.fc3_bn( self.fc3(x_fin) ) )

        return x_fin.squeeze()