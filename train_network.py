##########################################################
# %%
# Import Libraries
##########################################################
import os
import sys

sys.path.append('utils')

import numpy as np
import time
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from scipy.io import savemat

# Custom imports for utility functions
import library_utils as mf  
import library_net as mn
from library_dat import DataGenerator  

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


##########################################################
# %%
# Imaging Parameters
##########################################################
model_name  = 'net/zero-MIRID-v02.pth'
hist_name   = 'net/zero-MIRID-v02.npy'
data_name   = 'dat/example.mat'
num_block   = 10
num_layer   = 16
num_filter  = 46

num_epoch   = 50
num_split   = 50
rho_val     = 0.2
rho_trn     = 0.4

##########################################################
# %%
# load data
# axis : diffusion, channel, x, y, shot
##########################################################

print('loading data')

if not os.path.exists(data_name):
    os.system(f"wget -O {data_name} https://www.dropbox.com/s/2rteu3vmtbj15kx/example.mat?dl=1")

dt  = mf.load_h5py(data_name)
csm = np.expand_dims(np.transpose(dt['csm']['real'] + 1j * dt['csm']['imag'], (2, 0, 1)), axis=0)
kspace = np.transpose(dt['kspace']['real'] + 1j * dt['kspace']['imag'], (4, 2, 0, 1, 3))
del dt

nd, nc, nx, ny, ns = kspace.shape

##########################################################
# %%
# find EPI mask 
##########################################################
print('finding k-space mask')
mask_all            =   np.zeros([1,nx,ny,2],   dtype=np.complex64)
ind_non_z           =   np.nonzero(np.reshape(np.sum(np.sum(np.abs(kspace),1),0),mask_all.shape)) 
mask_all[ind_non_z] =   1

del ind_non_z

print('generating validating mask')
kdata1 = np.transpose(kspace[0, :, :, :, 0], axes=(1, 2, 0))
kdata2 = np.transpose(kspace[0, :, :, :, 1], axes=(1, 2, 0))

mask_trn1 = np.empty((nd, nx, ny), dtype=np.float32)
mask_trn2 = np.empty((nd, nx, ny), dtype=np.float32)
mask_val1 = np.empty((nd, nx, ny), dtype=np.float32)
mask_val2 = np.empty((nd, nx, ny), dtype=np.float32)

for ii in range(nd):
    mask_trn1[ii, ...], mask_val1[ii, ...] = mf.uniform_selection(kdata1, mask_all[0, :, :, 0].real, rho_val)
    mask_trn2[ii, ...], mask_val2[ii, ...] = mf.uniform_selection(kdata2, mask_all[0, :, :, 1].real, rho_val)

print('generating training mask')
mask_trn_split1 = np.empty((num_split * nd, nx, ny), dtype=np.complex64)
mask_trn_split2 = np.empty((num_split * nd, nx, ny), dtype=np.complex64)
mask_lss_split1 = np.empty((num_split * nd, nx, ny), dtype=np.complex64)
mask_lss_split2 = np.empty((num_split * nd, nx, ny), dtype=np.complex64)

for jj in range(num_split * nd):
    mask_trn_split1[jj, ...], mask_lss_split1[jj, ...] = mf.uniform_selection(kdata1, np.copy(mask_trn1[jj % nd, ...]), rho=rho_trn)
    mask_trn_split2[jj, ...], mask_lss_split2[jj, ...] = mf.uniform_selection(kdata2, np.copy(mask_trn2[jj % nd, ...]), rho=rho_trn)

del ii, jj, kdata1, kdata2


##########################################################
# %%
# define generator 
##########################################################
trn_par = {
    'kdata_all': kspace,
    'csm': csm,
    'mask_trn1': mask_trn_split1,
    'mask_trn2': mask_trn_split2,
    'mask_lss1': mask_lss_split1,
    'mask_lss2': mask_lss_split2,
}

val_par = {
    'kdata_all': kspace,
    'csm': csm,
    'mask_trn1': mask_trn1,
    'mask_trn2': mask_trn2,
    'mask_lss1': mask_val1,
    'mask_lss2': mask_val2,
}

trn_dat = DataGenerator(**trn_par)
val_dat = DataGenerator(**val_par)

train_loader = DataLoader(trn_dat, batch_size=1, shuffle=True)
val_loader   = DataLoader(val_dat, batch_size=1, shuffle=False)

    
##########################################################
# %%
# create modl network
##########################################################

model = mn.ZeroMIRIDModel(
    nx=nx,
    ny=ny,
    num_block=num_block,
    nLayers=num_layer,
    num_filters=num_filter,
).to(device)

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-09, weight_decay=0.001)

# Loss function
criterion = mn.loss_custom_v1  

##########################################################
# %%
# Training Loop
##########################################################
print("Starting training...")

best_val_loss = float('inf')
early_stop_patience = 10
patience_counter = 0

for epoch in range(num_epoch):
    model.train()
    running_loss = 0.0
    for inputs in train_loader:
        optimizer.zero_grad()

        inputs = [x.to(device) if torch.is_tensor(x) else x for x in inputs]  # Move inputs to GPU
        outputs = model(inputs)
        loss = criterion(outputs)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for inputs in val_loader:
            inputs = [x.to(device) if torch.is_tensor(x) else x for x in inputs]
            outputs = model(inputs)
            loss = criterion(outputs)
            val_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epoch}, Loss: {running_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}")

    # Checkpointing
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), model_name)
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= early_stop_patience:
        print("Early stopping triggered.")
        break

# Save final model and training history
torch.save(model.state_dict(), model_name)