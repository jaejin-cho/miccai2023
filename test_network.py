##########################################################
# %%
# 
##########################################################
import os
import sys
sys.path.append('utils')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import savemat
from torch.utils.data import DataLoader

import library_utils as mf
import library_net as mn
from library_dat import DataGenerator 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


##########################################################
# %%
# imaging parameters
##########################################################
model_name = 'net/zero-MIRID-v01.pth'  
data_name = 'dat/example.mat'
num_block = 10
num_layer = 16
num_filter = 46

    
##########################################################
# %%
# load data
# axis : diffusion, channel, x, y, shot
##########################################################

print('loading data')

if not os.path.exists(data_name):
    os.system(f"wget -O {data_name} https://www.dropbox.com/s/2rteu3vmtbj15kx/example.mat?dl=1")

dt = mf.load_h5py(data_name)
csm = np.expand_dims(np.transpose(dt['csm']['real'] + 1j * dt['csm']['imag'], (2, 0, 1)), axis=0)
kspace = np.transpose(dt['kspace']['real'] + 1j * dt['kspace']['imag'], (4, 2, 0, 1, 3))
del dt

nd, nc, nx, ny, ns = kspace.shape


##########################################################
# %%
# find EPI mask 
##########################################################
print('finding k-space mask')
mask_all = np.zeros([1, nx, ny, 2], dtype=np.complex64)
ind_non_z = np.nonzero(np.reshape(np.sum(np.sum(np.abs(kspace), 1), 0), mask_all.shape)) 
mask_all[ind_non_z] = 1
del ind_non_z

##########################################################
# %%
# load the network
##########################################################

# Load and initialize the model

model = mn.ZeroMIRIDModel(
    nx=nx,
    ny=ny,
    num_block=num_block,
    nLayers=num_layer,
    num_filters=num_filter,
).to(device)

model = model.to(device)

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-09, weight_decay=0.001)

# Load model weights if available
if os.path.exists(model_name):
    model.load_state_dict(torch.load(model_name))
    print('Succeeded to load the model')
else:
    print('Failed to load the model')

    
##########################################################
# %%
# interfere
##########################################################
# Set model to evaluation mode
model.eval()

# Prepare data for inference
tst_par = {
    'kdata_all': kspace,
    'csm': csm,
    'mask_trn1': np.tile(mask_all[..., 0], (nd, 1, 1)),
    'mask_trn2': np.tile(mask_all[..., 1], (nd, 1, 1)),
    'mask_lss1': np.tile(mask_all[..., 0], (nd, 1, 1)),
    'mask_lss2': np.tile(mask_all[..., 1], (nd, 1, 1)),
}


tst_dat = DataGenerator(**tst_par)
tst_loader = DataLoader(tst_dat, batch_size=1, shuffle=False)

# Perform inference
predictions = []
with torch.no_grad():
    for inputs in tst_loader:
        inputs = [x.to(device) if torch.is_tensor(x) else x for x in inputs]
        output = model(inputs)
        predictions.append(output.cpu().numpy())  # Move to CPU for further processing

pred = np.concatenate(predictions, axis=0)

# Post-process predictions
recon_all = pred[..., 0:2]
recon_dif = mf.msos(np.transpose(recon_all, (1, 2, 0, 3)), axis=-1)
recon_dwi = mf.msos(recon_dif, axis=-1)

# Visualization
mf.mosaic(np.rot90(np.abs(np.squeeze(recon_dif))), 4, 8, 101, [0, 0.6], 'Zero-MIRID')
mf.mosaic(np.rot90(np.abs(np.squeeze(recon_dwi))), 1, 1, 102, [0, 4], 'DWI')

# Save the results
savemat('res/example_results.mat', {"msEPI": recon_all, 'dwi': recon_dwi, 'dif': recon_dif})
