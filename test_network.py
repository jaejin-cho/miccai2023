##########################################################
# %%
# 
##########################################################
import os, sys

os.environ["CUDA_DEVICE_ORDER"]     = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]  = "0"

sys.path.append('utils')

##########################################################
# %%
# import libraries
##########################################################
import  numpy               as      np
import  library_utils       as      mf
import  library_net         as      mn
from    library_dat         import  data_generator
from    scipy.io            import  savemat

# keras
from tensorflow.keras.optimizers import  Adam

##########################################################
# %%
# imaging parameters
##########################################################
model_name              =   'net/zero-MIRID-v01.h5'
data_name               =   'dat/example.mat'
num_block               =   10
num_layer               =   16
num_filter              =   46
    
##########################################################
# %%
# load data
# axis : diffusion, channel, x, y, shot
##########################################################

print('loading data')

if os.path.exists(data_name)==False:
    os.system("wget -O " + data_name + " https://www.dropbox.com/s/2rteu3vmtbj15kx/example.mat?dl=1")
    
dt      =   mf.load_h5py(data_name)
csm     =   np.expand_dims(np.transpose(dt['csm']['real'] + 1j*dt['csm']['imag'],(2,0,1)),axis=0)
kspace  =   np.transpose(dt['kspace']['real'] + 1j*dt['kspace']['imag'],(4,2,0,1,3))
del dt

[nd,nc,nx,ny,ns] = kspace.shape

##########################################################
# %%
# find EPI mask 
##########################################################
print('finding k-space mask')
mask_all            =   np.zeros([1,nx,ny,2],   dtype=np.complex64)
ind_non_z           =   np.nonzero(np.reshape(np.sum(np.sum(np.abs(kspace),1),0),mask_all.shape)) 
mask_all[ind_non_z] =   1
del ind_non_z

##########################################################
# %%
# load the network
##########################################################

model = mn.create_zero_MIRID_model(     nx =   nx,
                                        ny =   ny,
                                        nc =   nc,
                                        num_block   =   num_block,
                                        nLayers     =   num_layer,
                                        num_filters =   num_filter, )
# Define an optimizer
adam_opt = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-09, decay=0.001)

# Compile the model
model.compile(optimizer=adam_opt, loss= mn.loss_custom_v1)
# model.summary()

try:
    model.load_weights(model_name)
    print('succeeded to load the model')
except:
    print('failed to load the model')

    
##########################################################
# %%
# interfere
##########################################################

tst_par = { 'kdata_all'     : kspace,
            'csm'           : csm,
            'mask_trn1'     : np.tile(mask_all[...,0], (nd,1,1)),
            'mask_trn2'     : np.tile(mask_all[...,1], (nd,1,1)),
            'mask_lss1'     : np.tile(mask_all[...,0], (nd,1,1)),
            'mask_lss2'     : np.tile(mask_all[...,1], (nd,1,1)), }

tst_dat =   data_generator(**tst_par)
pred    =   model.predict(tst_dat)

recon_all = pred[...,0:2] 
recon_dif = mf.msos(np.transpose(recon_all,(1,2,0,3)),axis=-1)
recon_dwi = mf.msos(recon_dif,axis=-1)

mf.mosaic(np.rot90(np.abs(np.squeeze(recon_dif))),4,8,101,[0,0.6],'Zero-MIRID')
mf.mosaic(np.rot90(np.abs(np.squeeze(recon_dwi))),1,1,102,[0,4],'DWI')


savemat('res/example_results.mat', {"msEPI": recon_all, 'dwi': recon_dwi, 'dif': recon_dif  })

