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
from    matplotlib          import  pyplot   as plt
from    scipy.io            import  savemat
import  time

# keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks  import  ModelCheckpoint, EarlyStopping


##########################################################
# %%
# imaging parameters
##########################################################
model_name  =   'net/zero-MIRID-v02.h5'
hist_name   =   'net/zero-MIRID-v02.npy'
data_name   =   'dat/example.mat'
num_block   =   10
num_layer   =   16
num_filter  =   46

num_epoch   =   50
num_split   =   50
rho_val     =   0.2
rho_trn     =   0.4
    

##########################################################
# %%
# load data
# axis : diffusion, channel, x, y, shot
##########################################################

print('loading data')
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
# gen validation mask
##########################################################
print('generating validating mask')

kdata1 = np.transpose(kspace[0,:,:,:,0],axes=(1,2,0))
kdata2 = np.transpose(kspace[0,:,:,:,1],axes=(1,2,0))

mask_trn1 = np.empty((nd, nx, ny), dtype=np.float32)
mask_trn2 = np.empty((nd, nx, ny), dtype=np.float32)
mask_val1 = np.empty((nd, nx, ny), dtype=np.float32)
mask_val2 = np.empty((nd, nx, ny), dtype=np.float32)

for ii in range(nd):
    mask_trn1[ii,...], mask_val1[ii,...] = mf.uniform_selection(kdata1, mask_all[0,:,:,0].real, rho_val)
    mask_trn2[ii,...], mask_val2[ii,...] = mf.uniform_selection(kdata2, mask_all[0,:,:,1].real, rho_val)


##########################################################
# %%
# gen training mask
##########################################################
print('generating training mask')
mask_trn_split1 = np.empty((num_split*nd, nx, ny), dtype=np.complex64)
mask_trn_split2 = np.empty((num_split*nd, nx, ny), dtype=np.complex64)
mask_lss_split1 = np.empty((num_split*nd, nx, ny), dtype=np.complex64)
mask_lss_split2 = np.empty((num_split*nd, nx, ny), dtype=np.complex64)

for jj in range(num_split*nd):
    mask_trn_split1[jj, ...], mask_lss_split1[jj, ...] = mf.uniform_selection(kdata1,np.copy(mask_trn1[jj%nd, ...]),rho=rho_trn)
    mask_trn_split2[jj, ...], mask_lss_split2[jj, ...] = mf.uniform_selection(kdata2,np.copy(mask_trn2[jj%nd, ...]),rho=rho_trn)

del ii, jj, kdata1, kdata2


##########################################################
# %%
# define generator 
##########################################################
trn_par = { 'kdata_all'     : kspace,
            'csm'           : csm,
            'mask_trn1'     : mask_trn_split1,
            'mask_trn2'     : mask_trn_split2,
            'mask_lss1'     : mask_lss_split1,
            'mask_lss2'     : mask_lss_split2,     }
trn_dat =   data_generator(**trn_par)

val_par = { 'kdata_all'     : kspace,
            'csm'           : csm,
            'mask_trn1'     : mask_trn1,
            'mask_trn2'     : mask_trn2,
            'mask_lss1'     : mask_val1,
            'mask_lss2'     : mask_val2,     }
val_dat =   data_generator(**val_par)
    
##########################################################
# %%
# create modl network
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


##########################################################
# %%
# callback
##########################################################

model_callback = ModelCheckpoint(   model_name,
                                    monitor           =   'val_loss',
                                    verbose           =   1,
                                    save_best_only    =   True,
                                    save_weights_only =   True,
                                    mode              =   'auto'    )

model_EarlyStop = EarlyStopping(    monitor                 =   "val_loss",
                                    min_delta               =   0,
                                    patience                =   10,
                                    verbose                 =   1,
                                    mode                    =   "auto",
                                    baseline                =   None,
                                    restore_best_weights    =   True,   )

try:
    model.load_weights(model_name)
    print('loading the pretrained model...')
except:
    print('without loading a pretrained model...')

##########################################################
# %%
# train the network
##########################################################

t_start         =   time.time()
hist            =   model.fit(  trn_dat,
                                validation_data =   val_dat,
                                epochs          =   num_epoch,
                                batch_size      =   1,
                                verbose         =   1,
                                steps_per_epoch =   None,
                                callbacks       =   [model_EarlyStop, model_callback]    )

t_end           =   time.time()

# model.save_weights(model_name)
np.save(hist_name,{'hist':hist.history,'training_time':t_end-t_start})

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])

