##########################################################
# %%
# import libraries
##########################################################

import numpy                as  np
import tensorflow           as  tf

# keras
from tensorflow.keras import backend as K
from tensorflow.keras import Input
from tensorflow.keras.layers import Layer, Concatenate
from tensorflow.keras.models import Model

from library_net_function  import *
from library_net_recon     import create_recon

##########################################################
# %%
# forward model
##########################################################

class pForward(Layer):
    def __init__(self, **kwargs):
        super(pForward, self).__init__(**kwargs)

    def build(self, input_shape):
        super(pForward, self).build(input_shape)

    def call(self, x):
        o1,c1,m1    =  x
        
        def forward1(tmp):      
            o2,c2,m2    =   tmp       
            nx, ny      =   c2.shape[1:3]
            CI          =   c2 * tf.expand_dims(o2,axis=0)     
            kspace      =   tf.signal.fftshift(tf.signal.fft2d(tf.signal.fftshift(CI,axes=(1,2))),axes=(1,2)) # / tf.math.sqrt(tf.cast(nx*ny,tf.complex64))
            masked      =   kspace*tf.expand_dims(m2,axis=0)    
            return masked   

        inp1 = (o1[...,0],c1,m1)
        rec = tf.map_fn(forward1, inp1, dtype=tf.complex64)

        return rec

##########################################################
# %%
# network
##########################################################
    
def create_zero_MIRID_model(nx, ny, nc, nLayers, num_block, num_filters = 64):

    # define the inputs    
    input_c         = Input(shape=(nc,nx,ny),   dtype = tf.complex64,       name = 'input_c')
    input_k_trn1    = Input(shape=(nc,nx,ny),   dtype = tf.complex64,       name = 'input_k_trn1') 
    input_k_trn2    = Input(shape=(nc,nx,ny),   dtype = tf.complex64,       name = 'input_k_trn2') 
    input_k_lss1    = Input(shape=(nc,nx,ny),   dtype = tf.complex64,       name = 'input_k_lss1') 
    input_k_lss2    = Input(shape=(nc,nx,ny),   dtype = tf.complex64,       name = 'input_k_lss2') 
    input_m_trn1    = Input(shape=(nx,ny),      dtype = tf.complex64,       name = 'input_m_trn1') 
    input_m_trn2    = Input(shape=(nx,ny),      dtype = tf.complex64,       name = 'input_m_trn2') 
    input_m_lss1    = Input(shape=(nx,ny),      dtype = tf.complex64,       name = 'input_m_lss1') 
    input_m_lss2    = Input(shape=(nx,ny),      dtype = tf.complex64,       name = 'input_m_lss2') 

    recon_joint     =   create_recon(nx = nx, ny = ny, nc = nc, num_block = num_block,  nLayers = nLayers, num_filters=   num_filters)
                                           
    # functions and variables
    oForward    =   pForward()

    # Joint Recon
    [out1,out2] =   recon_joint([input_c, input_m_trn1, input_m_trn2, input_k_trn1, input_k_trn2])    
    out3        =   K.expand_dims(r2c(out1),axis=-1)
    out4        =   K.expand_dims(r2c(out2),axis=-1)

    # loss points in k-space
    lss_1st     =   oForward([out3,   input_c,    input_m_lss1+input_m_trn1]) # we might want input_m_lss1+input_m_trn1
    lss_2nd     =   oForward([out4,   input_c,    input_m_lss2+input_m_trn2])

    lss_l1      =   K.sum(K.abs(lss_1st - input_k_lss1 - input_k_trn1),axis=1) / (K.sum(K.abs(input_k_lss1+input_k_trn1))+K.sum(K.abs(input_k_lss2+input_k_trn2)))   \
                  + K.sum(K.abs(lss_2nd - input_k_lss2 - input_k_trn2),axis=1) / (K.sum(K.abs(input_k_lss1+input_k_trn1))+K.sum(K.abs(input_k_lss2+input_k_trn2)))  
    lss_l2      =   K.sum(K.square(K.abs(lss_1st - input_k_lss1 - input_k_trn1)),axis=1) / K.sqrt(K.sum(K.square(K.abs(input_k_lss1+input_k_trn1))+K.square(K.abs(input_k_lss2+input_k_trn2))))   \
                  + K.sum(K.square(K.abs(lss_2nd - input_k_lss2 - input_k_trn2)),axis=1) / K.sqrt(K.sum(K.square(K.abs(input_k_lss1+input_k_trn1))+K.square(K.abs(input_k_lss2+input_k_trn2))))   
        
    # outputs
    out_final   =   Concatenate(axis=-1)([  out3,   out4,   \
                                            K.cast(K.expand_dims(lss_l1,axis=-1),tf.complex64),          \
                                            K.cast(K.expand_dims(lss_l2,axis=-1),tf.complex64),          \
                                          ])   
    
    return Model(inputs     =   [ input_c,  input_k_trn1,   input_k_trn2,   input_k_lss1,   input_k_lss2,    
                                            input_m_trn1,   input_m_trn2,   input_m_lss1,   input_m_lss2    ],
                 outputs    =   [ out_final  ],
                 name       =   'zero-MIRID' )


##########################################################
# %%
# custom loss
##########################################################

def loss_custom_v1(y_true, y_pred):
    
    # l1 norm
    l1      =   K.sum(K.abs(y_pred[...,-2]))  
    # l2 norm
    l2      =   K.sqrt(K.sum(K.abs(y_pred[...,-1]))) 

    return ( l1 + l2 )  

