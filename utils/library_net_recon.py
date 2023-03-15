##########################################################
# %%
# import libraries
##########################################################

import numpy                as  np
import tensorflow           as  tf

# keras
from tensorflow.keras import Input
from tensorflow.keras.layers import Layer, Add, Concatenate
from tensorflow.keras.models import Model
from scipy.linalg import dft

from library_net_function import *


##########################################################
# %%
# SENSE
##########################################################

class Aclass:
    def __init__(self, csm,mask,lam):
        with tf.name_scope('Ainit'):
            self.mask       =       mask
            self.csm        =       csm
            self.lam        =       lam
            self.nx, self.ny    =   csm.shape[1:3]
    def myAtA(self,img):
        with tf.name_scope('AtA'):
            coilImages      =       self.csm*tf.expand_dims(img,axis=0)
            kspace          =       tf.signal.fftshift(tf.signal.fft2d(tf.signal.fftshift(coilImages,axes=(1,2))),axes=(1,2)) # / tf.math.sqrt(tf.cast(self.nx*self.ny,tf.complex64))
            temp            =       kspace*tf.expand_dims(self.mask,axis=0)
            coilImgs        =       tf.signal.ifftshift(tf.signal.ifft2d(tf.signal.ifftshift(temp,axes=(1,2))),axes=(1,2)) # * tf.math.sqrt(tf.cast(self.nx*self.ny,tf.complex64))
            coilComb        =       tf.reduce_sum(coilImgs*tf.math.conj(self.csm),axis=0)
            coilComb        =       coilComb+self.lam*img
        return coilComb


def myCG(A,rhs):
    rhs=r2c(rhs)
    cond=lambda i,rTr,*_: tf.logical_and( tf.less(i,10), rTr>1e-8)
    def body(i,rTr,x,r,p):
        with tf.name_scope('cgBody'):
            Ap      =   A.myAtA(p)
            alpha   =   rTr / tf.cast(tf.reduce_sum(tf.math.conj(p)*Ap),dtype=tf.float32)
            alpha   =   tf.complex(alpha,0.)
            x       =   x + alpha * p
            r       =   r - alpha * Ap
            rTrNew  =   tf.cast( tf.reduce_sum(tf.math.conj(r)*r),dtype=tf.float32)
            beta    =   rTrNew / rTr
            beta    =   tf.complex(beta,0.)
            p       =   r + beta * p
        return i+1,rTrNew,x,r,p

    x       =   tf.zeros_like(rhs)
    i,r,p   =   0,rhs,rhs
    rTr     =   tf.cast( tf.reduce_sum(tf.math.conj(r)*r),dtype=tf.float32)
    loopVar =   i,rTr,x,r,p
    out     =   tf.while_loop(cond,body,loopVar,name='CGwhile',parallel_iterations=1)[2]
    return c2r(out)


class myDC(Layer):

    def __init__(self, **kwargs):
        super(myDC, self).__init__(**kwargs)        
        self.lam1 = self.add_weight(name='lam1', shape=(1,), initializer=tf.constant_initializer(value=0.03),
                                     dtype='float32', trainable=True)
        self.lam2 = self.add_weight(name='lam2', shape=(1,), initializer=tf.constant_initializer(value=0.03),
                                     dtype='float32', trainable=True)

    def build(self, input_shape):
        super(myDC, self).build(input_shape)

    def call(self, x):
        rhs, csm, mask = x
        lam3 = tf.complex(self.lam1 + self.lam2, 0.)

        def fn(tmp):
            c, m, r = tmp
            Aobj = Aclass(c, m, lam3)
            y = myCG(Aobj, r)
            return y
        
        inp = (csm, mask, rhs)
        # Mapping functions with multi-arity inputs and outputs
        rec = tf.map_fn(fn, inp, dtype=tf.float32, name='mapFn')
        return rec

    def lam_weight(self, x):
        in0, in1 = x
        res = self.lam1 * in0 + self.lam2 * in1
        return res


class Aty(Layer):
    def __init__(self, **kwargs):
        super(Aty, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Aty, self).build(input_shape)

    def call(self, x):
        kdata, csm, mask    =       x

        def backward(tmp):
            k, c, m = tmp
            nx, ny = k.shape[-2:]
            ks  =   k * tf.expand_dims(m,axis=0)
            ci  =   tf.signal.ifftshift(tf.signal.ifft2d(tf.signal.ifftshift(ks,axes=(-2,-1))),axes=(-2,-1)) # * tf.math.sqrt(tf.cast(nx*ny,tf.complex64))
            y1  =   tf.reduce_sum(ci*tf.math.conj(c),axis=0)
            return y1

        inp = (kdata, csm, mask)
        rec = tf.map_fn(backward, inp, dtype=tf.complex64, name='mapBack')

        return rec

def create_recon(nx, ny, nc, nLayers, num_block, num_filters = 64):

    # define the inputs
    input_c     =   Input(shape=(nc,nx,ny), dtype = tf.complex64,   name = 'input_c')
    input_m1    =   Input(shape=(nx,ny),    dtype = tf.complex64,   name = 'input_m1')    
    input_m2    =   Input(shape=(nx,ny),    dtype = tf.complex64,   name = 'input_m2')    
    input_k1    =   Input(shape=(nc,nx,ny), dtype = tf.complex64,   name = 'input_k1') 
    input_k2    =   Input(shape=(nc,nx,ny), dtype = tf.complex64,   name = 'input_k2') 
 
    # define functions
    UpdateDC    =   myDC()
    rmbg        =   rm_bg()    
    calc_Aty    =   Aty()
    myFFT       =   tfft2()
    myIFFT      =   tifft2()    

    # calc Atb
    Atb1        =   calc_Aty([input_k1,input_c,input_m1])
    Atb2        =   calc_Aty([input_k2,input_c,input_m2])
    
    # calc init
    dc1         =   c2r(Atb1)
    dc2         =   c2r(Atb2)
    
    # define networks
    RegConv_k   =   RegConvLayers(nx,ny,4,nLayers,num_filters)
    RegConv_i   =   RegConvLayers(nx,ny,4,nLayers,num_filters)
   
    # loop    
    for blk in range(0,num_block):          
        # concat shots with VC             
        dc_cat_i    = Concatenate(axis=-1)([dc1,dc2,tf.math.conj(dc1),tf.math.conj(dc2)])
        dc_cat_k    = Concatenate(axis=-1)([myFFT([dc1]),myFFT([dc2]),myFFT([tf.math.conj(dc1)]),myFFT([tf.math.conj(dc2)])])             
        # CNN Regularization
        rg_term_i   = RegConv_i(dc_cat_i)
        rg_term_k   = RegConv_k(dc_cat_k)    
        # separate shots        
        irg1        = (rg_term_i[:,:,:,0:2] + tf.math.conj(rg_term_i[:,:,:,4:6]))/2
        irg2        = (rg_term_i[:,:,:,2:4] + tf.math.conj(rg_term_i[:,:,:,6:8]))/2
        krg1        = (myIFFT([rg_term_k[:,:,:,0:2]]) + tf.math.conj(myIFFT([rg_term_k[:,:,:,4:6]])))/2
        krg2        = (myIFFT([rg_term_k[:,:,:,2:4]]) + tf.math.conj(myIFFT([rg_term_k[:,:,:,6:8]])))/2
        rg1         = UpdateDC.lam_weight([irg1,krg1])
        rg2         = UpdateDC.lam_weight([irg2,krg2])        
        # AtA update
        rg1         = Add()([c2r(Atb1), rg1])
        rg2         = Add()([c2r(Atb2), rg2])         
        # Update DC
        dc1         = UpdateDC([rg1, input_c, input_m1])      
        dc2         = UpdateDC([rg2, input_c, input_m2])      
    
    # remove background                
    out1 = rmbg([dc1,input_c])   
    out2 = rmbg([dc2,input_c])     

    return Model(inputs     =   [ input_c, input_m1, input_m2, input_k1, input_k2],
                 outputs    =   [ out1, out2],
                 name       =   'RECON' )

