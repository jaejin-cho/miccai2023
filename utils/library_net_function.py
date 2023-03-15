##########################################################
# %%
# import libraries
##########################################################

# import numpy                as  np
import tensorflow           as  tf

# keras
from tensorflow.keras import backend as K
from tensorflow.keras import Input
from tensorflow.keras.layers import Layer, Conv2D, Activation, BatchNormalization, \
    Add, Conv2DTranspose, LeakyReLU, Lambda 
from tensorflow.keras.models import Model

##########################################################
# %%
# functions
##########################################################

c2r=Lambda(lambda x:tf.stack([tf.math.real(x),tf.math.imag(x)],axis=-1))
r2c=Lambda(lambda x:tf.complex(x[...,0],x[...,1]))

class tfft2(Layer):
    def __init__(self, **kwargs):
        super(tfft2, self).__init__(**kwargs)
    def build(self, input_shape):
        super(tfft2, self).build(input_shape)
    def call(self, x):
        xc = r2c(x[0])
        # fft2 over last two dimension
        xt = tf.signal.fftshift(xc, axes=(-2,-1))
        kt = tf.signal.fft2d(xt) 
        kt = tf.signal.fftshift(kt, axes=(-2,-1))        
        return c2r(kt)

class tifft2(Layer):
    def __init__(self, **kwargs):
        super(tifft2, self).__init__(**kwargs)
    def build(self, input_shape):
        super(tifft2, self).build(input_shape)
    def call(self, x):
        xc = r2c(x[0])        
        # ifft2 over last two dimension
        it = tf.signal.ifftshift(xc, axes=(-2,-1))
        it = tf.signal.ifft2d(it) 
        it = tf.signal.ifftshift(it, axes=(-2,-1))   
        return c2r(it)

class rm_bg(Layer):
    def __init__(self, **kwargs):
        super(rm_bg, self).__init__(**kwargs)

    def build(self, input_shape):
        super(rm_bg, self).build(input_shape)

    def call(self, x):
        img, csm    = x
        rcsm        = tf.expand_dims(tf.reduce_sum(tf.math.abs(csm),axis=1), axis=-1)
        cmask       = tf.math.greater(rcsm,tf.constant(0,dtype=tf.float32))
        rec         = tf.cast(cmask,dtype=tf.float32) * img
        return rec
    
def conv2D_bn_nonlinear(x, num_out_chan, kernel_size, activation_type='relu', USE_BN=True, layer_name='', strides = (1, 1), alpha = 0.3 ):
    with K.name_scope(layer_name):
        x = Conv2D(num_out_chan, kernel_size, activation=None, padding='same', kernel_initializer='truncated_normal', strides=strides)(x)
        if USE_BN:
            x = BatchNormalization()(x)
        if activation_type == 'LeakyReLU':
            return LeakyReLU(alpha=alpha)(x)
        else:
            return Activation(activation_type)(x)

def conv2Dt_bn_nonlinear(x, num_out_chan, kernel_size, activation_type='relu', USE_BN=True, layer_name=''):
    with K.name_scope(layer_name):
        x = Conv2DTranspose(num_out_chan, kernel_size, strides=(2, 2), padding='same',
                            kernel_initializer='truncated_normal')(x)
        if USE_BN:
            x = BatchNormalization()(x)

        if activation_type == 'LeakyReLU':
            return LeakyReLU()(x)
        else:
            return Activation(activation_type)(x)
              
def RegConvLayers(nx,ny,ne,nLayers,num_filters):
    
    input_x     = Input(shape=(nx,ny,2*ne), dtype = tf.float32)
    filter_size = (3,3)
    bool_USE_BN = True
    AT          = 'LeakyReLU'

    rg_term     = input_x
    for lyr in range(0,nLayers):
        rg_term = conv2D_bn_nonlinear(rg_term, num_filters, filter_size, activation_type=AT, USE_BN=bool_USE_BN, layer_name='')

    # go to image space
    rg_term = conv2D_bn_nonlinear(rg_term, 2*ne, (1,1), activation_type='linear', USE_BN=False, layer_name='')

    # skip connection
    rg_term = Add()([rg_term,input_x])

    return Model(inputs     =   input_x, outputs    =   rg_term)

