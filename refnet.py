# third party
import tensorflow as tf
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.layers import Conv3D, Activation, Input, UpSampling3D, concatenate
from keras.layers import LeakyReLU, Reshape, Lambda, BatchNormalization, Dense, Flatten
from keras.initializers import RandomNormal
import keras
import numpy as np
from keras import backend as K
# local
from dense_3D_spatial_transformer import Dense3DSpatialTransformer
#import ref_loss
import keras.utils


def ref_net(vol_size, enc_nf, dec_nf):
    
    src = Input(shape=vol_size + (1,))
    tgt = Input(shape=vol_size + (1,))

    out_img,flow,warped=auto_encoder(src,tgt,vol_size, enc_nf, dec_nf)

    recons_Model= Model(inputs=[src,tgt],outputs=[warped,flow])
    return recons_Model

def auto_encoder(src,tgt,vol_size, enc_nf, dec_nf):

    x_in = concatenate([src, tgt])
    x0 = myConv(x_in, enc_nf[0], 2)  # 80x96x112
    #x0 = BatchNormalization()(x0)
    x1 = myConv(x0, enc_nf[1], 2)  # 40x48x56
    #x1 = BatchNormalization()(x1)
    x2 = myConv(x1, enc_nf[2], 2)  # 20x24x28
    #x2 = BatchNormalization()(x2)
    x3 = myConv(x2, enc_nf[3], 2)  # 10x12x14
    x3 = BatchNormalization()(x3)

    
    x = myConv(x3, dec_nf[0])
    x = UpSampling3D()(x)
    #x= BatchNormalization()(x)
    x = concatenate([x, x2])
    x = myConv(x, dec_nf[1])
    x = UpSampling3D()(x)
    x = concatenate([x, x1])
    #x= BatchNormalization()(x)
    x = myConv(x, dec_nf[2])
    x = UpSampling3D()(x)
    x = concatenate([x, x0])
    #x= BatchNormalization()(x)
    x = myConv(x, dec_nf[3])
    x = myConv(x, dec_nf[4])

    x = UpSampling3D()(x)
    x = concatenate([x, src])
    #x= BatchNormalization()(x)
    out_img=myConv(x,dec_nf[-1],1)
    #flow = Conv3D(3, kernel_size=3, padding='same',
    #               kernel_initializer='he_normal', strides=1)(out_img)

    flow=Conv3D(3, kernel_size=3, padding='same',
                  kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='flow')(out_img)
    
    warped=Dense3DSpatialTransformer()([src,flow])

    return out_img,flow,warped


def myConv(x_in, nf, strides=1):
    x_out = Conv3D(nf, kernel_size=3, padding='same',
                   kernel_initializer='he_normal', strides=strides)(x_in)
    x_out = LeakyReLU(0.2)(x_out)
    return x_out

