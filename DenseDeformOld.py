import tensorflow as tf
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.layers import Conv3D, Activation, Input, UpSampling3D, concatenate,MaxPooling3D, Multiply
from keras.layers import LeakyReLU, Reshape, Lambda, BatchNormalization, Dense, Flatten, ReLU,Dropout
from keras.initializers import RandomNormal
import keras
import numpy as np
from keras import backend as K
# local
from dense_3D_spatial_transformer import Dense3DSpatialTransformer
#import ref_loss
import keras.utils

class DenseDeformNet:
    def __init__(self,vol_size,isHalfNet,isBN,dropout_rate=0.2):
        self.init = tf.truncated_normal_initializer(stddev=0.01)
        self.dropout_rate = dropout_rate
        self.vol_size=vol_size
        self.isBN =isBN
        self.isHalfNet = isHalfNet

    def createModel(self):

        src = Input(shape=self.vol_size + (1,))
        tgt = Input(shape=self.vol_size + (1,))

        x_in = concatenate([src, tgt])
        conv1 = self.myConv(x_in,16,2)
        ########## Dense Block 1 with 32 filters #############
        dense_bl1 = self.denseBlock(conv1,4)

        ########## Transformation Layer #####################
        conv14, pooling = self.transformLayer(dense_bl1,64)
            
        ###########  Dense Block 2 with 32 filters ##########
        dense_bl2 = self.denseBlock(pooling,4)
        
        bn = BatchNormalization()(dense_bl2)

        conv27 = self.myConv(bn,112)

        ############## Upsampling Layers ################
        deconv1 = UpSampling3D()(conv27)
        deconv1 = self.myConv(deconv1,64)
        deconv2 = UpSampling3D()(deconv1)
        deconv2 = self.myConv(deconv2,8)
        deconv2 = self.myConv(deconv2,3)
        
        ############## Flow estimation #################
        conv14 = UpSampling3D()(conv14)
        g_flow = Conv3D(3, kernel_size=3, padding='same',
                  kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='g_flow')(conv14)
        
        l_flow = Conv3D(3, kernel_size=3, padding='same',
                  kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='l_flow')(deconv2)
        
        #c_flow = Multiply()([g_flow,l_flow])
        c_flow = concatenate([g_flow,l_flow])

        flow = Conv3D(3, kernel_size=3, padding='same',
                  kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='flow')(c_flow)
        
        warped = Dense3DSpatialTransformer()([src,flow])

        dense_deform = Model(inputs=[src,tgt],outputs=[warped,flow])

        return dense_deform

    def myConv(self,x_in, nf, strides=1):
        x_out = Conv3D(nf, kernel_size=3, padding='same',kernel_initializer='he_normal', strides=strides)(x_in)
        x_out = LeakyReLU(0.2)(x_out)
        return x_out

    def denseBlock(self, x,nf, n_layers=12):
        for _ in range(n_layers):
            x = self.unitLayer(x,nf)
        return x

    def unitLayer(self, x, nf):
        #bn = ReLU()(BatchNormalization()(x))
        bn = ReLU()(x)
        conv = self.myConv(bn,nf)
        drop = Dropout(self.dropout_rate)(conv)
        return concatenate([drop, x], axis=4)

    def transformLayer(self, x, n_filters):
        #bn = ReLU()(BatchNormalization()(x))
        bn = ReLU()(x)
        conv = self.myConv(bn,n_filters)
        drop = Dropout(self.dropout_rate)(conv)
        pool = MaxPooling3D(pool_size=2,strides=2)(drop)
        return conv, pool

    def batch_normalization(self, x, training):
        training = tf.constant(training)
        depth = x.get_shape()[-1]
        beta = tf.Variable(tf.constant(0.0, shape=[depth]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[depth]), name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2, 3], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.99)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(training,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed_tensor = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return normed_tensor
