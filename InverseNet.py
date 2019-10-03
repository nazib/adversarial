import tensorflow as tf
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.layers import Conv3D, Activation, Input, UpSampling3D, concatenate,MaxPooling3D, Add,Subtract
from keras.layers import LeakyReLU, Reshape, Lambda, BatchNormalization, Dense, Flatten, ReLU,Dropout
from keras.initializers import RandomNormal
import keras
import numpy as np
from keras import backend as K
# local
from dense_3D_spatial_transformer import Dense3DSpatialTransformer
#import ref_loss
import keras.utils

class InverseNet:
    def __init__(self,vol_size,dilation,dropout_rate=0.2):
        self.init = tf.truncated_normal_initializer(stddev=0.01)
        self.dropout_rate = dropout_rate
        self.vol_size=vol_size
        self.isDilation =dilation

        if self.isDilation == 'on':
            self.inFactors =[1,1,2,2,3,3,4,4,5,5,6,6]
            self.decFectors =[6,6,5,5,4,4,3,3,2,2,1,1]

    def myConv(self,x_in, nf, strides=1,rate=1):

        if self.isDilation == 'off':
            x_out = Conv3D(nf, kernel_size=3, padding='same',kernel_initializer='he_normal', strides=strides)(x_in)
            x_out = LeakyReLU(0.2)(x_out)
        else:
            x_out = Conv3D(nf,kernel_size=3, padding='same',kernel_initializer='he_normal',strides=strides,dilation_rate=rate)(x_in)
            x_out = LeakyReLU(0.2)(x_out)

        return x_out

    def FowardFlow(self):

        x = self.myConv(self.Encoder_layer3, 32)
        x = UpSampling3D()(x)
        x = Add()([x ,self.Encoder_layer2])
        x = concatenate([x, self.Encoder_layer2])
        x = self.myConv(x, 32)
        x = UpSampling3D()(x)
        x = Add()([x ,self.Encoder_layer1])
        x = concatenate([x, self.Encoder_layer1])
        x = self.myConv(x, 32)
        x = UpSampling3D()(x)
        x = Add()([x ,self.Encoder_layer0])
        x = concatenate([x, self.Encoder_layer0])
        x = self.myConv(x, 32)
        x = self.myConv(x, 8)
        x = UpSampling3D()(x)
        #x = x + self.Encoder_layer_in
        x = concatenate([x, self.Encoder_layer_in])
        x = self.myConv(x, 8)

        f_flow = Conv3D(3, kernel_size=3, padding='same',
                        kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='f_flow')(x)

        return  f_flow

    def InverseFlow(self):

        self.isDilation = "on"

        x = self.myConv(self.Encoder_layer3, 32)
        x = UpSampling3D()(x)
        x = Subtract()([x,self.Encoder_layer2])
        x = concatenate([x, self.Encoder_layer2])
        x = self.myConv(x, 32)
        x = UpSampling3D()(x)
        x = Subtract()([x,self.Encoder_layer1])
        x = concatenate([x, self.Encoder_layer1])
        x = self.myConv(x, 32)
        x = UpSampling3D()(x)
        x = Subtract()([x,self.Encoder_layer0])
        x = concatenate([x, self.Encoder_layer0])
        x = self.myConv(x, 32)
        x = self.myConv(x, 8)
        x = UpSampling3D()(x)
        #x = x - self.Encoder_layer_in
        x = concatenate([x, self.Encoder_layer_in])
        x = self.myConv(x, 8)

        i_flow = Conv3D(3, kernel_size=3, padding='same',
                        kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='i_flow')(x)

        return i_flow

    def createModel(self):

        src = Input(shape=self.vol_size + (1,))
        tgt = Input(shape=self.vol_size + (1,))

        #inp_fake1 =Input(shape=self.vol_size + (1,))
        #inp_fake2 =Input(shape=self.vol_size + (1,))
        self.Encoder_layer_in = concatenate([src, tgt])

        # Common Encoder
        #self.Encoder_layer_in = Input(shape=self.vol_size+(4,))
        self.Encoder_layer0 = self.myConv(self.Encoder_layer_in, 32, 2)
        self.Encoder_layer1 = self.myConv(self.Encoder_layer0, 32, 2)
        self.Encoder_layer2 = self.myConv(self.Encoder_layer1, 32, 2)
        self.Encoder_layer3 = self.myConv(self.Encoder_layer2, 32, 2)

        F_flow = self.FowardFlow()
        I_flow = self.InverseFlow()
        
        # Warping
        #y = tf.unstack(self.Encoder_layer_in, num=None, axis=4)
        #src= Reshape([64,64,32,1])(y[0])
        #tgt= Reshape([64, 64, 32, 1])(y[1])

        fwarped = Dense3DSpatialTransformer()([src,F_flow])
        iwarped = Dense3DSpatialTransformer()([tgt,I_flow])

        #out = concatenate([fwarped,iwarped])

        #inverse_net = Model(inputs= self.Encoder_layer_in, outputs=out)
        inverse_net = Model(inputs=[src,tgt], outputs=[fwarped,iwarped])

        return inverse_net
