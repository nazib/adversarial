import tensorflow as tf
from dense_3D_spatial_transformer import Dense3DSpatialTransformer
import  numpy as np
from keras.layers import Conv3D, Activation, Input, UpSampling3D, concatenate,MaxPooling3D, Add,Subtract
from keras.layers import LeakyReLU, Reshape, Lambda, BatchNormalization, Dense, Flatten, ReLU,Dropout
from keras.initializers import RandomNormal
from DenseTranform_tf import *
import sys

#sys.path.append('/home/n9614885/adversarial/pynd-lib/pynd')
sys.path.append('/home/n9614885/adversarial/pytools-lib/')
sys.path.append('/home/n9614885/adversarial/neuron')


import neuron.layers as nrn
import neuron.utils as nrn_utils

class InverseNetDense_tf:

    def __init__(self,src,tgt,vol_size,batch_size,reuse=False):

        self.src =src
        self.tgt =tgt
        self.reuse =reuse

    def Encoder(self):
        x  = tf.concat([self.src,self.tgt],4)
        self.E_l1 = self.Downsample(x,32,2,2)
        self.E_l2 = self.Downsample(self.E_l1, 32, 2, 2)
        self.E_l3 = self.Downsample(self.E_l2, 32, 2, 2)
        self.E_l4 = self.Downsample(self.E_l3, 32, 2, 2)

        return self.E_l4

    def FowwardFlow(self):

        x = self.Conv(self.E_l4, 32, 3, 1)
        x = x + self.E_l4
        x = tf.concat([x,self.E_l4],4)

        x = self.Upsample(x, 32, 2, 2)
        x = self.Conv(x,32,3,1)
        E_l1_3 = self.Downsample(self.E_l1, 32, 3, 4)
        E_l2_3 = self.Downsample(self.E_l2, 32, 3, 2)
        x = x + self.E_l3 + E_l1_3 + E_l2_3
        x = tf.concat([x, self.E_l3,E_l1_3,E_l2_3], 4)

        x = self.Upsample(x, 32, 2, 2)
        x = self.Conv(x, 32, 3, 1)
        E_l1_2 = self.Downsample(self.E_l1, 32, 2, 2)
        E_l3_2 = self.Upsample(self.E_l3, 32, 2, 2)
        x = x + self.E_l2 + E_l1_2 + E_l3_2
        x = tf.concat([x, self.E_l2,E_l1_2,E_l3_2], 4)

        x = self.Upsample(x, 32, 2, 2)
        x = self.Conv(x, 32, 3, 1)
        E_l2_1 = self.Upsample(self.E_l2, 32, 2, 2)
        E_l3_1 = self.Upsample(self.E_l3, 32, 2, 4)
        x = x + self.E_l1 + E_l2_1 + E_l3_1
        x = tf.concat([x, self.E_l1, E_l2_1, E_l3_1], 4)

        x = self.Upsample(x, 32, 2, 2)
        x = self.Conv(x,8,3,1)
        #flow = self.Conv(x, 3, 3, 1)
        flow = tf.layers.conv3d(x, 3, 3, 1,padding="same", kernel_initializer = tf.truncated_normal_initializer(mean=0.0 ,stddev=1e-5))
        return flow

    def InverseFlow(self):

        x = self.Conv(self.E_l4, 32, 3, 1)
        x = x - self.E_l4
        x = tf.concat([x, self.E_l4], 4)

        x = self.Upsample(x, 32, 2, 2)
        x = self.Conv(x, 32, 3, 1)
        E_l1_3 = self.Downsample(self.E_l1, 32, 2, 4)
        E_l2_3 = self.Downsample(self.E_l2, 32, 2, 2)
        x = x - self.E_l3 - E_l1_3 - E_l2_3
        x = tf.concat([x, self.E_l3, E_l1_3, E_l2_3], 4)

        x = self.Upsample(x, 32, 2, 2)
        x = self.Conv(x, 32, 3, 1)
        E_l1_2 = self.Downsample(self.E_l1, 32, 2, 2)
        E_l3_2 = self.Upsample(self.E_l3, 32, 2, 2)
        x = x - self.E_l2 - E_l1_2 - E_l3_2
        x = tf.concat([x, self.E_l2, E_l1_2, E_l3_2], 4)

        x = self.Upsample(x, 32, 2, 2)
        x = self.Conv(x, 32, 3, 1)
        E_l2_1 = self.Upsample(self.E_l2, 32, 2, 2)
        E_l3_1 = self.Upsample(self.E_l3, 32, 2, 4)
        x = x - self.E_l1 - E_l2_1 - E_l3_1
        x = tf.concat([x, self.E_l1, E_l2_1, E_l3_1], 4)

        x = self.Upsample(x, 32, 2, 2)
        x = self.Conv(x, 8, 3, 1)
        # flow = self.Conv(x, 3, 3, 1)
        flow = tf.layers.conv3d(x, 3, 3, 1, padding="same",
                                kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1e-5))
        return flow

    def Conv(self,volume,output,kernel,stride):

        x = tf.layers.conv3d(volume, output, kernel, stride,padding="same")
        x =  tf.nn.leaky_relu(x,0.2)
        return x

    def Downsample(self,volume,output,kernel,stride):
        x = tf.layers.conv3d(volume, output, kernel, stride,padding="same")
        x = tf.nn.leaky_relu(x, 0.2)
        return x

    def Upsample(self,volume,output,kernel,stride):
        x = tf.layers.conv3d_transpose(volume,output,kernel,stride,padding="same")
        x = tf.nn.leaky_relu(x, 0.2)
        return x

    def Build(self):

        with tf.variable_scope("Generator",reuse=self.reuse):
            Enc = self.Encoder()
            f_flow = self.FowwardFlow()
            i_flow = self.InverseFlow()

            #f_warp = Dense3DSpatialTransformer()([self.src,f_flow])
            #i_warp = Dense3DSpatialTransformer()([self.tgt, i_flow])

            f_warp = nrn.SpatialTransformer(interp_method='linear', indexing='ij')([self.src, f_flow])
            src_cyc = nrn.SpatialTransformer(interp_method='linear', indexing='ij')([f_warp, i_flow])
            i_warp = nrn.SpatialTransformer(interp_method='linear', indexing='ij')([self.tgt, i_flow])
            tgt_cyc = nrn.SpatialTransformer(interp_method='linear', indexing='ij')([i_warp, f_flow])

            f_warp = tf.identity(f_warp,"Forward_im")
            i_warp = tf.identity(i_warp, "Inverse_im")
            Flow = tf.concat([f_flow, i_flow], 4)

        return f_warp, i_warp, src_cyc, tgt_cyc, Flow

def discriminator(Input1, Input2, type, reuse = False):
    '''
    ## Discriminator for 10% Resolution ##
    with tf.variable_scope("Discriminator_"+type, reuse=reuse):
        x = tf.concat([Input1, Input2],4)
        x = tf.layers.conv3d(x, 2, 3, 1, padding="same")
        x = tf.nn.leaky_relu(x,0.2)
        x = tf.layers.flatten(x)
        x = tf.layers.dense(inputs=x, units=512, activation=tf.nn.leaky_relu)
        x = tf.layers.batch_normalization(inputs=x)
        x = tf.layers.dense(inputs=x, units=256, activation=tf.nn.leaky_relu)
        x = tf.layers.batch_normalization(inputs=x)
        x = tf.layers.dense(inputs=x, units=128, activation=tf.nn.leaky_relu)
        x = tf.layers.batch_normalization(inputs=x)
        x = tf.layers.dense(inputs=x, units=1, activation=tf.nn.leaky_relu)
        x = tf.layers.batch_normalization(inputs=x)
        x = tf.sigmoid(x)
    '''
    with tf.variable_scope("Discriminator_" + type, reuse=reuse):
        x = tf.concat([Input1, Input2], 4) #Input
        x = tf.layers.conv3d(x, 2, 2, 2)
        x = tf.nn.leaky_relu(x, 0.2)
        x = tf.layers.conv3d(x, 8, 2, 2)
        x = tf.nn.leaky_relu(x, 0.2)
        x = tf.layers.conv3d(x, 16, 2, 2)
        x = tf.nn.leaky_relu(x, 0.2)
        x = tf.layers.conv3d(x, 32, 2, 2)
        x = tf.nn.leaky_relu(x, 0.2)
        x = tf.layers.conv3d(x, 64, 2, 2)
        x = tf.nn.leaky_relu(x, 0.2)
        x = tf.layers.conv3d(x, 1, 2, 2)
        x = tf.nn.leaky_relu(x, 0.2)
        x = tf.layers.flatten(x)
        #x = tf.tanh(x,name='Tanh')
        #x = tf.sigmoid(x)
        if not reuse:
            x = tf.identity(x, "Real")
        else:
            x = tf.identity(x, "Fake")
    return x













