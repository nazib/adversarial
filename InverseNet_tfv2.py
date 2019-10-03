import tensorflow as tf
from DenseTranform_tf import *
import sys
#sys.path.append('/home/n9614885/adversarial/pynd-lib/pynd')
sys.path.append('/home/n9614885/adversarial/pytools-lib/')
sys.path.append('/home/n9614885/adversarial/neuron')


import neuron.layers as nrn
import neuron.utils as nrn_utils

class InverseNet_tfv2:

    def __init__(self, src, tgt, diff, vol_size, batch_size, reuse=False):
        self.src = src
        self.tgt = tgt
        self.diff = diff
        self.reuse = reuse
        self.latend_dim = 4

    def Encoder(self,img):
        #x  = tf.concat([self.src,self.tgt],4)
        if img == "src":
            x = self.src
            self.srcEncoder_layer0 = self.Downsample(x,32,2,2)
            self.srcEncoder_layer1 = self.Downsample(self.srcEncoder_layer0, 32, 2, 2)
            self.srcEncoder_layer2 = self.Downsample(self.srcEncoder_layer1, 32, 2, 2)
            self.srcEncoder_layer3 = self.Downsample(self.srcEncoder_layer2, 32, 2, 2)

        elif img == "tgt":
            x = self.tgt
            self.tgtEncoder_layer0 = self.Downsample(x, 32, 2, 2)
            self.tgtEncoder_layer1 = self.Downsample(self.tgtEncoder_layer0, 32, 2, 2)
            self.tgtEncoder_layer2 = self.Downsample(self.tgtEncoder_layer1, 32, 2, 2)
            self.tgtEncoder_layer3 = self.Downsample(self.tgtEncoder_layer2, 32, 2, 2)
        else:
            x = self.diff
            self.diffEncoder_layer0 = self.Downsample(x, 32, 2, 2)
            self.layer0_f_flow, self.layer0_b_flow = self.Distribution(self.diffEncoder_layer0)
            self.diffEncoder_layer1 = self.Downsample(self.diffEncoder_layer0, 32, 2, 2)
            self.layer1_f_flow, self.layer1_b_flow = self.Distribution(self.diffEncoder_layer1)

            self.diffEncoder_layer2 = self.Downsample(self.diffEncoder_layer1, 32, 2, 2)
            self.layer2_f_flow, self.layer2_b_flow = self.Distribution(self.diffEncoder_layer2)
            self.diffEncoder_layer3 = self.Downsample(self.diffEncoder_layer2, 32, 2, 2)
            self.layer3_f_flow, self.layer3_b_flow = self.Distribution(self.diffEncoder_layer3)

    def FowwardFlow(self):
        warped = nrn.SpatialTransformer(interp_method='linear', indexing='ij')([self.srcEncoder_layer3, self.layer3_f_flow])
        x = self.Conv(self.tgtEncoder_layer3,32,3,1)
        #x = x + self.Encoder_layer3
        x = tf.concat([warped,x],4)

        x = self.Upsample(x,32,2,2)
        x = self.Conv(x, 32, 3, 1)
        #x = x + self.Encoder_layer2
        warped = nrn.SpatialTransformer(interp_method='linear', indexing='ij')([self.srcEncoder_layer2, self.layer2_f_flow])
        x = tf.concat([warped, x], 4)

        x = self.Upsample(x, 32, 2, 2)
        x = self.Conv(x, 32, 3, 1)
        #x = x + self.Encoder_layer1
        warped = nrn.SpatialTransformer(interp_method='linear', indexing='ij')([self.srcEncoder_layer1, self.layer1_f_flow])
        x = tf.concat([warped, x], 4)

        x = self.Upsample(x, 32, 2, 2)
        x = self.Conv(x, 32, 3, 1)
        #x = x + self.Encoder_layer0
        warped = nrn.SpatialTransformer(interp_method='linear', indexing='ij')([self.srcEncoder_layer0, self.layer0_f_flow])
        x = tf.concat([warped, x], 4)
        x = self.Upsample(x, 32, 2, 2)
        x = self.Conv(x,8,3,1)
        #flow = self.Conv(x, 3, 3, 1)
        flow = tf.layers.conv3d(x, 3, 3, 1,padding="same", kernel_initializer = tf.truncated_normal_initializer(mean=0.0 ,stddev=1e-5))
        return flow

    def InverseFlow(self):
        x = self.Conv(self.srcEncoder_layer3, 32, 3, 1)
        #x = x - self.Encoder_layer3
        warped = nrn.SpatialTransformer(interp_method='linear', indexing='ij')([self.tgtEncoder_layer3, self.layer3_b_flow])
        x = tf.concat([warped, x], 4)

        x = self.Upsample(x, 32, 2, 2)
        x = self.Conv(x, 32, 3, 1)
        #x = x - self.Encoder_layer2
        warped = nrn.SpatialTransformer(interp_method='linear', indexing='ij')([self.tgtEncoder_layer2, self.layer2_b_flow])
        x = tf.concat([warped, x], 4)

        x = self.Upsample(x, 32, 2, 2)
        x = self.Conv(x, 32, 3, 1)
        warped = nrn.SpatialTransformer(interp_method='linear', indexing='ij')([self.tgtEncoder_layer1, self.layer1_b_flow])
        #x = x - self.Encoder_layer1
        x = tf.concat([warped, x], 4)

        x = self.Upsample(x, 32, 2, 2)
        x = self.Conv(x, 32, 3, 1)
        #x = x - self.Encoder_layer0
        warped = nrn.SpatialTransformer(interp_method='linear', indexing='ij')([self.tgtEncoder_layer0, self.layer0_b_flow])
        x = tf.concat([warped, x], 4)

        x = self.Upsample(x,32,2,2)
        x = self.Conv(x, 8, 3, 1)
        #flow = self.Conv(x, 3, 3, 1,)
        flow = tf.layers.conv3d(x, 3, 3, 1, padding="same", kernel_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=1e-5))
        return flow

    def Conv(self,volume,output,kernel,stride):
        x = tf.layers.conv3d(volume, output, kernel, stride, padding="same")
        x =  tf.nn.leaky_relu(x,0.2)
        return x

    def Downsample(self,volume,output,kernel,stride):
        x = tf.layers.conv3d(volume, output, kernel, stride)
        x = tf.nn.leaky_relu(x, 0.2)
        return x

    def Upsample(self,volume,output,kernel,stride):
        x = tf.layers.conv3d_transpose(volume,output,kernel,stride)
        x = tf.nn.leaky_relu(x, 0.2)
        return x

    def Reparameterize(self, mean, logvar, size):
        size = [3,size[1],size[2],size[3]]
        pdf = tf.distributions.Normal(loc=mean,scale=logvar)
        flow = pdf.sample(size)
        flow = tf.squeeze(flow, axis=5)
        flow = tf.transpose(flow)
        return flow

    def Distribution(self, feature):
        size = feature.get_shape().as_list()
        feature = tf.layers.flatten(feature)
        feature = tf.layers.dense(feature,self.latend_dim,activation=None)
        meanF, logvarF, meanB, logvarB = tf.split(feature, num_or_size_splits=4, axis=1)
        flowF = self.Reparameterize(meanF, logvarF, size)
        flowB = self.Reparameterize(meanB, logvarB, size)
        return flowF, flowB

    def Build(self):

        with tf.variable_scope("Generator",reuse=self.reuse):
            self.Encoder("src")
            self.Encoder("tgt")
            self.Encoder("diff")

            f_flow = self.FowwardFlow()
            i_flow = self.InverseFlow()
            #f_warp = Dense3DSpatialTransformer()([self.src,f_flow])
            #i_warp = Dense3DSpatialTransformer()([self.tgt, i_flow])

            f_warp = nrn.SpatialTransformer(interp_method='linear', indexing='ij')([self.src, f_flow])
            i_warp = nrn.SpatialTransformer(interp_method='linear', indexing='ij')([self.tgt, i_flow])

            f_warp = tf.identity(f_warp,"Forward_im")
            i_warp = tf.identity(i_warp, "Inverse_im")

            f_flow = tf.identity(f_flow, "Forward_flow")
            i_flow = tf.identity(i_flow, "Inverse_flow")
        return f_warp, i_warp, f_flow, i_flow

class Discriminator:

    def __init__(self, Input,vol_size,batch_size,reuse=False):
        self.Input = Input
        self.reuse = reuse

    def Conv(self,volume,output,kernel,stride):

        x = tf.layers.conv3d(volume, output, kernel, stride,padding="same")
        x =  tf.nn.leaky_relu(x,0.2)
        return x

    def Downsample(self,volume,output,kernel,stride):
        x = tf.layers.conv3d(volume, output, kernel, stride)
        x = tf.nn.leaky_relu(x, 0.2)
        return x

    def Build(self):
        with tf.variable_scope("Discriminator",reuse=self.reuse):
            x = self.Input
            x = tf.layers.flatten(x)
            x = tf.layers.dense(inputs=x,units=512,activation=tf.nn.leaky_relu)
            x = tf.layers.dense(inputs=x, units=256, activation=tf.nn.leaky_relu)
            x = tf.layers.dense(inputs=x, units=128, activation=tf.nn.leaky_relu)
            x = tf.layers.dense(inputs=x, units=1, activation=tf.nn.leaky_relu)
            x = tf.sigmoid(x)

        return x


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
        x = tf.concat([Input1, Input2], 4)
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
        x = tf.layers.conv3d(x, 1, 2, 1,padding="same")
        #x = tf.nn.leaky_relu(x, 0.2)
        x = tf.layers.flatten(x)
        #x = tf.tanh(x,name='Tanh')
        #x = tf.sigmoid(x)
        if not reuse:
            x = tf.identity(x, "Real")
        else:
            x = tf.identity(x, "Fake")
    return x













