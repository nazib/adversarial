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
            #self.layer0_f_flow, self.layer0_b_flow, self.layer0_muF, self.layer0_sigF, self.layer0_muB, self.layer0_sigB = self.Distribution(self.diffEncoder_layer0)
            self.diffEncoder_layer1 = self.Downsample(self.diffEncoder_layer0, 32, 2, 2)
            #self.layer1_f_flow, self.layer1_b_flow, self.layer1_muF, self.layer1_sigF, self.layer1_muB, self.layer1_sigB  = self.Distribution(self.diffEncoder_layer1)

            self.diffEncoder_layer2 = self.Downsample(self.diffEncoder_layer1, 32, 2, 2)
            #self.layer2_f_flow, self.layer2_b_flow, self.layer2_muF, self.layer2_sigF, self.layer2_muB, self.layer2_sigB  = self.Distribution(self.diffEncoder_layer2)
            self.diffEncoder_layer3 = self.Downsample(self.diffEncoder_layer2, 32, 2, 2)
            self.layer3_f_flow, self.layer3_b_flow, self.layer3_muF, self.layer3_sigF, self.layer3_muB, self.layer3_sigB  = self.Distribution(self.diffEncoder_layer3)

    def FowwardFlow(self):
        warped = nrn.SpatialTransformer(interp_method='linear', indexing='ij')([self.srcEncoder_layer3, self.layer3_f_flow])
        x = tf.concat([self.tgtEncoder_layer3,warped],4)
        x = self.Conv(x, 32, 3, 1)
        #x = tf.concat([x, self.layer3_f_flow], 4)

        Ef3 = self.Upsample(x,3,2,2)
        warped = nrn.SpatialTransformer(interp_method='linear', indexing='ij')(
                                        [self.srcEncoder_layer2, Ef3])
        x = tf.concat([self.tgtEncoder_layer2,warped], 4)
        x = self.Conv(x, 32, 3, 1)
        #x = tf.concat([x, self.layer2_f_flow], 4)

        Ef2 = self.Upsample(x, 3, 2, 2)
        warped = nrn.SpatialTransformer(interp_method='linear', indexing='ij')([self.srcEncoder_layer1, Ef2])
        x = tf.concat([self.tgtEncoder_layer1, warped],4)
        #x = tf.concat([x, self.layer1_f_flow], 4)

        Ef1 = self.Upsample(x, 3, 2, 2)
        warped = nrn.SpatialTransformer(interp_method='linear', indexing='ij')([self.srcEncoder_layer0, Ef1])
        x = tf.concat([self.tgtEncoder_layer0,warped],4)
        #x = tf.concat([x, self.layer0_f_flow], 4)

        x = self.Upsample(x, 3, 2, 2)
        x = self.Conv(x,8,3,1)
        flow = self.Conv(x, 3, 3, 1)
        flow = tf.layers.conv3d(x, 3, 3, 1,padding="same", kernel_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=1e-5))
        return flow

    def InverseFlow(self):
        warped = nrn.SpatialTransformer(interp_method='linear', indexing='ij')([self.tgtEncoder_layer3, self.layer3_b_flow])
        x = tf.concat([self.srcEncoder_layer3,warped],4)
        x = self.Conv(x, 32, 3, 1)
        #x = tf.concat([x, self.layer3_b_flow], 4)

        Ef3 = self.Upsample(x, 3, 2, 2)
        warped = nrn.SpatialTransformer(interp_method='linear', indexing='ij')([self.tgtEncoder_layer2, Ef3])
        x = tf.concat([self.srcEncoder_layer2,warped], 4)
        x = self.Conv(x, 32, 3, 1)
        #x = tf.concat([x, self.layer2_b_flow], 4)

        Ef2 = self.Upsample(x, 3, 2, 2)
        warped = nrn.SpatialTransformer(interp_method='linear', indexing='ij')([self.tgtEncoder_layer1, Ef2])
        x = tf.concat([self.srcEncoder_layer1,warped],4)
        x = self.Conv(x, 32, 3, 1)
        #x = tf.concat([x, self.layer1_b_flow], 4)

        Ef1 = self.Upsample(x, 3, 2, 2)
        warped = nrn.SpatialTransformer(interp_method='linear', indexing='ij')([self.tgtEncoder_layer3, Ef1])
        x = tf.concat([self.srcEncoder_layer0,warped],4)
        x = self.Conv(x, 32, 3, 1)
        #x = tf.concat([x, self.layer0_b_flow], 4)

        x = self.Upsample(x,3,2,2)
        x = self.Conv(x, 8, 3, 1)
        flow = self.Conv(x, 3, 3, 1,)
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

    def sample(self,mu, log_sigma):
        noise = tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
        z = mu + tf.exp(log_sigma/2.0) * noise
        return z
    
    def Reparameterize(self, mean, logvar, size):
        size = [3,size[1],size[2],size[3]]
        pdf = tf.distributions.Normal(loc=mean,scale=logvar)
        flow = pdf.sample(size)
        flow = tf.squeeze(flow, axis=5)
        flow = tf.transpose(flow)
        return flow

    def Distribution(self, feature):
        meanF =  tf.layers.conv3d(feature, 3, 3, 1, padding="same", kernel_initializer= tf.truncated_normal_initializer(mean=0.0, stddev=1e-5))
        meanB = tf.layers.conv3d(feature, 3, 3, 1, padding="same",  kernel_initializer= tf.truncated_normal_initializer(mean=0.0, stddev=1e-5))
        logvarF = tf.layers.conv3d(feature, 3, 3, 1, padding="same", kernel_initializer= tf.truncated_normal_initializer(mean=0.0, stddev=1e-10))
        logvarB = tf.layers.conv3d(feature, 3, 3, 1, padding="same", kernel_initializer= tf.truncated_normal_initializer(mean=0.0, stddev=1e-10))
        flowF = self.sample(meanF,logvarF) #self.Reparameterize(meanF, logvarF, size)
        flowB = self.sample(meanB,logvarB) #self.Reparameterize(meanB, logvarB, size)
        return flowF, flowB, meanF, logvarF, meanB, logvarB

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
            src_cyc = nrn.SpatialTransformer(interp_method='linear', indexing='ij')([f_warp, i_flow])
            i_warp = nrn.SpatialTransformer(interp_method='linear', indexing='ij')([self.tgt, i_flow])
            tgt_cyc = nrn.SpatialTransformer(interp_method='linear', indexing='ij')([i_warp, f_flow])

            f_warp = tf.identity(f_warp,"Forward_im")
            i_warp = tf.identity(i_warp, "Inverse_im")

            f_flow = tf.identity(f_flow, "Forward_flow")
            i_flow = tf.identity(i_flow, "Inverse_flow")
            Flow = tf.concat([f_flow, i_flow], 4)

        return f_warp, i_warp, src_cyc, tgt_cyc, Flow













