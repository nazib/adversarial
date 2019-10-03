import tensorflow as tf
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, load_model
from keras.layers import Conv3D, Activation, Input, UpSampling3D, concatenate,MaxPooling3D, Multiply
from keras.layers.merge import Concatenate
from keras.layers import LeakyReLU, Reshape, Lambda, BatchNormalization, Dense, Flatten, ReLU,Dropout
from keras.optimizers import Adam,RMSprop, SGD
from keras.initializers import RandomNormal
import keras
import numpy as np
from keras import backend as K
# local
from dense_3D_spatial_transformer import Dense3DSpatialTransformer
from InverseNet import *


def encoder(tgt, enc_nf,isSkip=True):
    #x_in = concatenate([src, tgt])
    x0 = myConv(tgt, enc_nf[0], 2)
    x1 = myConv(x0, enc_nf[1], 2)

    x2 = myConv(x1, enc_nf[2], 2)  # 20x24x28
    # x2 = BatchNormalization()(x2)
    x3 = myConv(x2, enc_nf[3], 2)  # 10x12x14
    x3 = BatchNormalization()(x3)

    if isSkip:
        skip = [x2,x1,x0]
        return x3, skip
    else:
        return x3

def decoder(features,skip,dec_nf):

    x = myConv(features, dec_nf[0])
    x = UpSampling3D()(x)
    # x= BatchNormalization()(x)
    x = concatenate([x, skip[0]])
    x = myConv(x, dec_nf[1])
    x = UpSampling3D()(x)
    x = concatenate([x, skip[1]])
    # x= BatchNormalization()(x)
    x = myConv(x, dec_nf[2])
    x = UpSampling3D()(x)
    x = concatenate([x, skip[2]])
    # x= BatchNormalization()(x)
    x = myConv(x, dec_nf[3])
    x = myConv(x, dec_nf[4])

    x = UpSampling3D()(x)
    x = concatenate([x, skip[3]])
    # x= BatchNormalization()(x)
    out_img = myConv(x, dec_nf[-1], 1)
    # flow = Conv3D(3, kernel_size=3, padding='same',
    #               kernel_initializer='he_normal', strides=1)(out_img)

    flow = Conv3D(3, kernel_size=3, padding='same',
                  kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='flow')(out_img)

    warped = Dense3DSpatialTransformer()([skip[3], flow])

    return out_img, flow, warped


def myConv(x_in, nf, strides=1):
    x_out = Conv3D(nf, kernel_size=3, padding='same',
                   kernel_initializer='he_normal', strides=strides)(x_in)
    x_out = LeakyReLU(0.2)(x_out)
    return x_out

def Generator(vol_size, enc_nf,dec_nf):
    src = Input(shape=vol_size + (1,))
    tgt = Input(shape=vol_size + (1,))
    #diff = Input(shape=vol_size + (1,))

    enc_out,skip = encoder(src, tgt, enc_nf,True)
    out_img, flow, warped = decoder(enc_out,skip,dec_nf)
    generator_Model = Model(inputs=[src, tgt], outputs=[warped, flow])

    return generator_Model

def Discriminator(vol_size,enc_nf):

    tgt = Input(shape=vol_size+(1,))
    #src = Input(shape=vol_size + (1,))

    #X = Input(shape=vol_size+(2,))

    enc_feat = encoder(tgt,enc_nf,False)

    out = Flatten()(enc_feat)
    out = Dense(512)(out)
    out = Dense(256)(out)
    out = Dense(1)(out)
    out = Activation('sigmoid')(out)

    discriminator_model = Model(inputs=[tgt],outputs=[out])
    return discriminator_model

class AdversarialNet:

    def __init__(self, generator, discriminator,generator_optimizer=Adam(0.0001, beta_1=.5, beta_2=0.9),
                 discriminator_optimizer=Adam(0.0001, beta_1=.5, beta_2=0.9),
                 batch_size=64, custom_objects={}, **kwargs):
        if type(generator) == str:
            self._generator = load_model(generator, custom_objects=custom_objects)
        else:
            self._generator = generator
        if type(discriminator) == str:
            self._discriminator = load_model(discriminator, custom_objects=custom_objects)
        else:
            self._discriminator = discriminator

        self._generator_optimizer = generator_optimizer
        self._discriminator_optimizer = discriminator_optimizer

        generator_input = self._generator.input

        discriminator_input = self._discriminator.input

        if type(generator_input) == list:
            self._generator_input = generator_input
        else:
            self._generator_input = [generator_input]

        if type(discriminator_input) == list:
            self._discriminator_input = discriminator_input
        else:
            self._discriminator_input = [discriminator_input]

        self._batch_size = batch_size

        self.generator_metric_names = []
        self.discriminator_metric_names = ['true', 'fake']

    def _set_trainable(self, net,trainable):
        for layer in net.layers:
            layer.trainable = trainable
        net.trainable = trainable

    def _compile_generator(self):
        """
            Create Generator model that from noise produce images. It`s trained usign discriminator
        """
        self._set_trainable(self._generator, True)
        self._set_trainable(self._discriminator, False)

        discriminator_output_fake = self._discriminator(self._discriminator_fake_input)

        generator_model = Model(inputs=self._generator_input, outputs=discriminator_output_fake)
        loss, metrics = self._compile_generator_loss()
        generator_model.compile(optimizer=self._generator_optimizer, loss=loss, metrics=metrics)

        return generator_model

    def _compile_generator_loss(self):
        """
            Create generator loss and metrics
        """
        def generator_crossentrohy_loss(y_true, y_pred):
            return -K.mean(K.log(y_pred + 1e-7))

        return generator_crossentrohy_loss, []

    def _compile_discriminator(self):
        """
            Create model that produce discriminator scores from real_data and noise(that will be inputed to generator)
        """
        self._set_trainable(self._generator, False)
        self._set_trainable(self._discriminator, True)

        disc_in = [Concatenate(axis=0)([true, fake])
                   for true, fake in zip(self._discriminator_input, self._discriminator_fake_input)]

        discriminator_model = Model(inputs=self._discriminator_input + self._generator_input,
                                    outputs=self._discriminator(disc_in))
        loss, metrics = self._compile_discriminator_loss()
        discriminator_model.compile(optimizer=self._discriminator_optimizer, loss=loss, metrics=metrics)

        return discriminator_model

    def _compile_discriminator_loss(self):
        """
            Create generator loss and metrics
        """

        def fake_loss(y_true, y_pred):
            return -K.mean(K.log(1 - y_pred[self._batch_size:] + 1e-7))

        def true_loss(y_true, y_pred):
            return -K.mean(K.log(y_pred[:self._batch_size] + 1e-7))

        def discriminator_crossentrohy_loss(y_true, y_pred):
            return fake_loss(y_true, y_pred) + true_loss(y_true, y_pred)

        return discriminator_crossentrohy_loss, [true_loss, fake_loss]

    def compile_models(self):
        #self._discriminator_fake_input = self._generator(self._generator_input)[0]
        self._discriminator_fake_input = [self._generator(self._generator_input)[0], self._generator.inputs[1]]
        if type(self._discriminator_fake_input) != list:
            self._discriminator_fake_input = [self._discriminator_fake_input]
        return self._compile_generator(), self._compile_discriminator()

    def get_generator(self):
        return self._generator

    def get_discriminator(self):
        return self._discriminator

    def get_losses_as_string(self, generator_losses, discriminator_losses):
        def combine(name_list, losses):
            losses = np.array(losses)
            if len(losses.shape) == 0:
                losses = losses.reshape((1,))
            return '; '.join([name + ' = ' + str(loss) for name, loss in zip(name_list, losses)])

        generator_loss_str = combine(['Generator loss'] + self.generator_metric_names, generator_losses)
        discriminator_loss_str = combine(['Disciminator loss'] + self.discriminator_metric_names, discriminator_losses)
        return generator_loss_str, discriminator_loss_str





















