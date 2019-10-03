# python imports
import os
import glob
import sys
import random

import nibabel as nib
# third-party imports
import tensorflow as tf
import numpy as np
import scipy.io as sio
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam, RMSprop, SGD
from keras.models import load_model, Model
from keras.utils import multi_gpu_model
from keras.losses import mse, binary_crossentropy
# import datagenerators
import refnet
import losses
import DenseDeform
import InverseNet
# from datagenerators import example_gen
from measurment import *
from utility import *
import cv2
import nibabel as nib
import h5py
import pdb
import BatchDataReader
from itertools import permutations
from ConfigReader import *


def train(config_file):
    config = Configuration()
    config.CreateConfig(config_file)
    config.PrintConfiguration()

    vol_size = (config.patch_size, config.patch_size, config.patch_size)
    batch_size = 4

    train_patch_pairs = glob.glob(config.base_directory + "/train/*.h5")
    random.shuffle(train_patch_pairs)

    model_dir = config.base_directory + config.Model_name
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    gpu = '/gpu:' + str(config.GPU)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.GPU)
    configurnet = tf.ConfigProto()
    configurnet.gpu_options.allow_growth = True
    configurnet.gpu_options.per_process_gpu_memory_fraction = 0.9
    configurnet.allow_soft_placement = True
    set_session(tf.Session(config=configurnet))

    with tf.device(gpu):
        if config.ModelType == 'VM':
            nf_enc = [16, 32, 32, 32, 16]
            nf_dec = [32, 32, 32, 32, 8]
            model = refnet.ref_net((64, 64, 64), nf_enc, nf_dec)
        if config.ModelType == 'DDN':
            model = DenseDeform.DenseDeformNet(vol_size, config.HalfNet, config.BN, config.Dilation)
            model = model.createModel()
        if config.ModelType == 'Inverse':
            model = InverseNet.InverseNet(vol_size, config.HalfNet, config.BN, config.Dilation)
            model = model.createModel()
        '''
        else:
            model = refnet.ref_net(vol_size,nf_enc,nf_dec)
        '''
        # sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)

        model.compile(optimizer=Adam(lr=config.Learning_Rate), loss=[losses.cc3D(), losses.gradientLoss('l2')],
                      loss_weights=[1.0, 1.5])
        # model.metrics_tensors = [model.layers[2].output[0,:,:,:,0],model.layers[2].output[0,:,:,:,1],model.layers[166].output[0,:,:,:,0],model.layers[170].output[0,:,:,:,:],
        # model.layers[168].output[0,:,:,:,0],model.layers[167].output[0,:,:,:,0]]
        # model.metrics_tensors = [model.layers[2].output[0,:,:,:,0],model.layers[2].output[0,:,:,:,1],model.layers[83].output[0,:,:,:,0],model.layers[82].output[0,:,:,:,:]]

        if np.int(config.iteration_start) > 0:
            model.load_weights(model_dir + '/' + str(config.iteration_start) + '.h5')
    '''
    input_patch_1 = np.zeros((batch_size,vol_size[0], vol_size[1], vol_size[2], 1),dtype=np.float)
    input_patch_2 = np.zeros((batch_size,vol_size[0], vol_size[1], vol_size[2], 1),dtype=np.float)
    output_patch = np.zeros((batch_size,vol_size[0], vol_size[1], vol_size[2], 1),dtype=np.float)
    deformation = np.zeros((batch_size,vol_size[0], vol_size[1], vol_size[2], 3),dtype=np.float)
    '''
    inpx_fake = np.zeros((batch_size, vol_size[0], vol_size[1], vol_size[2], 1), dtype=np.float)
    inpy_fake = np.zeros((batch_size, vol_size[0], vol_size[1], vol_size[2], 1), dtype=np.float)
    f_flow = np.zeros((batch_size, vol_size[0], vol_size[1], vol_size[2], 3), dtype=np.float)
    i_flow = np.zeros((batch_size, vol_size[0], vol_size[1], vol_size[2], 3), dtype=np.float)
    start = config.iteration_start

    iteration = 1734 / batch_size
    # n_iteration=iteration*len(train_patch_pairs)

    ep_st = 0
    step_st = 0
    paris_st = 0

    if start > 0:
        if start > iteration:
            step_st = start - np.int((start / iteration)) * iteration
            paris_st = np.int(start / np.float(iteration))
        else:
            step_st = start

    # total_pairs = len(train_patch_pairs)*(len(train_patch_pairs)-1)
    x = np.arange(len(train_patch_pairs))

    total_pairs = list(permutations(x, 2))
    random.shuffle(total_pairs)

    for pairs in list(total_pairs):

        src_im = h5py.File(train_patch_pairs[pairs[0]],"r")
        src_im = src_im.get("moving").value
        random.shuffle(src_im)
        tgt_im = h5py.File(train_patch_pairs[pairs[1]],"r")
        tgt_im = tgt_im.get("moving").value
        random.shuffle(tgt_im)

        print ("{0} Patches are selected".format(len(src_im)))

        s = 0
        for step in range(step_st, iteration):

            src = src_im[s:s + batch_size, :, :, :]
            src = np.reshape(src, (batch_size, vol_size[0], vol_size[1], vol_size[2], 1))
            tgt = tgt_im[s:s + batch_size, :, :, :]
            tgt = np.reshape(tgt, (batch_size, vol_size[0], vol_size[1], vol_size[2], 1))

            train_loss = model.train_on_batch([src, tgt], [tgt, f_flow])

            print(" Paris : " + str(pairs) + " Step :" + str(step) + " Patch :" + str(s) + " Loss: " + str(
                train_loss[0]))

            if not isinstance(train_loss, list):
                train_loss = [train_loss]
            if (start % config.Model_saver == 0):
                model.save(model_dir + '/' + str(start) + '.h5')
            if (start % 100 == 0):
                sess = tf.keras.backend.get_session()
                write_summary(start, sess, train_loss, model_dir)

            s = s + batch_size
            start = start + 1




if __name__ == "__main__":
    if (len(sys.argv) != 2):
        print "Configuration file name is required"
    else:
        train(sys.argv[1])


