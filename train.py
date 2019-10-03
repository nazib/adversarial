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
from keras import backend as K
from keras.optimizers import Adam,RMSprop, SGD
from keras.models import load_model, Model
from keras.utils import multi_gpu_model
from keras.losses import mse, binary_crossentropy
import losses
import InverseNet
#from datagenerators import example_gen
from measurment import *
from utility import *
import h5py
import  BatchDataReader
from itertools import permutations
from ConfigReader import *
from AdversarialNet import *


def generator_loss(fake_output):
    return binary_crossentropy(tf.ones_like(fake_output), tf.convert_to_tensor(fake_output))


def discriminator_loss(real_output, fake_output):
    real_loss = binary_crossentropy(tf.ones_like(real_output),  tf.convert_to_tensor(real_output))
    fake_loss = binary_crossentropy(tf.zeros_like(fake_output),  tf.convert_to_tensor(fake_output))
    total_loss = real_loss + fake_loss
    return total_loss


def train(config_file):

    config = Configuration()
    config.CreateConfig(config_file)
    config.PrintConfiguration()

    vol_size=(config.patch_size[0],config.patch_size[1],config.patch_size[2])
    batch_size =4
     
    train_patch_pairs = glob.glob(config.base_directory+"/train/*.nii.gz")
    random.shuffle(train_patch_pairs)

    train_dataset_reader = BatchDataReader.BatchDataset(train_patch_pairs)

    
    #f=h5py.File(base_data_dir+"atlas/atlas_patches.h5","r")
    #atlas_patches=f['atlas']
    '''
    atlas_image = base_data_dir+'atlas/003_nuclear.nii.gz'
    train_patch_reader = BatchDataset(train_patch_pairs,atlas_image)
    '''

    model_dir = config.base_directory + config.Model_name
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    gpu = '/gpu:' + str(config.GPU)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.GPU)
    configurnet = tf.ConfigProto()
    configurnet.gpu_options.allow_growth = True
    configurnet.gpu_options.per_process_gpu_memory_fraction=0.9
    configurnet.allow_soft_placement = True
    set_session(tf.Session(config=configurnet))

    with tf.device(gpu):
        nf_enc = [16,32,32,32,16]
        nf_dec = [32,32,32,32,8]

        #gmodel = Generator(vol_size,enc_nf=nf_enc,dec_nf=nf_dec)
        gmodel = InverseNet(vol_size,'off')
        gmodel = gmodel.createModel()
        dmodel_T = Discriminator(vol_size,enc_nf=nf_enc)
        dmodel_S = Discriminator(vol_size,enc_nf=nf_enc)

        #gmodel.compile(loss=[losses.cc3D(),losses.cc3D()],loss_weights=[1.0,-1.0],optimizer=Adam(0.0001, beta_1=.5, beta_2=0.9))
        #gmodel.compile(loss=losses.l13D(),optimizer=Adam(0.0001, beta_1=.5, beta_2=0.9))

        dmodel_T.compile(loss=losses.l13D(),optimizer=Adam(0.005, beta_1=.5, beta_2=0.9))
        dmodel_T.trainable = True
        dmodel_S.compile(loss=losses.l13D(), optimizer=Adam(0.005, beta_1=.5, beta_2=0.9))
        dmodel_S.trainable = True

        ## Building GAN ##
        #z = [Input(vol_size+(1,)) for i in range(2)]

        src_input = Input(vol_size+(1,))
        tgt_input = Input(vol_size+(1,))
        z = [src_input,tgt_input]

        gen_out = gmodel(z)

        dmodel_T_output = dmodel_T([gen_out[1]])

        dmodel_S_output = dmodel_T([gen_out[0]])

        GAN = Model(inputs=z, outputs=[dmodel_T_output, dmodel_S_output])

        GAN.compile(loss=[losses.cycle_loss(),losses.l13D()],loss_weights=[0.5,0.5],optimizer=Adam(0.0001, beta_1=.5, beta_2=0.9))
        #dmodel.trainable = False


    inpx_fake =  np.zeros((batch_size, vol_size[0], vol_size[1], vol_size[2], 1),dtype=np.float)
    inpy_fake =  np.zeros((batch_size,vol_size[0], vol_size[1], vol_size[2], 1),dtype=np.float)
    f_flow = np.zeros((batch_size,vol_size[0], vol_size[1], vol_size[2], 3),dtype=np.float)
    i_flow = np.zeros((batch_size,vol_size[0], vol_size[1], vol_size[2], 3),dtype=np.float)
    D_output_fake = np.zeros((batch_size,1),dtype=np.float)
    D_output_real = np.ones((batch_size, 1), dtype=np.float)
    start=config.iteration_start
    
    
    iteration=2500/batch_size
    #n_iteration=iteration*len(train_patch_pairs)
    
    ep_st=0
    step_st=0
    paris_st=0

    loc_file = open('location.txt','w')
    loc_file.close()
    sim_file = open('pair_similarity.txt', 'w')
    sim_file.close()

    if start>0:
        if start > iteration:
            step_st = start-np.int((start/iteration))*iteration
            paris_st = np.int(start/np.float(iteration))
        else:
            step_st=start


    #total_pairs = len(train_patch_pairs)*(len(train_patch_pairs)-1)
    x = np.arange(len(train_patch_pairs))
    
    total_pairs = list(permutations(x,2))
    random.shuffle(total_pairs)
    dmodel_T.trainable = True
    dmodel_S.trainable = True
    d_inp1 = np.zeros((batch_size,vol_size[0],vol_size[1],vol_size[2],1))
    d_inp2 = np.zeros((batch_size, vol_size[0], vol_size[1], vol_size[2], 1))

    for pairs in list(total_pairs):

        src_im,tgt_im,locations,pair_similarity = train_dataset_reader.create_pairs(pairs[0],pairs[1],config.patch_size)
        print ("2500 Patches are selected")

        if sim_file.close:
            sim_file =  open('pair_similarity.txt', 'a')
            sim_file.writelines("Pair:{0} and {1} is CC:{2} \n".format(train_patch_pairs[pairs[0]],train_patch_pairs[pairs[1]],pair_similarity))
            sim_file.close()

        s=0
        for step in range(step_st,iteration):

            src = src_im[s:s + batch_size, :, :, :]
            src = np.reshape(src, (batch_size, vol_size[0], vol_size[1], vol_size[2], 1))
            tgt = tgt_im[s:s + batch_size, :, :, :]
            tgt = np.reshape(tgt, (batch_size, vol_size[0], vol_size[1], vol_size[2], 1))


            gen_out = gmodel.predict([src, tgt])

            #d_inp1[0:2,:,:,:,:] = tgt[0:2,:,:,:,:]
            #d_inp2[2:4, :, :, :, :] = gen_out[0][2:4, :, :, :, :]

            d_T_fake = dmodel_T.train_on_batch([gen_out[1]],[D_output_fake])
            d_T_real = dmodel_T.train_on_batch([tgt], [D_output_real])

            d_T = d_T_fake + d_T_real
            #d_inp1[0:2, :, :, :, :] = src[0:2, :, :, :, :]
            #d_inp1[2:4, :, :, :, :] = gen_out[1][2:4, :, :, :, :]

            d_S_fake = dmodel_S.train_on_batch([gen_out[0]], [D_output_fake])
            d_S_real = dmodel_S.train_on_batch([src], [D_output_real])

            d_S = d_S_fake + d_S_real

            #d_fake = dmodel.predict()

            g_loss = GAN.train_on_batch([src,tgt], [D_output_real,D_output_real])

            train_loss = [g_loss[0],d_T,d_S]


            print(" Paris : "+str(pairs)+" Step :"+str(step)+" Patch :"+str(s)+" Loss: "+str(train_loss))

            if not isinstance(train_loss, list):
                train_loss = [train_loss]
            if(start % config.Model_saver == 0):
                gmodel.save(model_dir + '/' + str(start) + '.h5')
            if(start % 100 == 0):
                sess=tf.keras.backend.get_session()
                write_summary(start,sess,train_loss,model_dir)

            s=s+batch_size
            start=start+1

    loc_file.close()

if __name__ == "__main__":
    if(len(sys.argv) != 2):
        print "Configuration file name is required"
    else:
        train(sys.argv[1])
    
    
