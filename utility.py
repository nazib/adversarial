import sys
import glob
import os
import numpy as np
import tensorflow as tf
import scipy.io as sio
import math
from keras.backend.tensorflow_backend import set_session
import pdb
from scipy.interpolate import interpn
import BatchDataReader

loss1=tf.placeholder(shape=(), dtype=tf.double)
tf.summary.scalar("G_loss",loss1)

loss2=tf.placeholder(shape=(), dtype=tf.double)
tf.summary.scalar("D_loss",loss2)

loss3=tf.placeholder(shape=(), dtype=tf.double)
tf.summary.scalar("D_source_loss",loss3)

loss4=tf.placeholder(shape=(), dtype=tf.double)
tf.summary.scalar("D_target_loss",loss4)

cc = tf.placeholder(shape=(), dtype=tf.double)
tf.summary.scalar("CC", cc)
mi = tf.placeholder(shape=(), dtype=tf.double)
tf.summary.scalar("MI", mi)
merged = tf.summary.merge_all()



def write_summary(i, sess, data, model_dir):

    tf.global_variables_initializer()
    writer = tf.summary.FileWriter(model_dir)
    x1=np.double(data[0])
    x2=np.double(data[1])
    x3=np.double(data[2])
    x4 = np.double(data[3])
    x5 = np.double(data[4])
    x6 = np.double(data[5])
    summary = sess.run(merged, feed_dict={loss1: x1, loss2: x2, loss3: x3, loss4: x4, cc: x5, mi: x6})
    writer.add_summary(summary,i)


def gpu_configure(gpu_id_1,gpu_id_2):
    gpus =  str(gpu_id_1)+","+str(gpu_id_2)
    gpu1 = '/gpu:' + str(gpu_id_1)
    gpu2 = '/gpu:' + str(gpu_id_2)

    os.environ["CUDA_VISIBLE_DEVICES"] =str(gpus)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction=0.9
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))

    return gpus,gpu1,gpu2, config 


def ApplyDeform(input_img,flow,vol_size):
    
    xx = np.arange(vol_size[1])
    yy = np.arange(vol_size[0])
    zz = np.arange(vol_size[2])
    grid = np.rollaxis(np.array(np.meshgrid(xx, yy, zz)), 0, 4)
    sample = flow+grid
    sample = np.stack((sample[:, :, :, 1], sample[:, :, :, 0], sample[:, :, :, 2]), 3)
    warp_cube = interpn((yy, xx, zz),input_img, sample, method='nearest', bounds_error=False, fill_value=0)

    return warp_cube






