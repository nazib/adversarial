import sys
import glob
import os
import numpy as np
import tensorflow as tf
import scipy.io as sio
import math
from scipy.ndimage.filters import gaussian_filter
from keras.backend.tensorflow_backend import set_session
import pdb
from scipy.interpolate import interpn
import BatchDataReader
import nibabel as nib
from partition import set_mid_img,crop_mid_img

loss1=tf.placeholder(shape=(), dtype=tf.double)
tf.summary.scalar("Total_loss",loss1)

loss2=tf.placeholder(shape=(), dtype=tf.double)
tf.summary.scalar("CC_loss",loss2)

loss3=tf.placeholder(shape=(), dtype=tf.double)
tf.summary.scalar("Cyc_loss",loss3)

loss4=tf.placeholder(shape=(), dtype=tf.double)
tf.summary.scalar("D tgt",loss4)

cc = tf.placeholder(shape=(), dtype=tf.double)
tf.summary.scalar("D src", cc)
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

def Generate_deformation(img, im_shape, sigma):

    img = nib.load(img).get_data()
    img = set_mid_img(img,cSize=(512,540,169),dSize=(576,576,192))
    flow = np.zeros((im_shape[0],im_shape[1],im_shape[2],6))
    #patch = np.reshape(patch, im_shape)
    Points = 150
    maxdeform = 0.5
    #mu = np.mean(patch)
    #pmax = np.max(patch)
    #above_zero = np.where(img <= (pmax-mu))
    above_zero = np.where(img <= 0.3)

    RDF = np.zeros([im_shape[0], im_shape[1], im_shape[2], 6], dtype=np.float64)
    RDFx = np.zeros(im_shape, dtype=np.float64)
    RDFy = np.zeros(im_shape, dtype=np.float64)
    RDFz = np.zeros(im_shape, dtype=np.float64)
    RDFxf = np.zeros(im_shape, dtype=np.float64)
    RDFyf = np.zeros(im_shape, dtype=np.float64)
    RDFzf = np.zeros(im_shape, dtype=np.float64)

    iRDFx = np.zeros(im_shape, dtype=np.float64)
    iRDFy = np.zeros(im_shape, dtype=np.float64)
    iRDFz = np.zeros(im_shape, dtype=np.float64)
    iRDFxf = np.zeros(im_shape, dtype=np.float64)
    iRDFyf = np.zeros(im_shape, dtype=np.float64)
    iRDFzf = np.zeros(im_shape, dtype=np.float64)

    k=0

    while (k < Points):
        voxel_idx = np.long(np.random.randint(0, len(above_zero[0]) - 1, 1, dtype=np.int64))
        x = above_zero[0][voxel_idx]
        y = above_zero[1][voxel_idx]
        z = above_zero[2][voxel_idx]

        Dx = ((np.random.ranf([1]))[0] - 0.5) * maxdeform*x
        Dy = ((np.random.ranf([1]))[0] - 0.5) * maxdeform*y
        Dz = ((np.random.ranf([1]))[0] - 0.5) * maxdeform*z

        RDFx[x, y, z] = Dx
        RDFy[x, y, z] = Dy
        RDFz[x, y, z] = Dz

        Dx = ((np.random.ranf([1]))[0] + 0.5) * maxdeform*x
        Dy = ((np.random.ranf([1]))[0] + 0.5) * maxdeform*y
        Dz = ((np.random.ranf([1]))[0] + 0.5) * maxdeform*z

        iRDFx[x, y, z] = Dx
        iRDFy[x, y, z] = Dy
        iRDFz[x, y, z] = Dz

        # print "Point:"+str(k)
        k += 1

        # del BorderMask

    RDFxf = gaussian_filter(RDFx, sigma=sigma)
    RDFyf = gaussian_filter(RDFy, sigma=sigma)
    RDFzf = gaussian_filter(RDFz, sigma=sigma)

    iRDFxf = gaussian_filter(RDFx, sigma=-sigma)
    iRDFyf = gaussian_filter(RDFy, sigma=-sigma)
    iRDFzf = gaussian_filter(RDFz, sigma=-sigma)
    ####################################### Normalization #############################################
    IXp = np.where(RDFxf > 0)
    IXn = np.where(RDFxf < 0)
    IYp = np.where(RDFyf > 0)
    IYn = np.where(RDFyf < 0)
    IZp = np.where(RDFzf > 0)
    IZn = np.where(RDFzf < 0)

    #### Normalizing x-direction ###
    if (len(IXp[0]) > 0):
        RDFxf[IXp] = ((np.max(RDFx) - 0) / (np.max(RDFxf[IXp]) - np.min(RDFxf[IXp])) * (
                    RDFxf[IXp] - np.min(RDFxf[IXp])) + 0)

    if (len(IXn[0]) > 0):
        RDFxf[IXn] = ((0 - np.min(RDFxf[IXn])) / (0 - np.min(RDFxf[IXn])) * (RDFxf[IXn] - np.min(RDFxf[IXn])) + np.min(
            RDFxf[IXn]))

    #### Normalizing y-direction ####
    if (len(IYp[0]) > 0):
        RDFyf[IYp] = ((np.max(RDFy) - 0) / (np.max(RDFyf[IYp]) - np.min(RDFyf[IYp])) * (
                    RDFyf[IYp] - np.min(RDFyf[IYp])) + 0)
    if (len(IYn[0]) > 0):
        RDFyf[IYn] = ((0 - np.min(RDFyf[IYn])) / (0 - np.min(RDFyf[IYn])) * (RDFyf[IYn] - np.min(RDFyf[IYn])) + np.min(
            RDFyf[IYn]))

    #######Normalizing z-direction ####

    if (len(IZp[0]) > 0):
        RDFzf[IZp] = ((np.max(RDFz) - 0) / (np.max(RDFzf[IZp]) - np.min(RDFzf[IZp])) * (
                    RDFzf[IZp] - np.min(RDFzf[IZp])) + 0)
    if (len(IZn[0]) > 0):
        RDFzf[IZn] = ((0 - np.min(RDFzf[IZn])) / (0 - np.min(RDFzf[IZn])) * (RDFzf[IZn] - np.min(RDFzf[IZn])) + np.min(
            RDFzf[IZn]))

    RDF[:, :, :, 0] = RDFxf
    RDF[:, :, :, 1] = RDFyf
    RDF[:, :, :, 2] = RDFzf
    RDF[:, :, :, 3] = iRDFxf
    RDF[:, :, :, 4] = iRDFyf
    RDF[:, :, :, 5] = iRDFzf

    flow = RDF#ApplyDeform(patch,RDF,im_shape)

    return flow









