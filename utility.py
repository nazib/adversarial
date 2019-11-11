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

def normalize_flow(flow,flow_m):
    
    IXp = np.where(flow > 0)
    IXn = np.where(flow < 0)

    if (len(IXp[0]) > 0):
        flow[IXp] = ((np.max(flow_m) - 0) / (np.max(flow[IXp]) - np.min(flow[IXp])) * (
                    flow[IXp] - np.min(flow[IXp])) + 0)

    if (len(IXn[0]) > 0):
        flow[IXn] = ((0 - np.min(flow[IXn])) / (0 - np.min(flow[IXn])) * (flow[IXn] - np.min(flow[IXn])) + np.min(
            flow[IXn]))
    return flow
    

def Generate_deformation(img, sigma):

    #img = nib.load(img).get_data()
    #img = set_mid_img(img,cSize=(512,540,169),dSize=(576,576,192))
    m,n,c = img.shape
    if m!=n:
       img = set_mid_img(img, cSize=(m-128,n,c), dSize=(m,m,c))
     
    flow = np.zeros((img.shape[0],img.shape[1],img.shape[2],6))
    #patch = np.reshape(patch, im_shape)
    Points = 150
    maxdeform = 0.5
    #mu = np.mean(patch)
    #pmax = np.max(patch)
    #above_zero = np.where(img <= (pmax-mu))
    above_zero = np.where(img <= 0.3)

    RDF = np.zeros([img.shape[0], img.shape[1], img.shape[2], 6], dtype=np.float64)
    RDFx = np.zeros(img.shape, dtype=np.float64)
    RDFy = np.zeros(img.shape, dtype=np.float64)
    RDFz = np.zeros(img.shape, dtype=np.float64)
    RDFxf = np.zeros(img.shape, dtype=np.float64)
    RDFyf = np.zeros(img.shape, dtype=np.float64)
    RDFzf = np.zeros(img.shape, dtype=np.float64)

    iRDFx = np.zeros(img.shape, dtype=np.float64)
    iRDFy = np.zeros(img.shape, dtype=np.float64)
    iRDFz = np.zeros(img.shape, dtype=np.float64)
    iRDFxf = np.zeros(img.shape, dtype=np.float64)
    iRDFyf = np.zeros(img.shape, dtype=np.float64)
    iRDFzf = np.zeros(img.shape, dtype=np.float64)

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
    
    RDFxf = normalize_flow(RDFxf, RDFx)
    RDFyf = normalize_flow(RDFyf, RDFy)
    RDFzf = normalize_flow(RDFzf, RDFz)
    
    #### Inverse Flow #####
    iRDFxf = -RDFxf#gaussian_filter(RDFx, sigma=-sigma)
    iRDFyf = -RDFyf#gaussian_filter(RDFy, sigma=-sigma)
    iRDFzf = -RDFzf#gaussian_filter(RDFz, sigma=-sigma)
    ####################################### Normalization #############################################
   
    RDF[:, :, :, 0] = RDFxf
    RDF[:, :, :, 1] = RDFyf
    RDF[:, :, :, 2] = RDFzf
    RDF[:, :, :, 3] = iRDFxf
    RDF[:, :, :, 4] = iRDFyf
    RDF[:, :, :, 5] = iRDFzf

    flow = RDF#ApplyDeform(patch,RDF,im_shape)

    return flow









