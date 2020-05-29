# py imports
import os
import sys
import glob
# third party
import tensorflow as tf
import scipy.io as sio
import numpy as np
from keras.backend.tensorflow_backend import set_session
from scipy.interpolate import interpn
import time
# from DenseDeformOld import DenseDeformNet
from InverseNet_tf import *
from InverseNet_tfv2 import *
import nibabel as nib
import h5py
from measurment import *
from partition import *
# from utility import patches2Img
import SimpleITK as sitk
from ConfigReader import *
from colormap import *


def load_validation_data(validation_atlases,config):
    
    atlas_volumes = np.zeros((3,config.TestImageSize[0],config.TestImageSize[1],config.TestImageSize[2]))
    for i in range(len(validation_atlases)):
        F = h5py.File(validation_atlases[i],'r')
        patches = F['moving']
        if config.TestResolution =='25':
            patchEx = PatchUtility((64,64,64),config.TestImageSize,config.TestOverlap,patches=patches,image=[])
            atlas_volumes[i,:,:,:] =  patchEx.combine_patches()
        else:
            patchEx = PatchUtility((64, 64, 32), config.TestImageSize, config.TestOverlap, patches=patches, image=[])
            atlas_volumes[i, :, :, :] = patchEx.combine_patches()
            #atlas_volumes[i, :, :, :] = patches2Img(patches, config.TestImageSize, config.TestPatchSize[0] + 1,32)
    return atlas_volumes
        

def test(configFile):
    config = Configuration()
    config.ReadTestConfig(configFile)

    # Loading Test data
    test_file_list = glob.glob(config.TestDataDir + "*.h5")
    # for valoidation test
    #test_file_list =[config.TestDataDir+"moving_def_0.h5",config.TestDataDir+"moving_def_1.h5",config.TestDataDir+"moving_def_2.h5"]
    #validation_atlases = [config.TestDataDir+"moving_0.h5",config.TestDataDir+"moving_1.h5",config.TestDataDir+"moving_2.h5"]

    confignet = tf.ConfigProto()
    confignet.gpu_options.allow_growth = True
    confignet.allow_soft_placement = True
    #set_session(tf.Session(config=confignet))
    start_time = time.clock()

    # load weights of model
    with tf.device(config.TestGPU):
        sess = tf.Session(config=confignet)
        sess.run(tf.global_variables_initializer())
        model_dir = os.path.join(config.TestModelDir,"models")
        model_name = model_dir.split('/')[-2]
        saver = tf.train.import_meta_graph(model_dir +"/"+ model_name + "-0.meta")
        #saver = tf.train.import_meta_graph(model_dir + "/" +"-0.meta")
        saver.restore(sess, model_dir + "/" + model_name + "-"+config.TestModelNumber)
        #saver.restore(sess, model_dir + "/" + "-" + config.TestModelNumber)
        #print (tf.contrib.framework.list_variables(model_dir +"/"+ model_name + "-0.meta"))
        src_place = sess.graph.get_tensor_by_name("source:0")
        tgt_place = sess.graph.get_tensor_by_name("target:0")
        f_im = sess.graph.get_tensor_by_name('Generator/Forward_im:0')
        i_im = sess.graph.get_tensor_by_name('Generator/Inverse_im:0')
        Flow = sess.graph.get_tensor_by_name('Generator/concat_9:0')
        #dis_out = sess.graph.get_operation_by_name('Discriminator_target/sigmoid:0')


    n_batches = len(test_file_list)
    xx = np.arange(config.TestPatchSize[0])
    yy = np.arange(config.TestPatchSize[1])
    zz = np.arange(config.TestPatchSize[2])
    grid = np.rollaxis(np.array(np.meshgrid(xx, yy, zz)), 0, 4)
    flow = np.zeros((1, config.TestPatchSize[0], config.TestPatchSize[1], config.TestPatchSize[2], 3), dtype=np.float)
    warp_patches = []

    for k in range(len(test_file_list)-1):
        # vol_name, seg_name = test_brain_strings[k].split(",")
        start_time = time.clock()
        F = h5py.File(test_file_list[k], "r")
        X = F['moving']

        #Fc =h5py.File(validation_atlases[k],"r")
        #Xc =Fc['moving']

        warp_patches = []
        if config.TestFlowSave == 'on':
            flow_px = []
            flow_py = []
            flow_pz = []

        for v in range(0, len(X)):
            patch = X[v, :, :, :]
            patch = np.reshape(patch, (1, config.TestPatchSize[0], config.TestPatchSize[1], config.TestPatchSize[2], 1))
            #atlas_patch = Xc[v, :, :, :]
            atlas_patch = config.TestAtlas[v, :, :, :]
            atlas_patch = np.reshape(atlas_patch,
                                     (1, config.TestPatchSize[0], config.TestPatchSize[1], config.TestPatchSize[2], 1))

            f_image, i_image, flow = sess.run([f_im, i_im, Flow], feed_dict={src_place: patch, tgt_place: atlas_patch})
            f_image = np.reshape(f_image, config.TestPatchSize)
            warp_patches.append(f_image)
            print("Image File {0} Patch {1} Warped\n".format(test_file_list[k], v))
            
            if config.TestFlowSave == 'on':
                flow_px.append(flow[0,:, :, :, 0])
                flow_py.append(flow[0,:, :, :, 1])
                flow_pz.append(flow[0,:, :, :, 2])
                flow_px = list2array(flow_px)
                flow_py = list2array(flow_py)
                flow_pz = list2array(flow_pz)

        ########### Creating array from list ############
        warped = list2array(warp_patches)
        data_file= "00"+str(k+1)+"_registered_{0}.h5".format(config.TestModelNumber)
        hf = h5py.File(data_file,"w")
        hf.create_dataset('moving', data=warped)
        hf.close()
        end_time = time.clock()
        print("Registration Time :{0} s".format(end_time-start_time))
        print("Done\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Configuration File Name Required")
    else:
        test(sys.argv[1])


