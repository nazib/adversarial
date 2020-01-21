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
from refnet import ref_net
from DenseDeform import DenseDeformNet
# from DenseDeformOld import DenseDeformNet
from InverseNet import InverseNet
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

    # canny_file_list = glob.glob(config.TestDataDir+"*.canny.h5")

    if config.TestResolution == '10':
        #fused_atlas = patches2Img(config.TestAtlas, config.TestImageSize,config.TestPatchSize[0] + 1, 32)
        patchEx = PatchUtility((64,64,32),config.TestImageSize,config.TestOverlap,patches=config.TestAtlas,image=[])
        fused_atlas = patchEx.combine_patches()
        #fused_atlas = fused_atlas[:,:,:68]
        #atlas_volumes = load_validation_data(validation_atlases, config)

    if config.TestResolution == '25':
        #fused_atlas = nib.load(config.TestDataDir + "003_nuclear.nii.gz").get_data()
        #fused_atlas_slice = fused_atlas[:, :, 86]
        patchEx = PatchUtility((64,64,64),config.TestImageSize,config.TestOverlap,patches=config.TestAtlas,image=[])
        fused_atlas = patchEx.combine_patches()
        fused_atlas = set_mid_img(fused_atlas)
        #fused_atlas = fused_atlas[:, 0:375, :]
        #atlas_volumes = load_validation_data(validation_atlases,config)

    # atlas = nib.load('/home/n9614885/patchvm_data/original/test/003_nuclear.nii.gz')
    # atlas =atlas.get_data()

    #  Saving Atlas patch

    if config.TestCubeSave == 'on':
        cube = nib.Nifti1Image(config.TestAtlas[config.TestCubeNumber, :, :, :], config.TestAffineMatrix)
        nib.save(cube, "atlas_" + str(config.TestCubeNumber) + "_cube.nii.gz")

    confignet = tf.ConfigProto()
    confignet.gpu_options.allow_growth = True
    confignet.allow_soft_placement = True
    set_session(tf.Session(config=confignet))
    start_time = time.clock()

    # load weights of model
    with tf.device(config.TestGPU):
        if config.TestModelType == 'DDN':
            vol_size = (config.TestPatchSize[0], config.TestPatchSize[1], config.TestPatchSize[2])
            net = DenseDeformNet(vol_size,'off','on','on')
            net = net.createModel()
        if config.TestModelType == 'Inverse':
            vol_size = (config.TestPatchSize[0], config.TestPatchSize[1], config.TestPatchSize[2])
            net = InverseNet(vol_size, 'off', 'off', 'on')
            net = net.createModel()
        if config.TestModelType == 'VM':
            nf_enc = [16, 32, 32, 32, 16]
            nf_dec = [32, 32, 32, 32, 8]
            net = ref_net((config.TestPatchSize[0], config.TestPatchSize[1], config.TestPatchSize[2]), nf_enc, nf_dec)

        net.load_weights(config.TestModelDir + config.TestModelNumber + ".h5")

    n_batches = len(test_file_list)
    xx = np.arange(config.TestPatchSize[0])
    yy = np.arange(config.TestPatchSize[1])
    zz = np.arange(config.TestPatchSize[2])
    grid = np.rollaxis(np.array(np.meshgrid(xx, yy, zz)), 0, 4)
    flow = np.zeros((1, config.TestPatchSize[0], config.TestPatchSize[1], config.TestPatchSize[2], 3), dtype=np.float)
    warp_patches = []
    np.random.seed(17)
    i = 1

    for k in range(len(test_file_list)):
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

            with tf.device(config.TestGPU):
                pred = net.predict([patch, atlas_patch])

            # Warp segments with flow
            ''' 
            flow = pred[1][0, :, :, :, :]
            sample = flow + grid
            sample = np.stack((sample[:, :, :, 1], sample[:, :, :, 0], sample[:, :, :, 2]), 3)

            warp_cube = interpn((yy, xx, zz), patch[0, :, :, :, 0], sample, method='nearest', bounds_error=False,
                                fill_value=0)
            '''
            flow = pred[1][0, :, :, :, :]
            warp_cube = pred[0][0,:,:,:,:]
            warp_cube = np.reshape(warp_cube, (config.TestPatchSize[0], config.TestPatchSize[1], config.TestPatchSize[2]))
            warp_patches.append(warp_cube)
            if config.TestFlowSave == 'on':
                flow_px.append(flow[:, :, :, 0])
                flow_py.append(flow[:, :, :, 1])
                flow_pz.append(flow[:, :, :, 2])

        ########### Creating array from list ############
        warped = list2array(warp_patches)
        if config.TestFlowSave == 'on':
            flowx = list2array(flow_px)
            flowy = list2array(flow_py)
            flowz = list2array(flow_pz)

        if config.TestCubeSave == 'on':
            cube = nib.Nifti1Image(warped[config.TestCubeNumber, :, :, :], config.TestAffineMatrix)
            nib.save(cube, "00" + str(i) + "_" + str(config.TestCubeNumber) + "_cube.nii.gz")

        if config.TestResolution == '25':
            patchEx = PatchUtility((64,64,64),config.TestImageSize,config.TestOverlap,patches=warped,image=[])
            warp_image = patchEx.combine_patches()
            #warp_image = set_mid_img(warp_image)
            #fused_atlas = set_mid_img(atlas_volumes[k,:,:,:])
            #warp_image =warp_image[:,0:375,:]
            #overlay_slices(warp_image[:, :, config.TestSliceNo], fused_atlas[:, :, config.TestSliceNo], config, i)
            config.TestSliceNo = 50
            overlay_slices(warp_image[:, :, config.TestSliceNo], fused_atlas[:, :, config.TestSliceNo], config, i)
            config.TestSliceNo = 65
            overlay_slices(warp_image[:, :, config.TestSliceNo], fused_atlas[:, :, config.TestSliceNo], config, i)
            config.TestSliceNo = 75
            overlay_slices(warp_image[:, :, config.TestSliceNo], fused_atlas[:, :, config.TestSliceNo], config, i)

            if config.TestFlowSave == 'on':
                patchEx = PatchUtility((64,64,64),config.TestImageSize,config.TestOverlap,patches=flowx,image=[])
                flowx = patchEx.combine_patches()
                patchEx = PatchUtility((64,64,64),config.TestImageSize,config.TestOverlap,patches=flowy,image=[])
                flowy = patchEx.combine_patches()
                patchEx = PatchUtility((64,64,64),config.TestImageSize,config.TestOverlap,patches=flowz,image=[])
                flowz = patchEx.combine_patches()
                flowx = set_mid_img(flowx)
                flowy = set_mid_img(flowy)
                flowz = set_mid_img(flowz)

        if config.TestResolution == '10':
            ims = config.TestImageSize
            patchEx = PatchUtility((64,64,32),config.TestImageSize,config.TestOverlap,patches=warped,image=[])
            warp_image = patchEx.combine_patches()
            warp_image = crop_mid_img(warp_image, (256, 216, 68))
            fused_atlas = crop_mid_img(fused_atlas, (256, 216, 68))
            #fused_atlas = atlas_volumes[k,:,:,:]
            #warp_image = warp_image[:,:,:68]
            overlay_slices(warp_image[:, :, config.TestSliceNo], fused_atlas[:, :, config.TestSliceNo], config, i)
            #fused_atlas = atlas_volumes[k, :, :, :]
            if config.TestFlowSave == 'on':
                patchEx = PatchUtility((64, 64, 32), config.TestImageSize, config.TestOverlap, patches=flowx, image=[])
                flowx = patchEx.combine_patches()
                patchEx = PatchUtility((64, 64, 32), config.TestImageSize, config.TestOverlap, patches=flowy, image=[])
                flowy = patchEx.combine_patches()
                patchEx = PatchUtility((64, 64, 32), config.TestImageSize, config.TestOverlap, patches=flowz, image=[])
                flowz = patchEx.combine_patches()
                flowx = crop_mid_img(flowx,(256,216,68))
                flowy = crop_mid_img(flowy,(256,216,68))
                flowz = crop_mid_img(flowz,(256,216,68))

        warped = nib.Nifti1Image(warp_image, config.TestAffineMatrix)
        nib.save(warped, "00" + str(i) + "_registered.nii.gz")

        if config.TestFlowSave == 'on':

            flow = np.zeros((256,216,68,3))
            flow[:,:,:,0] = flowx
            flow[:, :, :, 1] = flowy
            flow[:, :, :, 2] = flowz
            flow_image = nib.Nifti1Image(flow, config.TestAffineMatrix)
            nib.save(flow_image, "00" + str(i) + ".flow.nii.gz")
            '''
            flow_x = nib.Nifti1Image(flowx, config.TestAffineMatrix)
            flow_y = nib.Nifti1Image(flowy, config.TestAffineMatrix)
            flow_z = nib.Nifti1Image(flowz, config.TestAffineMatrix)
            nib.save(flow_x, "00" + str(i) + "_x.flow.nii.gz")
            nib.save(flow_y, "00" + str(i) + "_y.flow.nii.gz")
            nib.save(flow_z, "00" + str(i) + "_z.flow.nii.gz")
            '''

        end_time = time.clock()
        cc = volume_cross(warp_image,fused_atlas)
        mi = mutual_info(warp_image, fused_atlas, 30)
        output = "Brain: " + test_file_list[k] + " Ref Brain:" + config.TestAtlasFile + " CC: " + str(cc) + " MI: " + str(mi) + " Time:" + str(end_time - start_time) + "\n"
        #output = "Brain: " + test_file_list[k] + " Ref Brain:" + validation_atlases[k] + " CC: " + str(
        #   cc) + " MI: " + str(mi) + " Time:" + str(end_time - start_time) + "\n"
        print(output)

        del warp_patches[:]
        i +=1

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Configuration File Name Required")
    else:
        test(sys.argv[1])
