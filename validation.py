import numpy as np
import  glob
from scipy.ndimage.filters import gaussian_filter
import SimpleITK as sitk
import nibabel as nib
from partition import set_mid_img, crop_mid_img, extract_patches, list2array, normalize_intensity, patition_img
import h5py

def ApplyRandomDeformation(image,sigma,Points,file_loc,shape):

    maxdeform = 1.0

    image = crop_mid_img(image,[shape[0],shape[1],shape[2]])
    image = image.T
    im_shape = image.shape
    RDF =np.zeros([im_shape[0],im_shape[1],im_shape[2],3],dtype=np.float64)
    RDFx =np.zeros(im_shape,dtype=np.float64)
    RDFy =np.zeros(im_shape,dtype=np.float64)
    RDFz =np.zeros(im_shape,dtype=np.float64)
    RDFxf =np.zeros(im_shape,dtype=np.float64)
    RDFyf =np.zeros(im_shape,dtype=np.float64)
    RDFzf =np.zeros(im_shape,dtype=np.float64)


    #image = image.T
    above_zero =np.where(image >0.2)

    k = 0
    while (k < Points):
        voxel_idx = long(np.random.randint(0, len(above_zero[0]) - 1, 1, dtype=np.int64))
        z = above_zero[0][voxel_idx]
        y = above_zero[1][voxel_idx]
        x = above_zero[2][voxel_idx]
        '''
        while 1:
            voxel_idx = long(np.random.randint(0, len(above_zero[0]) - 1, 1, dtype=np.int64))
            z = above_zero[0][voxel_idx]
            y = above_zero[1][voxel_idx]
            x = above_zero[2][voxel_idx]

            if x>=im_shape[0]-64 or y>= im_shape[1]-64 or z >= im_shape[2]-64:
                continue
            else:
                break
        '''
        Dx = ((np.random.ranf([1]))[0] - 0.5) * maxdeform
        Dy = ((np.random.ranf([1]))[0] - 0.5) * maxdeform
        Dz = ((np.random.ranf([1]))[0] - 0.5) * maxdeform

        RDFx[z, y, x] = Dx
        RDFy[z, y, x] = Dy
        RDFz[z, y, x] = Dz

        # print "Point:"+str(k)
        k += 1

    # del BorderMask

    RDFxf = gaussian_filter(RDFx, sigma=sigma)
    RDFyf = gaussian_filter(RDFy, sigma=sigma)
    RDFzf = gaussian_filter(RDFz, sigma=sigma)

    ####################################### Normalization #############################################
    IXp = np.where(RDFxf > 0)
    IXn = np.where(RDFxf < 0)
    IYp = np.where(RDFyf > 0)
    IYn = np.where(RDFyf < 0)
    IZp = np.where(RDFzf > 0)
    IZn = np.where(RDFzf < 0)

    #### Normalizing x-direction ###

    if len(IXp[0]) > 0:
        RDFxf[IXp] = ((np.max(RDFx) - 0) / (np.max(RDFxf[IXp]) - np.min(RDFxf[IXp])) * (
                    RDFxf[IXp] - np.min(RDFxf[IXp])) + 0)

    if len(IXn[0]) > 0:
        RDFxf[IXn] = ((0 - np.min(RDFxf[IXn])) / (0 - np.min(RDFxf[IXn])) * (
                    RDFxf[IXn] - np.min(RDFxf[IXn])) + np.min(RDFxf[IXn]))

    #### Normalizing y-direction ####
    if len(IYp[0]) > 0:
        RDFyf[IYp] = ((np.max(RDFy) - 0) / (np.max(RDFyf[IYp]) - np.min(RDFyf[IYp])) * (
                    RDFyf[IYp] - np.min(RDFyf[IYp])) + 0)
    if len(IYn[0]) > 0:
        RDFyf[IYn] = ((0 - np.min(RDFyf[IYn])) / (0 - np.min(RDFyf[IYn])) * (
                    RDFyf[IYn] - np.min(RDFyf[IYn])) + np.min(RDFyf[IYn]))

    #######Normalizing z-direction ####

    if len(IZp[0]) > 0:
        RDFzf[IZp] = ((np.max(RDFz) - 0) / (np.max(RDFzf[IZp]) - np.min(RDFzf[IZp])) * (
                    RDFzf[IZp] - np.min(RDFzf[IZp])) + 0)
    if len(IZn[0]) > 0:
        RDFzf[IZn] = ((0 - np.min(RDFzf[IZn])) / (0 - np.min(RDFzf[IZn])) * (
                    RDFzf[IZn] - np.min(RDFzf[IZn])) + np.min(RDFzf[IZn]))

    RDF[:, :, :, 0] = RDFxf
    RDF[:, :, :, 1] = RDFyf
    RDF[:, :, :, 2] = RDFzf

    RDFobj = sitk.GetImageFromArray(RDF, isVector=True)
    imgObj = sitk.ReadImage(file_loc)
    RDFobj.SetOrigin(imgObj.GetOrigin())
    RDFobj.SetSpacing(imgObj.GetSpacing())
    RDFobj.SetDirection(imgObj.GetDirection())

    RDFtr = sitk.DisplacementFieldTransform(RDFobj)
    DefomedIm = sitk.Resample(imgObj, RDFtr)

    def_file_name = file_loc.split('/')[-1].split('.')[0]+".deformed.nii.gz"
    directory = file_loc[:len(file_loc)-len(file_loc.split('/')[-1])]
    sitk.WriteImage(sitk.Cast(DefomedIm, sitk.sitkVectorFloat32),directory+def_file_name)

    DefomedIm = sitk.GetArrayFromImage(DefomedIm)

    return DefomedIm
    '''
    T_RDF = sitk.GetImageFromArray(RDF)
    T_RDF.SetOrigin(imgObj.GetOrigin())
    T_RDF.SetSpacing(imgObj.GetSpacing())
    sitk.WriteImage(sitk.Cast(T_RDF, sitk.sitkVectorFloat32), def_name)
    '''

if __name__=="__main__":

    dir = "/home/n9614885/patchvm_data/25%/test/RandomDeform/"
    files = glob.glob(dir+"*.deform.nii.gz")
    for i in range(len(files)):
        image = nib.load(files[i]).get_data()
        image = normalize_intensity(image,[0.0,1.0])

        #def_vol =ApplyRandomDeformation(image,112,150,files[i],[184,216,32]) # for 10%
        #padded = set_mid_img(image, (184, 216, 32), (256, 256, 32))
        #mov_patches = list2array(patition_img(padded,65,32))

        #def_vol = ApplyRandomDeformation(image, 112, 150, files[i], [512, 540, 169]) #for 25%
        padded = set_mid_img(image, (540, 540, 169), (576, 576, 192))  # for 25% resolution
        mov_patches = list2array(extract_patches(padded, (64, 64, 64), 0.0))

        data_file = dir + "moving_def_" + str(i) + ".h5"
        hf = h5py.File(data_file, "w")
        hf.create_dataset('moving', data=mov_patches)
        hf.close()





