import glob
from time import localtime, strftime
from subprocess import call
import sys
import os
import SimpleITK as sitk

data_dir="/home/n9614885/patchvm_data/test/"
ext="*.deform.nii.gz"

files = glob.glob(data_dir+ext)
#atlas_file = "/home/n9614885/myvoxelmorph/data/vols/15%/test/003_nuclear.nii.gz"
atlas_file = [data_dir+"001_nuclear.nii.gz",data_dir+"002_nuclear.nii.gz",data_dir+"003_nuclear.nii.gz"]
out_dir = "/home/n9614885/patchvm_data/test/"

for i in range(0,len(files)):
    
    moving_file = files[i]
    out_put_dir=out_dir+moving_file.split('/')[-1][:-7]
    if os.path.exists(out_put_dir) ==False:
        os.mkdir(out_put_dir)
    ## Regid+Affine+B-spline Registration    
    ## Regid Registration

    
    cmd="antsRegistration --dimensionality 3 --metric CC["+atlas_file[i]+","+moving_file+",1,2] "+\
    "--interpolation Linear --transform Rigid[0.1] --initial-moving-transform "+\
    "["+atlas_file[i]+","+moving_file+",1] "+\
    "--convergence [100x100x10,1e-6,10] --shrink-factors 12x8x4 "+ \
    "--smoothing-sigmas 4x3x1vox --float --output "+out_put_dir+"/rigid.mat -v"
    call([cmd],shell=True)

    
    cmd="antsRegistration --dimensionality 3 --metric CC["+atlas_file[i]+","+moving_file+",1,2] "+\
    "--interpolation Linear --transform Affine[0.1] --initial-moving-transform "+\
    "["+out_put_dir+"/rigid.mat0GenericAffine.mat] "+\
    "--convergence [100x100x10,1e-6,10] --shrink-factors 12x8x4 "+ \
    "--smoothing-sigmas 4x3x1vox --float --output "+out_put_dir+"/affine.mat -v"
    
    call([cmd],shell=True)
    
    cmd="antsApplyTransforms --dimensionality 3 -i "+\
    moving_file+" -o "+out_dir+moving_file.split('/')[-1]+" -r "+atlas_file[i]+" -t "+out_put_dir+"/affine.mat0GenericAffine.mat"+" -n NearestNeighbor -v"
    call([cmd],shell=True)
    
    message= "ANTS Affine Registration : Ends at--->"+ strftime("%H:%M:%S", localtime())
    print message
    print("File :"+files[i]+" Registered")
    
print("Done!")
