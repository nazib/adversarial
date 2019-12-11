import numpy as np
import nibabel as nib
import glob
import os
import sys
import scipy
import cv2
import random

vol_size = [2160,2560]
patch_size = [64,64,64]

patch_path = "/home/n9614885/100%_patches/"

def extract_patches(path):

    folders = os.listdir(path)
    for f in folders:
        slices = glob.glob(os.path.join(path,f)+"/*.tif")
        x = 2160
        y = 2560
        z = len(slices)
        k = 0
        while True:
            random_center_x = random.randint(patch_size[0] / 2, x - patch_size[0] / 2)
            random_center_y = random.randint(patch_size[1] / 2, y - patch_size[1] / 2)
            random_center_z = random.randint(patch_size[2] / 2, z - patch_size[2] / 2)

            s_x = random_center_x - patch_size[0] // 2
            e_x = random_center_x + patch_size[0] // 2
            s_y = random_center_y - patch_size[1] // 2
            e_y = random_center_y + patch_size[1] // 2
            s_z = random_center_z - patch_size[2] // 2
            e_z = random_center_z + patch_size[2] // 2

            patch = np.zeros(patch_size,dtype=np.float)

            i=0
            for s in slices:
                sl = cv2.imread(s,0)
                patch[:,:, i] = sl[s_x:e_x, s_y:e_y]
                i+=1

            k+=1
            dir = os.path.join(patch_path,f)

            if not os.path.isdir(dir):
                os.mkdir(dir)

            image = nib.Nifti1Image(patch,np.eye(4, 4))
            nib.save(image, dir+"/patch_{0}.nii.gz")

            if k ==2500:
                break

        print(" Folder :{0} Patches :{1}\n".format(f,k))




if __name__ =="__main__":
    if len(sys.argv) != 2:
        print("Enter the path:")
    else:
        path = sys.argv[1]
        extract_patches(path)