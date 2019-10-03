import numpy as np
import glob

epochs=10
lr = 1e-4
reg_param = 1.5
model_save_iter = 100

# UNET filters
nf_enc = [16,32,32,32,16]
# VM-1
nf_dec = [32,32,32,32,8]
# VM-2
#nf_dec = [32,32,32,32,32,16,16,3]
vol_size = (64, 64, 64)
base_data_dir = "/home/n9614885/patchvm_data/"
#train_vol_dir = '/vols/train/synthetic/*.nii.gz'
#st='/home/n9614885/myvoxelmorph/data/vols/train/synthetic/'
atlas_name = '003_nuclear_10%.nii.gz'
gpu_id=0
model_name="patchDDN_canny"
start="0"

