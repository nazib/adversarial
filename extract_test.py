import numpy as np
import tensorflow as tf
import nibabel as nib
import h5py

vol= nib.load('001_registered.nii.gz')
img = vol.get_data()
img = np.reshape(img,(1,640,540,169,1))

ksize = [1,64,64,64,1]
strides = [1,32,32,32,1]
rates = [1,1, 1,1, 1]
x = tf.placeholder(tf.float32,shape=(1,640,540,169,1),name="image")
patches = tf.extract_volume_patches(x,ksize,strides,'SAME')
patches=tf.reshape(patches,shape=(2040,64,64,64))

with tf.Session() as sess:
   all_patches= sess.run(patches,feed_dict={x:img})

print(all_patches.shape)
data_file="moving.h5"
hf = h5py.File(data_file,"w")
hf.create_dataset('moving', data=all_patches)
hf.close()
    






