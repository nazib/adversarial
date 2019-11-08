import random
import nibabel as nib
import h5py
import numpy as np
import SimpleITK as sitk
from measurment import *

class BatchDataset:
    images = []
    atlas  = []
    batch_offset = 0
    epochs_completed = 0

    def __init__(self, data_list, batch_size, number_of_patches=2500, lower=0.05, upper=0.2):   ### 0.1 for 10% and 0.2 for 25%
        print("Initializing Batch Dataset Reader...")
        self.image_files = data_list
        self.patch_num = number_of_patches
        self.mean_lower = lower
        self.mean_upper = upper
        self.Batch_size = batch_size
        self.min_prob, self.max_prob, self.Z,self.K = self.create_pdf()

    def create_pdf(self):
        k=11.5
        step = 0.5/self.patch_num
        mu = np.arange(0, 0.5, step)
        prob = np.ones(len(mu))
        prob[np.where(mu < self.mean_lower)] = 0
        prob[np.where(mu > self.mean_upper)] = 10.0 / np.exp(k * mu[np.where(mu > self.mean_upper)])
        Z = np.sum(prob)
        prob = prob/Z
        min_prob = np.mean(prob) - 0.0001
        max_prob = np.max(prob)
        return min_prob, max_prob, Z, k


    def check_probabiliy(self, patch):
        mu = np.mean(patch)
        if (mu >= self.mean_lower) and (mu <= self.mean_upper):
            prob = 1.0
            prob = prob/self.Z
        elif mu > self.mean_upper:
            prob = 10.0/np.exp(self.K*mu)
            prob = prob/self.Z
        else:
            prob = 0.0

        return prob

    def load_image(self, filename):
        f = nib.load(filename)
        image = f.get_data()
        return image

    def load_atlas(self, filename):
        f = nib.load(filename)
        atlas = f.get_data()
        atlas = np.squeeze(atlas)
        # label = np.transpose(label, [2, 1, 0])
        return atlas

    def random_crop(self, images, atlas,size):
        crop_image = []
        crop_labels = []
        
        x, y, z = images.shape
        if size[0] > x or size[1] > y or size[2] > z:
            raise IndexError("Please input the right size")
        random_center_x = random.randint(size[0] / 2, x - size[0] / 2)
        random_center_y = random.randint(size[1] / 2, y - size[1] / 2)
        random_center_z = random.randint(size[2] / 2, z - size[2] / 2)

        s_x = random_center_x - size[0] // 2
        e_x = random_center_x + size[0] // 2
        s_y = random_center_y - size[1] // 2
        e_y = random_center_y + size[1] // 2
        s_z = random_center_z - size[2] // 2
        e_z = random_center_z + size[2] // 2

        crop_image = images[s_x:e_x,s_y:e_y,s_z:e_z]

        crop_atlas = atlas[s_x:e_x,s_y:e_y,s_z:e_z]
        '''
        crop_image_ed = image_edges[random_center_x - size / 2: random_center_x + size / 2,
                        random_center_y - size / 2: random_center_y + size / 2,
                        random_center_z - size / 2: random_center_z + size / 2]
        crop_atlas_ed = atlas_edges[random_center_x - size / 2: random_center_x + size / 2,
                        random_center_y - size / 2: random_center_y + size / 2,
                        random_center_z - size / 2: random_center_z + size / 2]
        '''
        crop_image = np.array(crop_image)
        #crop_image = np.reshape(crop_image, (self.Batch_size, size[0], size[1], size[2], 1))

        crop_atlas = np.array(crop_atlas)
        #crop_atlas = np.reshape(crop_atlas, (self.Batch_size, size[0], size[1], size[2], 1))
        #location = np.array([random_center_x,random_center_y,random_center_z])

        return crop_image, crop_atlas #location
    
    def create_pairs(self, selector_src,selector_tgt,patch_size):

        src = self.load_image(self.image_files[selector_src])
        tgt = self.load_image(self.image_files[selector_tgt])

        '''
        ############  Canny Edge detection ###########
        src_sitk = sitk.GetImageFromArray(src)
        tgt_sitk = sitk.GetImageFromArray(tgt)
        
        src_sitk = sitk.Cast(src_sitk, sitk.sitkFloat32)
        tgt_sitk = sitk.Cast(tgt_sitk, sitk.sitkFloat32)

        src_edges = sitk.CannyEdgeDetection(src_sitk, lowerThreshold=0.02, 
                                upperThreshold=0.05)
        tgt_edges = sitk.CannyEdgeDetection(tgt_sitk, lowerThreshold=0.02, 
                                upperThreshold=0.05)
        
        src_edges = sitk.GetArrayFromImage(src_edges)
        tgt_edges = sitk.GetArrayFromImage(tgt_edges)
        '''
        p_count =0
        
        src_patches = []
        tgt_patches = []
        #src_patch_ed =[]
        #tgt_patch_ed =[]
        all_locations =[]
        while True:
            src_patch, tgt_patch = self.random_crop(src,tgt,patch_size)
            src_prob = self.check_probabiliy(src_patch)
            tgt_prob = self.check_probabiliy(tgt_patch)
            '''
            if (np.mean(src_patch)>=self.mean_condition) and (np.mean(tgt_patch)>=self.mean_condition):
                src_patches.append(src_patch)
                tgt_patches.append(tgt_patch)
                p_count+=1
            '''

            if (src_prob >= self.min_prob and src_prob<= self.max_prob) and (tgt_prob >=self.min_prob and tgt_prob<=self.max_prob):
                src_patches.append(src_patch)
                tgt_patches.append(tgt_patch)
                #print("src mu:{0} src p:{1}  tgt mu:{2} tgt p:{3}\n".format(np.mean(src_patch), src_prob,
                #                                                           np.mean(tgt_patch), tgt_prob))
                p_count += 1

            if p_count >=self.patch_num:
                break
        
        src_patches = np.concatenate([pat[np.newaxis, ...] for pat in src_patches], axis=0)
        tgt_patches = np.concatenate([pat[np.newaxis, ...] for pat in tgt_patches], axis=0)

        batches = len(src_patches)//self.Batch_size
        src_batches = []
        tgt_batches = []
        s=0

        for b in range(batches):
            src_batch = src_patches[s:s+self.Batch_size,:,:,:]
            src_batch = np.reshape(src_batch, (self.Batch_size, patch_size[0], patch_size[1], patch_size[2], 1))
            tgt_batch = tgt_patches[s:s+self.Batch_size,:,:,:]
            tgt_batch = np.reshape(tgt_batch, (self.Batch_size, patch_size[0], patch_size[1], patch_size[2], 1))
            src_batches.insert(b, src_batch)
            tgt_batches.insert(b, tgt_batch)
            s += self.Batch_size

        return src_batches,tgt_batches

