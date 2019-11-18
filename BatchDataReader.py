import random
import nibabel as nib
import h5py
import numpy as np
import SimpleITK as sitk
from measurment import *
import utility

class BatchDataset:
    images = []
    atlas  = []
    batch_offset = 0
    epochs_completed = 0

    def __init__(self, data_list, batch_size, config, lower=0.1, upper=0.5):   ### 0.1 for 10% and 0.2 for 25%
        print("Initializing Batch Dataset Reader...")
        self.image_files = data_list
        self.patch_num = config.Number_of_patches
        self.mean_lower = lower
        self.mean_upper = upper
        self.Batch_size = batch_size
        self.flow_loss = config.flow_loss
        self.patch_size = config.patch_size
        self.min_prob, self.max_prob, self.Z,self.K = self.create_pdf()

    def create_pdf(self):
        k = 4.6
        step = 0.9/self.patch_num
        mu = np.arange(0, 0.9, step)
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
    
    def list2array(self,list_data):
        array_data = np.concatenate([pat[np.newaxis, ...] for pat in list_data], axis=0)
        return array_data

    def create_batch(self,patch_data):
        batches = len(patch_data)//self.Batch_size
        data_batches = []
        
        s=0

        for b in range(batches):
            batch = patch_data[s:s+self.Batch_size,:,:,:]
            batch = np.reshape(batch, (self.Batch_size, self.patch_size[0], self.patch_size[1], self.patch_size[2], 1))
            data_batches.insert(b, batch)
            s += self.Batch_size
        return data_batches

        
        

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

        crop_image = np.array(crop_image)

        crop_atlas = np.array(crop_atlas)

        #location = np.array([random_center_x, random_center_y, random_center_z])

        if self.flow_loss == 'on':
           fx_patch = self.flow_fx[s_x:e_x,s_y:e_y,s_z:e_z]
           fy_patch = self.flow_fy[s_x:e_x,s_y:e_y,s_z:e_z]
           fz_patch = self.flow_fz[s_x:e_x,s_y:e_y,s_z:e_z]
           ix_patch = self.flow_ix[s_x:e_x,s_y:e_y,s_z:e_z]
           iy_patch = self.flow_iy[s_x:e_x,s_y:e_y,s_z:e_z]
           iz_patch = self.flow_iz[s_x:e_x,s_y:e_y,s_z:e_z]
           return crop_image, crop_atlas, [fx_patch,fy_patch,fz_patch,ix_patch,iy_patch,iz_patch] 
        else:    
            return crop_image, crop_atlas
    
    def create_pairs(self, selector_src, selector_tgt):

        src = self.load_image(self.image_files[selector_src])
        tgt = self.load_image(self.image_files[selector_tgt])

        locat = np.zeros((2502, 7), dtype=np.float)
        
        if self.flow_loss == 'on':
           flow = utility.Generate_deformation(src, 60)
           self.flow_fx = flow[:,:,:,0]
           self.flow_fy = flow[:,:,:,1]
           self.flow_fz = flow[:,:,:,2]
           self.flow_ix = flow[:,:,:,3]
           self.flow_iy = flow[:,:,:,4]
           self.flow_iz = flow[:,:,:,5]

        p_count =0
        
        src_patches = []
        tgt_patches = []
        #src_patch_ed =[]
        #tgt_patch_ed =[]
        if self.flow_loss == 'on':
           fx_patches = []
           fy_patches = []
           fz_patches = []
           ix_patches = []
           iy_patches = []
           iz_patches = []

        while True:
            if self.flow_loss == 'on':
               src_patch, tgt_patch, flow_patches = self.random_crop(src,tgt,self.patch_size)
            else:
               src_patch, tgt_patch = self.random_crop(src, tgt, self.patch_size)

            src_prob = self.check_probabiliy(src_patch)
            tgt_prob = self.check_probabiliy(tgt_patch)

            if (src_prob >= self.min_prob and src_prob<= self.max_prob) and (tgt_prob >=self.min_prob and tgt_prob<=self.max_prob):
                src_patches.append(src_patch)
                tgt_patches.append(tgt_patch)
                #locat[p_count,:] = np.array([loc[0], loc[1], loc[2], np.mean(src_patch), src_prob, np.mean(tgt_patch), tgt_prob])
                #f = open("pacth_locations.txt","a")
                #s = "X={0} Y={1} Z={2} \t mu_s={3} mu_t={4} \t prob_s={5} prob_t={6}\n".format(loc[0], loc[1], loc[2],
                #                                                                         np.mean(src_patch),
                #                                                                         np.mean(tgt_patch), src_prob,
                #                                                                         tgt_prob)
                #f.write(s)
                #f.close()
                
                #print("src mu:{0} src p:{1}  tgt mu:{2} tgt p:{3}\n".format(np.mean(src_patch), src_prob,
                #                                                           np.mean(tgt_patch), tgt_prob))
                if self.flow_loss == 'on':
                   fx_patches.append(flow_patches[0])
                   fy_patches.append(flow_patches[1])
                   fz_patches.append(flow_patches[2])
                   ix_patches.append(flow_patches[3])
                   iy_patches.append(flow_patches[4])
                   iz_patches.append(flow_patches[5])
                
                p_count += 1

            if p_count >= self.patch_num:
                break
        
        src_patches = self.list2array(src_patches)
        src_patches = self.create_batch(src_patches)
        tgt_patches = self.list2array(tgt_patches)
        tgt_patches = self.create_batch(tgt_patches)
        '''
        from scipy import stats
        import pandas as pd
        m = stats.mode(locat)
        locat[2501,:] = m[0]
        df = pd.DataFrame(locat)
        df.to_csv("2500_0.1_to_0.5.csv")
        '''
        if self.flow_loss == 'on':
           fx_patches = self.list2array(fx_patches)
           fx_patches = self.create_batch(fx_patches)
           fy_patches = self.list2array(fy_patches)
           fy_patches = self.create_batch(fy_patches)
           fz_patches = self.list2array(fz_patches)
           fz_patches = self.create_batch(fz_patches)
           ix_patches = self.list2array(ix_patches)
           ix_patches = self.create_batch(ix_patches)
           iy_patches = self.list2array(iy_patches)
           iy_patches = self.create_batch(iy_patches)
           iz_patches = self.list2array(iz_patches)
           iz_patches = self.create_batch(iz_patches)

           flow = [fx_patches, fy_patches, fz_patches, ix_patches, iy_patches, iz_patches]

           return src_patches, tgt_patches, flow
        else:
            return src_patches, tgt_patches

