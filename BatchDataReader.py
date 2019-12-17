import random
import nibabel as nib
import h5py
import numpy as np
import SimpleITK as sitk
#from measurment import *
from itertools import permutations
import partition
#import BatchDataReader
import os
import sys
import glob
import time
#from scipy.ndimage.filters import gaussian_filter

class BatchDataset:
    images = []
    atlas  = []
    batch_offset = 0
    epochs_completed = 0

    def __init__(self, data_dir, batch_size, patch_dir=None, config=None, lower=0.1, upper=0.5):   ### 0.1 for 10% and 0.2 for 25%
        print("Initializing Batch Dataset Reader...")
        self.image_files = glob.glob(data_dir+"/*.nii.gz")
        self.mean_lower = lower
        self.mean_upper = upper
        self.Batch_size = batch_size
        
        if patch_dir!=None:
            self.Patch_dir = patch_dir
     
        if config != None:
            self.flow_loss = config.flow_loss
            self.patch_size = config.patch_size
            self.patch_num = config.Number_of_patches
        else:
            self.patch_num = 2500
            self.patch_size =[64,64,64]
            self.flow_loss = None

    
        self.min_prob, self.max_prob, self.Z, self.K = self.create_pdf()

    def create_pdf(self):
        k = 6.6
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
    
    def normalize_intensity(self,brain,range):
        m,n,c=brain.shape
        imMin=np.min(brain)
        imMax=np.max(brain)

        nMin=range[0]
        nMax=range[1]

        multi=((nMax-nMin)/np.float((imMax-imMin)))+nMin
        imgMin=np.zeros([m,n,c],dtype=float)
        imgMin[:,:,:]=imMin

        brain=(brain-imgMin)*multi
        return brain


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
            #batch = np.reshape(batch, (self.Batch_size, self.patch_size[0], self.patch_size[1], self.patch_size[2], 1))
            data_batches.insert(b, batch)
            s += self.Batch_size
        return data_batches

    def random_crop(self, vol1, vol2):
        
        x, y, z = vol1.shape
        if self.patch_size[0] > x or self.patch_size[1] > y or self.patch_size[2] > z:
            raise IndexError("Please input the right size")
        random_center_x = random.randint(self.patch_size[0] / 2, x - self.patch_size[0] / 2)
        random_center_y = random.randint(self.patch_size[1] / 2, y - self.patch_size[1] / 2)
        random_center_z = random.randint(self.patch_size[2] / 2, z - self.patch_size[2] / 2)

        s_x = random_center_x - self.patch_size[0] // 2
        e_x = random_center_x + self.patch_size[0] // 2
        s_y = random_center_y - self.patch_size[1] // 2
        e_y = random_center_y + self.patch_size[1] // 2
        s_z = random_center_z - self.patch_size[2] // 2
        e_z = random_center_z + self.patch_size[2] // 2

        crop_vol1 = vol1[s_x:e_x,s_y:e_y,s_z:e_z]

        crop_vol2 = vol2[s_x:e_x,s_y:e_y,s_z:e_z]

        crop_vol1 = np.array(crop_vol1)

        crop_vol2 = np.array(crop_vol2)

        #location = np.array([random_center_x, random_center_y, random_center_z])

        if self.flow_loss == 'on':
           fx_patch = self.flow_fx[s_x:e_x,s_y:e_y,s_z:e_z]
           fy_patch = self.flow_fy[s_x:e_x,s_y:e_y,s_z:e_z]
           fz_patch = self.flow_fz[s_x:e_x,s_y:e_y,s_z:e_z]
           ix_patch = -self.flow_fx[s_x:e_x,s_y:e_y,s_z:e_z]
           iy_patch = -self.flow_fy[s_x:e_x,s_y:e_y,s_z:e_z]
           iz_patch = -self.flow_fz[s_x:e_x,s_y:e_y,s_z:e_z]
           return crop_vol2, crop_vol1, [fx_patch,fy_patch,fz_patch,ix_patch,iy_patch,iz_patch] 
        else:    
            return crop_vol2, crop_vol1
    
    def create_pairs(self, selector_src, selector_tgt):

        src = self.load_image(self.image_files[selector_src])
        tgt = self.load_image(self.image_files[selector_tgt])

        #locat = np.zeros((2502, 7), dtype=np.float)
        
        if self.flow_loss == 'on':
           flow = self.Generate_deformation(src, 60)
           self.flow_fx = flow[:,:,:,0]
           self.flow_fy = flow[:,:,:,1]
           self.flow_fz = flow[:,:,:,2]
           #self.flow_ix = flow[:,:,:,3]
           #self.flow_iy = flow[:,:,:,4]
           #self.flow_iz = flow[:,:,:,5]

        p_count =0
        src_patches = np.zeros((self.patch_num, self.patch_size[0], self.patch_size[1], self.patch_size[2], 1))
        tgt_patches = np.zeros((self.patch_num, self.patch_size[0], self.patch_size[1], self.patch_size[2], 1))
        s = 0
        #src_patch_ed =[]
        #tgt_patch_ed =[]
        if self.flow_loss == 'on':
           fx_patches = np.zeros((self.patch_num, self.patch_size[0], self.patch_size[1], self.patch_size[2], 1))
           fy_patches = np.zeros((self.patch_num, self.patch_size[0], self.patch_size[1], self.patch_size[2], 1))
           fz_patches = np.zeros((self.patch_num, self.patch_size[0], self.patch_size[1], self.patch_size[2], 1))
           ix_patches = np.zeros((self.patch_num, self.patch_size[0], self.patch_size[1], self.patch_size[2], 1))
           iy_patches = np.zeros((self.patch_num, self.patch_size[0], self.patch_size[1], self.patch_size[2], 1))
           iz_patches = np.zeros((self.patch_num, self.patch_size[0], self.patch_size[1], self.patch_size[2], 1))

        while True:
            if self.flow_loss == 'on':
               src_patch, tgt_patch, flow_patches = self.random_crop(src, tgt)
            else:
               src_patch, tgt_patch = self.random_crop(src, tgt)

            src_prob = self.check_probabiliy(src_patch)
            tgt_prob = self.check_probabiliy(tgt_patch)

            if (src_prob >= self.min_prob and src_prob <= self.max_prob) and (tgt_prob >=self.min_prob and tgt_prob<=self.max_prob):
                src_patches[p_count, :, :, :, 0] = src_patch
                tgt_patches[p_count, :, :, :, 0] = tgt_patch
                #src_patches.append(src_patch)
                #tgt_patches.append(tgt_patch)

                #locat[p_count,:] = np.array([loc[0], loc[1], loc[2], np.mean(src_patch), src_prob, np.mean(tgt_patch), tgt_prob])
                #f = open("pacth_locations.txt","a")
                #s = "X={0} Y={1} Z={2} \t mu_s={3} mu_t={4} \t prob_s={5} prob_t={6}\n".format(loc[0], loc[1], loc[2],
                #                                                                         np.mean(src_patch),
                #                                                                         np.mean(tgt_patch), src_prob,
                #                                                                         tgt_prob)
                #f.write(s)
                #f.close()
                
                print("src mu:{0} src p:{1}  tgt mu:{2} tgt p:{3} Count= {4}\n".format(np.mean(src_patch), src_prob,
                                                                           np.mean(tgt_patch), tgt_prob,p_count))
                if self.flow_loss == 'on':
                   fx_patches[p_count, :, :, :, 0] = flow_patches[0]
                   fy_patches[p_count, :, :, :, 0] = flow_patches[1]
                   fz_patches[p_count, :, :, :, 0] = flow_patches[2]
                   ix_patches[p_count, :, :, :, 0] = flow_patches[3]
                   iy_patches[p_count, :, :, :, 0] = flow_patches[4]
                   iz_patches[p_count, :, :, :, 0] = flow_patches[5]
                p_count += 1

            if p_count >= self.patch_num:
                break
        
        #src_patches = self.list2array(src_patches)
        src_patches = self.create_batch(src_patches)
        #tgt_patches = self.list2array(tgt_patches)
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
           #fx_patches = self.list2array(fx_patches)
           fx_patches = self.create_batch(fx_patches)
           #fy_patches = self.list2array(fy_patches)
           fy_patches = self.create_batch(fy_patches)
           #fz_patches = self.list2array(fz_patches)
           fz_patches = self.create_batch(fz_patches)
           #ix_patches = self.list2array(ix_patches)
           ix_patches = self.create_batch(ix_patches)
           #iy_patches = self.list2array(iy_patches)
           iy_patches = self.create_batch(iy_patches)
           #iz_patches = self.list2array(iz_patches)
           iz_patches = self.create_batch(iz_patches)

           del self.flow_fx
           del self.flow_fy
           del self.flow_fz
           del flow_patches
           del src_patch
           del tgt_patch

           flow = [fx_patches, fy_patches, fz_patches, ix_patches, iy_patches, iz_patches]

           return src_patches, tgt_patches, flow
        else:
            return src_patches, tgt_patches
    
    def extract_Hpatches(self):
    
        x = np.arange(len(self.image_files))
        total_pairs = list(permutations(x, 2))
        
        for pair in total_pairs:
            batch_pair=[]
            folder_str = "{0}/{1}_vs_{2}/".format(self.Patch_dir, pair[0],pair[1])
            
            if not os.path.isdir(folder_str):
                os.mkdir(folder_str)
            
            start = time.clock()

            im1 = self.load_image(self.image_files[pair[0]])
            im1 = self.normalize_intensity(im1,[0.0,1.0])
            im2 = self.load_image(self.image_files[pair[1]])
            im2 = self.normalize_intensity(im2,[0.0,1.0])
            
            p_count = 0
            src_patches = np.zeros((self.patch_num, 64, 64, 64, 1))
            tgt_patches = np.zeros((self.patch_num, 64, 64, 64, 1))

            while True:
                src_patch, tgt_patch = self.random_crop(im1, im2)
                src_prob = self.check_probabiliy(src_patch)
                tgt_prob = self.check_probabiliy(tgt_patch)
                
                if (src_prob >= self.min_prob and src_prob <= self.max_prob) and (tgt_prob >=self.min_prob and tgt_prob<=self.max_prob):
                    print("Mu 1 :{0} Mu 2:{1}\n".format(np.mean(src_patch),np.mean(tgt_patch)))
                    src_patches[p_count, :, :, :, 0] = src_patch
                    tgt_patches[p_count, :, :, :, 0] = tgt_patch

                p_count += 1

                if p_count >= self.patch_num:
                    break
                
            src_batch = self.create_batch(src_patches)
            tgt_batch = self.create_batch(tgt_patches)
            batch_pair.insert(0,src_batch)
            batch_pair.insert(1,tgt_batch)

            data_file = folder_str + "patch_pair.h5"
            hf = h5py.File(data_file,"w")
            hf.create_dataset('moving', data=batch_pair)
            hf.close()
            end = time.clock()
            print("Batch saved:{0} Time required:{1}".format(data_file, (end-start)))
        
        print("Dtabase Creation Complete")
    
    def Generate_deformation(self,img, sigma):
    
        #img = nib.load(img).get_data()
        #img = set_mid_img(img,cSize=(512,540,169),dSize=(576,576,192))
        m,n,c = img.shape
        if m!=n:
            img = partition.set_mid_img(img, cSize=(m-128,n,c), dSize=(m,m,c))
        
        flow = np.zeros((img.shape[0],img.shape[1],img.shape[2],6))
        #patch = np.reshape(patch, im_shape)
        Points = 50
        maxdeform = 0.5
        #mu = np.mean(patch)
        #pmax = np.max(patch)
        #above_zero = np.where(img <= (pmax-mu))
        above_zero = np.where(img <= 0.3)

        RDF = np.zeros([img.shape[0], img.shape[1], img.shape[2], 3], dtype=np.float64)
        RDFx = np.zeros(img.shape, dtype=np.float64)
        RDFy = np.zeros(img.shape, dtype=np.float64)
        RDFz = np.zeros(img.shape, dtype=np.float64)
        RDFxf = np.zeros(img.shape, dtype=np.float64)
        RDFyf = np.zeros(img.shape, dtype=np.float64)
        RDFzf = np.zeros(img.shape, dtype=np.float64)
        '''
        iRDFx = np.zeros(img.shape, dtype=np.float64)
        iRDFy = np.zeros(img.shape, dtype=np.float64)
        iRDFz = np.zeros(img.shape, dtype=np.float64)
        iRDFxf = np.zeros(img.shape, dtype=np.float64)
        iRDFyf = np.zeros(img.shape, dtype=np.float64)
        iRDFzf = np.zeros(img.shape, dtype=np.float64)
        '''
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

            #iRDFx[x, y, z] = Dx
            #iRDFy[x, y, z] = Dy
            #iRDFz[x, y, z] = Dz

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
        #iRDFxf = -RDFxf#gaussian_filter(RDFx, sigma=-sigma)
        #iRDFyf = -RDFyf#gaussian_filter(RDFy, sigma=-sigma)
        #iRDFzf = -RDFzf#gaussian_filter(RDFz, sigma=-sigma)
        ####################################### Normalization #############################################
    
        RDF[:, :, :, 0] = RDFxf
        RDF[:, :, :, 1] = RDFyf
        RDF[:, :, :, 2] = RDFzf
        #RDF[:, :, :, 3] = iRDFxf
        #RDF[:, :, :, 4] = iRDFyf
        #RDF[:, :, :, 5] = iRDFzf

        flow = RDF#ApplyDeform(patch,RDF,im_shape)

        return flow



