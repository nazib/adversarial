import numpy as np
import nibabel as nib
import math
import h5py
import cv2
import glob
import pdb
import numpy.matlib
from measurment import *
#from  utility import patches2Img
from matplotlib import pyplot as plt
#import tensorflow as tf
import SimpleITK as sitk
import math
import pandas as pd
#df = pd.DataFrame(columns=['Row','Col','Depth','Mean','Flag'])
data = []

class PatchUtility:
    def __init__(self,patchShape,imageShape,Overlap,patches=[],image=[]):
        
        if len(patches)>0:
            self.patchShape = patches[0,:,:,:].shape
        else:
            self.patchShape = patchShape
        
        if len(image)>0:
            self.image = image
        else:
            self.image = np.zeros(imageShape,dtype=np.float)

        self.imShape = imageShape
        self.overlap_allowed = Overlap
        self.patches = patches
        self.ExCount =0
        self.CmbCount =0

        if self.overlap_allowed > 0.0:
            self.jump_cols = int(self.patchShape[1] * self.overlap_allowed)-1
            self.jump_rows = int(self.patchShape[0] * self.overlap_allowed)-1

            if self.imShape[2]>self.patchShape[2]:
                self.jump_slice = int(self.patchShape[2] * self.overlap_allowed)-1
        else:
            self.jump_cols = int(self.patchShape[1])-1
            self.jump_rows = int(self.patchShape[0])-1

            if self.imShape[2]>self.patchShape[2]:
                self.jump_slice = int(self.patchShape[2])-1
        
        self.row_terminate =self.imShape[0]-int(self.patchShape[0]*self.overlap_allowed)
        self.col_terminate =self.imShape[1]-int(self.patchShape[1]*self.overlap_allowed)
        self.depth_terminate =self.imShape[2]-int(self.patchShape[2]*self.overlap_allowed)
    
    def weighted_patch(self,p1,p2,n=32,m=64):
        
        r,c,s =p1.shape
        w_patch = np.zeros(shape=(r,c,s),dtype=np.float)
        #w_patch[n:m,n:m,n:m] = p2[n:m,n:m,n:m]
        
        position = np.linspace(-1,1,m)

        weight = (2.0/(1+np.exp(-2*position)))-1  ### Tanh function
        weight = np.maximum(weight,0)             ### RELU function 
        #weight = (1.0-np.abs(position)           ### Traiangular function 
        #a= 0.1/(m-n)
        #b= n/(n-m)
        #weight = a*position+b                    ### Linear function
        #plt.plot(position,weight)
        #plt.show()

        weight = np.matlib.repmat(weight,m,1)
        weight = np.concatenate([weight[...,np.newaxis] for i in range(s)], axis=2)
        w_patch = p1*weight+(1-weight)*p2
        #w_patch[0:n,0:n,0:n] = p1[n:m,n:m,n:m]*weight1+(1-weight1)*p2[0:n,0:n,0:n]
        #w_patch[]
        return w_patch


    def depth_cut(self,all_starts):
        
        #f= open('state.txt',"a")
        while all_starts[2] < self.imShape[2]-int(self.patchShape[2]*self.overlap_allowed):
            region = (slice(all_starts[0], all_starts[0] + self.patchShape[0]),
                    slice(all_starts[1], all_starts[1] +self.patchShape[1]),
                    slice(all_starts[2], all_starts[2] + self.patchShape[2]))

            patch = self.image[region]

            self.patches.append(patch)
            #file = open("patch_mean.txt","a")
            if np.mean(patch)<0.2:
                k= "Row :{0}, Col:{1}, Dpth:{2}, Mean:{3}\n".format(all_starts[0], all_starts[1], all_starts[2], np.mean(patch))
                data.append([all_starts[0], all_starts[1], all_starts[2], np.mean(patch),0])
            else:
                k = "Row :{0}, Col:{1}, Dpth:{2}, Mean:{3}, Flaged\n".format(all_starts[0], all_starts[1], all_starts[2],
                                                                  np.mean(patch))
                data.append([all_starts[0], all_starts[1], all_starts[2], np.mean(patch),1])
            #file.write(k)
            #file.close()
            #if patch.shape!=(64,64,64):
            #    pdb.set_trace()
            #    print(" Shape :"+str(patch.shape))
            all_starts[2] += self.jump_slice
            self.ExCount+=1

            if all_starts[2]>self.imShape[2]:
                return self.patches
            else:
                all_starts[2] += 1
            
        return self.patches

    def depth_join(self,all_starts, option='image'):

        while all_starts[2] < self.depth_terminate:
            region = (slice(all_starts[0], all_starts[0] + self.patchShape[0]),
                      slice(all_starts[1], all_starts[1] + self.patchShape[1]),
                      slice(all_starts[2], all_starts[2] + self.patchShape[2]))
            patch = self.patches[self.CmbCount, :, :, :]
           
            if self.overlap_allowed > 0.0:

                if all_starts[2]>self.jump_slice and all_starts[2]<self.depth_terminate:
                    pre_start = all_starts[2]-self.jump_slice-1
                    d_region = (slice(all_starts[0], all_starts[0] + self.patchShape[0]),
                        slice(all_starts[1], all_starts[1] + self.patchShape[1]),
                        slice(pre_start, all_starts[2] + int(self.patchShape[2]/2)))
                    
                    c_patch = self.patches[self.CmbCount,:,:,:]
                    p_patch = self.image[d_region]
                    wpatch = self.weighted_patch(p_patch,c_patch)
                    self.image[region] = wpatch
                else:
                    patch = self.patches[self.CmbCount,:,:,:]
                    self.image[region] = patch

                
                #### weightening columnwise #### 
                if all_starts[1]> self.jump_cols and all_starts[1]< self.col_terminate:
                    pre_start = all_starts[1]-self.jump_cols-1
                    col_region =(slice(all_starts[0], all_starts[0] + self.patchShape[0]),
                        slice(pre_start, all_starts[1] + int(self.patchShape[1]/2)),
                        slice(all_starts[2], all_starts[2] + self.patchShape[2]))
                    pre_patch =self.image[col_region]
                    patch = self.patches[self.CmbCount,:,:,:]
                    wpatch = self.weighted_patch(pre_patch,patch)
                    self.image[region]=wpatch
                else:
                    patch = self.patches[self.CmbCount,:,:,:]
                    self.image[region] = patch
                
                #### Weightening Rowise ###
                if all_starts[0]> self.jump_rows and all_starts[0]< self.row_terminate:
                    pre_start = all_starts[0]-self.jump_rows-1
                    row_region =(slice(pre_start, all_starts[0] + int(self.patchShape[0]/2)),
                        slice(all_starts[1], all_starts[1] + self.patchShape[1]),
                        slice(all_starts[2], all_starts[2] + self.patchShape[2]))
                    pre_patch =self.image[row_region]
                    patch = self.patches[self.CmbCount,:,:,:]
                    wpatch = self.weighted_patch(pre_patch,patch)
                    self.image[region]=wpatch
                else:
                    patch = self.patches[self.CmbCount,:,:,:]
                    self.image[region] = patch
            else:
                self.image[region] = patch    
            
            #plt.imshow((self.image[:,:,86]*255).astype(int))
            #plt.show()
            all_starts[2] += self.jump_slice
            #plt.imsave("/media/nazib/E20A2DB70A2D899D/patches/test"+str(self.CmbCount)+".jpg",self.image[:,:,86])
            #print "Mean Intensity of patch {0} is {1}".format(self.CmbCount,np.mean(patch))
            self.CmbCount+=1
            if all_starts[2]>self.imShape[2]:
                return self.image
            else:
                all_starts[2] += 1
             
        return self.image

    def extract_patches(self):

        rowstart = 0; colstart = 0; slstart = 0
        
        while rowstart < self.row_terminate:

            colstart = 0
            while colstart < self.col_terminate:
                if self.imShape[2] > self.patchShape[2]:
                    all_starts = [rowstart,colstart,0]
                    patches= self.depth_cut(all_starts)
                    colstart += self.jump_cols
                    if colstart>self.imShape[1]:
                        break
                    else:
                        colstart += 1
                else:
                    region = (slice(rowstart, rowstart + self.patchShape[0]),
                            slice(colstart, colstart + self.patchShape[1]))
                    
                    patch = self.image[region]
                    self.patches.append(patch)
                    colstart += self.jump_cols  
                    self.ExCount+=1

                    if colstart>self.imShape[1]:
                        break
                    else:
                        colstart += 1
                ## Termination of Column loop ###
            rowstart += self.jump_rows
            if rowstart>self.imShape[0]:
                break
            else:
                rowstart += 1

        self.patches=self.list2array(self.patches)
        return self.patches

    def combine_patches(self, option='image'):

        rowstart = 0; colstart = 0; slstart = 0
        if self.patches.shape[0]== 0:
            print ("Patches are not initialized")
            return

        while rowstart < self.row_terminate:
        
            colstart = 0
            while colstart < self.col_terminate:
                
                if self.imShape[2] > self.patchShape[2]:
                    all_starts = [rowstart,colstart,0]
                    self.image= self.depth_join(all_starts,"image")
                    colstart += self.jump_cols
                    if colstart>self.imShape[1]:
                        break
                    else:
                        colstart += 1
                else:
                    region = (slice(rowstart, rowstart + self.patchShape[0]),
                            slice(colstart, colstart + self.patchShape[1]))
                    
                    # The actual pixels in that region.
                    patch = self.patches[self.CmbCount,:,:,:]
                    self.image[region] = patch
                    colstart += self.jump_cols
                    self.CmbCount+=1
                    
                    if colstart>self.imShape[1]:
                        break
                    else:
                        colstart += 1
                ## Termination of Column loop ###
            rowstart += self.jump_rows
            if rowstart>self.imShape[0]:
                break
            else:
                # Otherwise, shift the window down by one pixel.
                rowstart += 1

        return self.image
    
    def list2array(self,patch_list):
        patch_arr = np.concatenate([pat[np.newaxis, ...] for pat in patch_list], axis=0)
        return patch_arr


def normalize_intensity(brain,range):
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

def patition_img(src_img,patches,ita):

    r,c,h=src_img.shape
    r=float(r)
    c=float(c)
    h=float(h)
    
    ##### Overlape calculation in 1st dimension ####
    r_fold = r/patches
    r_mod = np.mod(r, patches)
    r_fold = r_fold + ita
    r_ovlap = math.ceil((r_fold*patches - r)/((r_fold)-1))
    
    # patch number and overlap in second dimension
    c_fold = c/patches
    c_mod = np.mod(c, patches)
    c_fold = c_fold + ita

    c_ovlap = math.ceil(((c_fold)*patches - c)/((c_fold)-1))
    
    # patch number and overlap in third dimension
    h_fold = h/patches
    h_mod = np.mod(h, patches)
    h_fold = h_fold + ita
    h_ovlap = math.ceil(((h_fold)*patches - h)/((h_fold)-1))

    patch_list=[]

    p_count = 0
    
    for R in range(1,np.int(r_fold)):
        r_s = np.int((R - 1)*patches + 1 - (R - 1)*r_ovlap)
        r_e = np.int(r_s + patches - 1)
        for C in range(1,np.int(c_fold)):
            c_s = np.int((C - 1)*patches + 1 - (C - 1)*c_ovlap)
            c_e = np.int(c_s + patches - 1)
            #for H in range(1,np.int(h_fold)):
            #    h_s = np.int((H - 1)*patches + 1 - (H - 1)*h_ovlap)
            #    h_e = np.int(h_s + patches - 1)
                # partition
            cube=src_img[r_s:r_e, c_s:c_e,:]
            print (str(p_count)+' cube shape: ',cube.shape)
                #if np.mean(cube) > 0.05:
            patch_list.append(cube)
            #cv2.imwrite("/home/n9614885/patchvm_data/slices/"+str(p_count)+".jpg",cube[:,:,15]*255.0)
            '''
                if cube.shape != (64,64,64) or np.mean(cube) <= 0.03:
                    print('Prebolem in '+str(p_count))
                else:
                    print (str(p_count)+' cube shape: ',cube.shape)
                    patch_list.append(cube)
            '''
            p_count = p_count + 1

    return patch_list

def patches2Img(patch_list,imageSize, patches, ita):

    fusion_Img = np.zeros((imageSize))
    
    r=float(imageSize[0])
    c=float(imageSize[1])
    h=float(imageSize[2])
    
    ##### Overlape calculation in 1st dimension ####
    r_fold = r/patches
    r_mod = np.mod(r, patches)
    r_fold = r_fold + ita
    r_ovlap = math.ceil((r_fold*patches - r)/((r_fold)-1))
    
    # patch number and overlap in second dimension
    c_fold = c/patches
    c_mod = np.mod(c, patches)
    c_fold = c_fold + ita

    c_ovlap = math.ceil(((c_fold)*patches - c)/((c_fold)-1))
    
    # patch number and overlap in third dimension
    h_fold = h/patches
    h_mod = np.mod(h, patches)
    h_fold = h_fold + ita
    h_ovlap = math.ceil(((h_fold)*patches - h)/((h_fold)-1))

    p_count = 0
    for R in range(1,np.int(r_fold)):
        r_s = np.int((R - 1)*patches + 1 - (R - 1)*r_ovlap)
        r_e = np.int(r_s + patches - 1)
        for C in range(1,np.int(c_fold)):
            c_s = np.int((C - 1)*patches + 1 - (C - 1)*c_ovlap)
            c_e = np.int(c_s + patches - 1)
            #for H in range(1,np.int(h_fold)):
               # h_s = np.int((H - 1)*patches + 1 - (H - 1)*h_ovlap)
               # h_e = np.int(h_s + patches - 1)
                # partition
            if(p_count<len(patch_list)):
                cube = patch_list[p_count]
            fusion_Img[r_s:r_e, c_s:c_e,:]=cube#patch_list[p_count]
            #print("Patch: ",str(p_count))
            #else:
            #    break
            #print (str(p_count)+' cube shape: ',cube.shape)
            p_count = p_count + 1
    return fusion_Img

def create_atlaspatches(atlas_dir,data_file):
    atlas = nib.load(atlas_dir)
    atlas_vol = atlas.get_data()
    
    data_file=data_file+"/MRI_atlas_patches.h5"
    atlas_file=h5py.File(data_file,"w")
    atlas_patches=patition_img(atlas_vol,65,0)
    atlas_file.create_dataset('atlas',data=atlas_patches)
    print ("Saved Atlas Patches")

def set_mid_img(img,cSize=(512,540,169),dSize=(640,540,169)):
    
    new_img = np.zeros(dSize,dtype=np.float)
    #cropped = img[0:cSize[0],0:cSize[1],0:cSize[2]]
    cropped = crop_mid_img(img,cSize)
    cx = math.ceil(np.float(dSize[0]/2.0))
    cy = math.ceil(np.float(dSize[1]/2.0))
    cz = math.ceil(np.float(dSize[2]/2.0))
    
    mx = math.ceil(np.float(cSize[0]/2.0))
    my = math.ceil(np.float(cSize[1]/2.0))
    mz = math.ceil(np.float(cSize[2]/2.0))

    sx = np.int(cx - mx)
    sy = np.int(cy - my)
    sz = np.int(cz - mz)

    ex = np.int(cx + mx)
    ey = np.int(cy + my)
    ez = np.int(cz + mz)
    
    if ez-sz == cropped.shape[2]:
        new_img[sx:ex,sy:ey,sz:ez] = cropped
    else:
        ez = cropped.shape[2]+sz
        new_img[sx:ex,sy:ey,sz:ez] = cropped

    return new_img

def crop_mid_img(img,cSize=(512,540,169)):
    
    imSize = img.shape

    cx = math.ceil(np.float(imSize[0]/2.0))
    cy = math.ceil(np.float(imSize[1]/2.0))
    if imSize[2]%2==0:
       cz = math.ceil(np.float(imSize[2]/2.0))
    else:
       cz = math.floor(np.float(imSize[2]/2.0))
    
    mx = math.ceil(np.float(cSize[0]/2.0))
    my = math.ceil(np.float(cSize[1]/2.0))
    if cSize[2]%2==0:
       mz = math.ceil(np.float(cSize[2]/2.0))
    else:
       mz = math.floor(np.float(cSize[2]/2.0))

    sx = np.int(cx - mx)
    sy = np.int(cy - my)
    sz = np.int(cz - mz)

    ex = np.int(cx + mx)
    ey = np.int(cy + my)
    ez = np.int(cz + mz)
    
    cropped = img[sx:ex,sy:ey,sz:ez]

    return cropped

def list2array(patch_list):
    patch_arr = np.concatenate([pat[np.newaxis, ...] for pat in patch_list], axis=0)
    return patch_arr


def CreateCannyImage(image,lower,upper):

    image_sitk = sitk.GetImageFromArray(image)
    image_sitk = sitk.Cast(image_sitk, sitk.sitkFloat32)

    image_edges = sitk.CannyEdgeDetection(image_sitk, lowerThreshold=lower,
                                        upperThreshold=upper)

    src_edges = sitk.GetArrayFromImage(image_edges)

    return src_edges


def extract_patches_tf(img,patchSize,overlap):
    img = np.reshape(img, (1, img.shape[0], img.shape[1], img.shape[2], 1))
    ksize = [1, patchSize, patchSize, patchSize, 1]
    strides = [1, overlap, overlap, overlap, 1]
    x = tf.placeholder(tf.float32, shape=(1,img.shape[1], img.shape[2], img.shape[3], 1), name="image")
    patches = tf.extract_volume_patches(x, ksize, strides, 'SAME')
    #patches = tf.reshape(patches, shape=(2040, 64, 64, 64))

    with tf.Session() as sess:
        all_patches = sess.run(patches, feed_dict={x:img})

    rows =all_patches.shape[1]*all_patches.shape[2]*all_patches.shape[3]
    all_patches = np.reshape(all_patches,(rows,patchSize,patchSize,patchSize))


    return  all_patches


if __name__=="__main__":
    
    #data_dir='/home/n9614885/myvoxelmorph/data/vols/15%/test/'
    #atlas_dir = '/home/n9614885/patchvm_data/atlas/003_nuclear.nii.gz'
    
    data_dir='/home/n9614885/CUBIC_Data/nii_merged/100%_registration/100%_data/DV/'
    #data_dir='/home/n9614885/myvoxelmorph/data/vols/15%/test/'
    #atlas_dir = '/home/n9614885/25%/test/003_nuclear_n.nii.gz'
    
    
    files = glob.glob(data_dir+"*.nii.gz")

    output_dir=data_dir

    for i in range(0,len(files)):
        data =[]
        moving = nib.load(files[i])
        moving_vol= moving.get_data()
        moving_vol = normalize_intensity(moving_vol,[0.0,1.0])

        ### For 25% resolution ####
        #padded =set_mid_img(moving_vol,(540,540,169),(576,576,192))
        padded =set_mid_img(moving_vol, moving_vol.shape,(2560,2176,704))
        #mov_patches = extract_patches_tf(padded,64,32)
        #patches = h5py.File(data_dir+"/0.5overlap/moving_2.h5")['moving']
        patchEx = PatchUtility((64, 64, 64), (2560,2176,704), 0.0, patches=[],image=padded)
        mov_patches = patchEx.extract_patches()
        fused = patchEx.combine_patches()
        #mov_patches = list2array(extract_patches(padded,(64,64,64),0.0))
        print("Number of patches : "+str(len(mov_patches)))


        data = np.asarray(data)
        df = pd.DataFrame(data)
        df.to_csv(files[i] + ".csv")
        #fused = combine_patches(mov_patches, (576, 576, 192), 0.5)
        #sl = fused[:,:,86]*255.0
        #cv2.imwrite('/home/n9614885/patchvm/fused_slice.jpg',sl)

        '''
        ### For 10% Resolution ###
        patchEx = PatchUtility((64,64,32),(256,256,96),0.5,patches=[],image=moving_vol)
        mov_patches = patchEx.extract_patches()
        print("Number of patches : "+str(len(mov_patches)))
        #mov_patches =list2array(mov_patches)
        fused = patchEx.combine_patches()
        #mov_patches = patition_img(moving_vol,65,32)
        #fused = patches2Img(mov_patches,(256,256,32),65,32)
        '''
        img=nib.Nifti1Image(fused, moving.affine)
        nib.save(img, str(i)+".nii.gz")

        data_file=output_dir+"moving_"+str(i)+".h5"
        hf = h5py.File(data_file,"w")
        hf.create_dataset('moving', data=mov_patches)
        hf.close()

        #print("Created Moving Patches : "+data_file+" Number of patches :"+str(len(mov_patches))+"\n")
        
print ("Done!!!!")


    
    

    




    
        
    





