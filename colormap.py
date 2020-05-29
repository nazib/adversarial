import numpy as np
import nibabel as nib
from scipy.misc import imsave, imresize
from matplotlib.colors import LinearSegmentedColormap
import sys
import math
from PIL import Image, ImageFont, ImageDraw
#from skimage.transform import resize
import cv2


def transpose(image):

    r,c,s = image.shape
    red = image[:,:,0].T
    green = image[:,:,1].T
    blue = image[:,:,2].T

    new_im = np.zeros((c,r,3),dtype=np.float)
    new_im[:,:,0] =red
    new_im[:,:,1] =green
    new_im[:,:,2] =blue
    return new_im


def crop_mid_img2D(img, cSize=(540, 540)):
    imSize = img.shape

    cx = math.ceil(np.float(imSize[0] / 2.0))
    cy = math.ceil(np.float(imSize[1] / 2.0))

    mx = math.ceil(np.float(cSize[0] / 2.0))
    my = math.ceil(np.float(cSize[1] / 2.0))


    sx = np.int(cx - mx)
    sy = np.int(cy - my)

    ex = np.int(cx + mx)
    ey = np.int(cy + my)

    cropped = img[sx:ex, sy:ey]
    #cropped = resize(cropped, (800, 800), anti_aliasing=True)
    #cropped = cv2.resize(cropped.astype(np.float64),(800,800))
    cropped = imresize(cropped, (800, 800), interp='bicubic').astype('float32')
    return cropped

def normalize_intensity2D(brain,range):

    m,n=brain.shape

    imMin=np.min(brain)
    imMax=np.max(brain)

    nMin=range[0]
    nMax=range[1]

    multi=((nMax-nMin)/np.float((imMax-imMin)))+nMin
    imgMin=np.zeros([m,n],dtype=float)
    imgMin[:,:]=imMin

    brain=(brain-imgMin)*multi

    return brain

def draw_text(image):
    
    image = Image.fromarray(np.uint8(image))
    draw = ImageDraw.Draw(image)
    import os
    font = ImageFont.truetype("/home/n9614885/bin/fonts/SUNA_Arial_Bold.ttf", 32)
    
    draw.text((230, 32),"A",(255,255,255),font=font)
    draw.text((90, 230),"B",(255,255,255),font=font)
    draw.text((725, 235),"C",(255,255,255),font=font)
    return np.asarray(image)
    
def draw_box(image,box):
    color = np.array([255,255,255], dtype=np.float)

    image[box[0]:box[0]+3,box[1]:box[3]]=color
    image[box[0]:box[2],box[1]:box[1]+3]=color
    image[box[2]:box[2]+3,box[1]:box[3]]=color
    image[box[0]:box[2],box[3]:box[3]+3]=color
    return image


def overlay_slices(slice1,slice2,config_obj,file_no):

    red = LinearSegmentedColormap.from_list("Black2Red",['black','red'])
    green = LinearSegmentedColormap.from_list("Black2Green",['black','green'])

    im1_s = green(slice1)
    im2_s = red(slice2)
    
    alpha = 0.4
    overlay1 = alpha * im2_s + (1 - alpha) * im1_s

    if config_obj.TestResolution == '25':
        overlay1 = crop_mid_img2D(transpose(overlay1))
        #overlay1 = transpose(overlay1)
    else:
        overlay1 = transpose(overlay1)
        overlay1 = imresize(overlay1, (800, 800), interp='bicubic').astype('float32')


    px =300
    py=300
    '''
    patch1 = overlay1[32:32+px,260:260+py]
    patch2 = overlay1[260:260+px,90:90+py]
    patch3 = overlay1[260:260+px,450:450+py]

    patch1_file_name = "00{0}_{1}_{2}{3}patch1.jpg".format(file_no, config_obj.TestSliceNo,
                                                               config_obj.TestModelNumber, config_obj.TestModelType)
    patch2_file_name = "00{0}_{1}_{2}{3}patch2.jpg".format(file_no, config_obj.TestSliceNo,
                                                               config_obj.TestModelNumber,
                                                               config_obj.TestModelType)
    patch3_file_name = "00{0}_{1}_{2}{3}patch3.jpg".format(file_no, config_obj.TestSliceNo,
                                                               config_obj.TestModelNumber,
                                                               config_obj.TestModelType)
    imsave(patch1_file_name, patch1)
    imsave(patch2_file_name, patch2)
    imsave(patch3_file_name, patch3)
    
    
    overlay1 = cv2.rectangle(overlay1,(32,260),(32+px,260+py),(255,255,255),3)
    overlay1 = cv2.rectangle(overlay1,(260,90),(260+px,90+py),(255,255,255),3)
    overlay1 = cv2.rectangle(overlay1,(450,260),(450+px,260+py),(255,255,255),3)
    overlay1 = draw_text(overlay1)
    
    overlay1 = cv2.rectangle(overlay1,(260,32),(260+px,32+py),(255,255,255),3)
    overlay1 = cv2.rectangle(overlay1,(90,260),(90+px,260+py),(255,255,255),3)
    overlay1 = cv2.rectangle(overlay1,(450,260),(450+px,260+py),(255,255,255),3)
    '''
    overlay1 = draw_box(overlay1,(32,260,332,560))
    overlay1 = draw_box(overlay1,(260,90,260+px,90+px))
    overlay1 = draw_box(overlay1,(260,450,260+px,450+py))
    overlay1 = draw_text(overlay1)

    
    overlay1_name = "OV_00{0}_{1}_{2}{3}.jpg".format(file_no,config_obj.TestSliceNo,config_obj.TestModelNumber,config_obj.TestModelType)
    imsave(overlay1_name, overlay1)
    diff = slice1 - slice2
    #diff = normalize_intensity2D(diff,[0,255])
    #diff  = normalize_intensity2D((1-np.abs(diff)),[0.0,1.0])
    diff = (1-np.abs(diff))

    if config_obj.TestResolution == '25':
        diff = crop_mid_img2D(diff.T)
        slice1 = crop_mid_img2D(slice1.T)
        imsave("brain_slice_{0}_{1}_{2}.jpg".format(file_no, config_obj.TestSliceNo, config_obj.TestModelNumber), slice1)
        #diff = diff.T
    else:
        diff = imresize(diff.T.astype(np.float64), (800, 800), interp='bicubic')

    overlay1_name = "DiffOV_00{0}_{1}_{2}{3}.jpg".format(file_no, config_obj.TestSliceNo, config_obj.TestModelNumber,
                                          config_obj.TestModelType)
    imsave(overlay1_name, diff)

    patch1 = diff[32:32 + px, 260:260 + py]
    patch2 = diff[260:260 + px, 90:90 + py]
    patch3 = diff[260:260 + px, 450:450 + py]

    patch1_file_name = "Diff00{0}_{1}_{2}{3}patch1.jpg".format(file_no, config_obj.TestSliceNo,
                                                               config_obj.TestModelNumber, config_obj.TestModelType)
    patch2_file_name = "Diff00{0}_{1}_{2}{3}patch2.jpg".format(file_no, config_obj.TestSliceNo,
                                                               config_obj.TestModelNumber,
                                                               config_obj.TestModelType)
    patch3_file_name = "Diff00{0}_{1}_{2}{3}patch3.jpg".format(file_no, config_obj.TestSliceNo,
                                                               config_obj.TestModelNumber,
                                                               config_obj.TestModelType)

    imsave(patch1_file_name,patch1)
    imsave(patch2_file_name,patch2)
    imsave(patch3_file_name,patch3)
    print ("Overlay saved")
    return 0

'''
if __name__=="__main__":

    if len(sys.argv)!=3:
        print ("Slice or iteration is missing as argument\n")
    else:
        slice_no = sys.argv[1]
        iter_no =sys.argv[2]

        im1 = nib.load("/home/n9614885/CUBIC_Data/nii_merged/25%_registration/25%_data/001_nuclear.nii.gz").get_data()
        #im3 = nib.load("001_registered.nii.gz").get_data()
        #im2 = nib.load("/home/n9614885/patchvm_data/25%/test/002_nuclear.nii.gz").get_data()
        #im3 = nib.load("/home/n9614885/CUBIC_Data/nii_merged/25%_registration/25%_data/003_nuclear.nii.gz").get_data()

        from partition import normalize_intensity
        im1 = normalize_intensity(im1,[0.0,1.0])
        im1_s =normalize_intensity2D(im1[:,:,86],[0.0,1.0])
        
        #im2_s =normalize_intensity2D(im2[:,:,86],[0.0,1.0])
        #im3_s =normalize_intensity2D(im3[:,:,86],[0.0,1.0])


        #color_red =[(0, 0, 0), (0.1, 0, 0), (0.1, 0, 0.1)]
        #color_green =[(0, 0, 0), (0, 0.1, 0), (0, 0.1, 0.1)]
        
        color_red = [(0, 0, 0), (0.01, 0.0, 0.0), (0.01, 0.0, 0.0)]
        color_green = [(0, 0, 0),(0.0, 0.01, 0.0),(0.0, 0.01, 0.0)]
        red = LinearSegmentedColormap.from_list("Black2Red", ['black','red'])
        green = LinearSegmentedColormap.from_list("Black2Green",['black','green'])

        im1_s = green(im1_s)
        #im2_s = green(im2_s)
        #im3_s = red(im3_s)
      
        alpha = 0.4
        #overlay1 = alpha*im3_s+(1-alpha)*im1_s
        #overlay2 = alpha * im3_s + (1 - alpha) * im2_s
        
        

        overlay1_name ="001_{0}_{1}DDN.jpg".format(slice_no,iter_no)
        overlay2_name = "002_{0}_{1}DDN.jpg".format(slice_no, iter_no)
        overlay1 = im1_s
        overlay1 = crop_mid_img2D(transpose(overlay1))
        #overlay2 = crop_mid_img2D(transpose(overlay2))
        
        px =300
        py=300
        patch1 = overlay1[32:32+px,260:260+py]
        patch2 = overlay1[260:260+px,90:90+py]
        patch3 = overlay1[260:260+px,450:450+py]
        
        
        overlay1 = cv2.rectangle(overlay1,(32,260),(32+px,260+py),(255.0,0.0,0.0),3)
        overlay1 = cv2.rectangle(overlay1,(260,90),(260+px,90+py),(0.0,255,0.0),3)
        overlay1 = cv2.rectangle(overlay1,(450,260),(450+px,260+py),(0.0,0.0,255.0),3)
        
        overlay1 = draw_box(overlay1,(32,260,332,560))
    	overlay1 = draw_box(overlay1,(260,90,260+px,90+px))
    	overlay1 = draw_box(overlay1,(260,450,260+px,450+py))
    	overlay1 = draw_text(overlay1)
        imsave("test.png",overlay1)
        
        imsave("patch1.png",patch1)
        imsave("patch2.png",patch2)
        imsave("patch3.png",patch3)
        imsave(overlay1_name,overlay1)
        imsave(overlay2_name,overlay2)

        diff = im1[:,:,86] - im3[:,:,86]
        diff = (1 - np.abs(diff))
        # diff = normalize_intensity2D(diff,[0,255])
        diff = normalize_intensity2D(diff, [0, 255.0])
        diff = crop_mid_img2D(diff.T)

        overlay1_name = "001_beforeAffine_vs_afterDeepnet2.jpg"
        imsave(overlay1_name, diff)
'''


