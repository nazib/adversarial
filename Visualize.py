# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 10:08:36 2018
@author: Dongyang
This script contains some utilize functions for data visualization
"""
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import nibabel as nib

#==============================================================================
# Define a custom colormap for visualiza Jacobian
#==============================================================================
class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

#==============================================================================
# Iterating Each Slice
# Modified from 
# datacamp: https://www.datacamp.com/community/tutorials/matplotlib-3d-volumetric-data
#==============================================================================
def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)

def multi_slice_viewer(volume, axis = 0, cmap = 'gray', Jac = False):
    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots()

    if axis == 1:
         ax.volume = np.moveaxis(volume, 0, -1)
    elif axis == 2:
         ax.volume = np.moveaxis(volume, [0, 1], [-1, -2])
    else:
         ax.volume = volume
    
    ax.index = volume.shape[0] // 2
    if Jac:
        ax.imshow(ax.volume[ax.index], cmap, norm= MidpointNormalize(midpoint=1))
    else:
        ax.imshow(ax.volume[ax.index], cmap)
        
    fig.canvas.mpl_connect('key_press_event', process_key) # use lambda to pass extra arguments


def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
    fig.canvas.draw()

def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])
    print(ax.index)

def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])
    print(ax.index)  # could create a slider for it


#==============================================================================
#  Overlay two images contained in numpy arrays  
#==============================================================================
def overlay(img1, img2, cmap1=None, cmap2=None, alpha=0.4, Jac = False):
#    plt.figure()
    plt.imshow(img1, cmap=cmap1)
    if Jac:
        plt.imshow(img2, cmap=cmap2, norm=MidpointNormalize(midpoint=1),alpha=alpha)
    plt.imshow(img2, cmap=cmap2, alpha=alpha)
    plt.axis('off')


#==============================================================================
# plot an array of images for comparison
#==============================================================================    
def show_sample_slices(sample_list,name_list, Jac = False, cmap = 'gray', attentionlist=None):
  num = len(sample_list)
  fig, ax = plt.subplots(1,num)
  for i in range(num):
    if Jac:
        ax[i].imshow(sample_list[i], cmap, norm=MidpointNormalize(midpoint=1))
    else:
        ax[i].imshow(sample_list[i], cmap)
    ax[i].set_title(name_list[i])
    ax[i].axis('off')
    if attentionlist:
        ax[i].add_artist(attentionlist[i])
  plt.subplots_adjust(wspace=0)
  
def my_cmap(name='mycmap', colors=[(0, 0, 0), (0, 1, 0), (1, 0, 0)]):
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list(name, colors, N=len(colors))
    return cm

#def examine_ROI(A, B):
#    '''
#    A, B are masks of ROI to be examined
#    '''
#    plt.figure()
#    plt.imshow(A, cmap=my_cmap(color))
    

'''
Grid Visualization
'''
#==============================================================================
# 2d grid visualization
# disp: displacement field
# res: ratio of the output size. eg. 0.5 means visualizing with a coarser grid with half the size along each direction
# direct: which direction to show. 0: x, 1:y, 2: both
#==============================================================================
from skimage.transform import resize
def vis_grid(disp, res = 1, direct = 2): # xy is of shape h*w*2
     
     w, h= np.shape(disp)[0], np.shape(disp)[1]
     
     x = np.linspace(-1., 1., w)
     y = np.linspace(-1., 1., h)
     
     xx , yy = np.meshgrid(x, y)
     
     xy = np.stack([xx,yy], 2) + disp
     
     if res != 1:
          w = np.floor(w*res).astype(int)
          h = np.floor(h*res).astype(int)
     
          dispx = resize(xy[...,0], (w,h), mode='constant', clip = False, order = 1)
          dispy = resize(xy[...,1], (w,h), mode='constant', clip = False, order = 1)
         
          xy = np.stack([dispx, dispy], 2)
     
     plt.figure()
     
     if direct == 0: #Only plot the x-direction
          for row in range(w):
               x, y = xy[row,:, 0], yy[row,:]       
               plt.plot(x,y, color = 'b')
#               plt.ylim(1,-1)
          for col in range(h):
               x, y = xy[:, col, 0], yy[:, col]       
               plt.plot(x,y, color = 'b') 
               plt.ylim(1,-1)
               plt.axis('equal')
     
     elif direct == 1: #Only plot the y-direction 
          for row in range(w):
               x, y = xx[row,:], xy[row,:, 1]       
               plt.plot(x,y, color = 'b')
#               plt.ylim(1,-1)
          for col in range(h):
               x, y = xx[:, col], xy[:, col, 1]       
               plt.plot(x,y, color = 'b') 
               plt.ylim(1,-1)
               plt.axis('equal')
     else:
          for row in range(w):
               x, y = xy[row,:, 0], xy[row,:, 1]       
               plt.plot(x,y, color = 'b')
          for col in range(h):
               x, y = xy[:, col, 0], xy[:, col, 1]       
               plt.plot(x,y, color = 'b') 
               plt.ylim(1,-1)
               plt.axis('equal')

#==============================================================================
# 3d grid visualization
#==============================================================================               
def vis_grid_3d(disp, res = 1):
     
     w, h, d= np.shape(disp)[0], np.shape(disp)[1], np.shape(disp)[2]
         
     x = np.linspace(-1., 1., w)
     y = np.linspace(-1., 1., h)
     z = np.linspace(-1., 1., d)
     
     xx, yy, zz = np.meshgrid(x, y, z)
     
     xyz = np.stack([xx, yy, zz], 3) + disp
     
     if res != 1:
          w = np.floor(w*res).astype(int)
          h = np.floor(h*res).astype(int)
          d = np.floor(d*res).astype(int)
          
          dispx = resize(xyz[...,0], (w,h,d), mode='constant', clip = False, order = 3)
          dispy = resize(xyz[...,1], (w,h,d), mode='constant', clip = False, order = 3)
          dispz = resize(xyz[...,2], (w,h,d), mode='constant', clip = False, order = 3)
          
          xyz = np.stack([dispx, dispy, dispz], 3)
          
     fig = plt.figure()
     ax = fig.add_subplot(111, projection='3d')
     
     for row in range(w):
          for col in range(h):
               x, y, z = xyz[row, col, :, 0], xyz[row, col, :, 1], xyz[row, col, :, 2]
               ax.plot(x,y,z,color = 'b')
     
     for row in range(h):
          for col in range(d):
               x, y, z = xyz[:,row, col,  0], xyz[:,row, col, 1], xyz[:, row, col,  2]
               ax.plot(x,y,z,color = 'b')     

     for row in range(w):
          for col in range(d):
               x, y, z = xyz[row, :, col,  0], xyz[row, :,col,  1], xyz[row, :, col,  2]
               ax.plot(x,y,z,color = 'b')
    
#==============================================================================
# Generating random colors, copied from https://github.com/delestro/rand_cmap
#==============================================================================
def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=True):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """
    from matplotlib.colors import LinearSegmentedColormap
    import colorsys
    import numpy as np


    if type not in ('bright', 'soft'):
        print ('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print('Number of labels: ' + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Display colorbar
    if verbose:
        from matplotlib import colors, colorbar
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                                   boundaries=bounds, format='%1i', orientation=u'horizontal')

    return random_colormap

def Jac(x):

    height, width, depth, num_channel = x.shape
    num_voxel = (height-1)*(width-1)*(depth-1)

    dx = np.reshape(x[1:,:-1,:-1,:]-x[:-1,:-1,:-1,:], [num_voxel, num_channel])
    dy = np.reshape(x[:-1,1:,:-1,:]-x[:-1,:-1,:-1,:], [num_voxel, num_channel])
    dz = np.reshape(x[:-1,:-1,1:,:]-x[:-1,:-1,:-1,:], [num_voxel, num_channel])
    J = np.stack([dx, dy, dz], 2)
    return np.reshape(J, [height-1, width-1, depth-1, 3, 3])


def Jac_5(x):

    height, width, depth, num_channel = x.shape
    num_voxel = (height-4)*(width-4)*(depth-4)

    dx = np.reshape((x[:-4,2:-2,2:-2,:]-8*x[1:-3,2:-2,2:-2,:] + 8*x[3:-1,2:-2,2:-2,:] - x[4:,2:-2,2:-2,:])/12.0, [num_voxel, num_channel])
    dy = np.reshape((x[2:-2,:-4,2:-2,:]-8*x[2:-2,1:-3,2:-2,:] + 8*x[2:-2,3:-1,2:-2,:] - x[2:-2,4:,2:-2,:])/12.0, [num_voxel, num_channel])
    dz = np.reshape((x[2:-2,2:-2,:-4,:]-8*x[2:-2,2:-2,1:-3,:] + 8*x[2:-2,2:-2,3:-1,:] - x[2:-2,2:-2,4:,:])/12.0, [num_voxel, num_channel])
    J = np.stack([dx, dy, dz], 2)

    return np.reshape(J, [height-4, width-4, depth-4, 3, 3])


#==============================================================================

# Calculate the Determinent of Jacobian of the transformation

#==============================================================================

def Get_Ja(displacement):

    '''
    '''
  

    D_y = (displacement[:,1:,:-1,:-1,:] - displacement[:,:-1,:-1,:-1,:])

    D_x = (displacement[:,:-1,1:,:-1,:] - displacement[:,:-1,:-1,:-1,:])

    D_z = (displacement[:,:-1,:-1,1:,:] - displacement[:,:-1,:-1,:-1,:])

    

    D1 = (D_x[...,0]+1)*( (D_y[...,1]+1)*(D_z[...,2]+1) - D_z[...,1]*D_y[...,2])

    D2 = (D_x[...,1])*(D_y[...,0]*(D_z[...,2]+1) - D_y[...,2]*D_x[...,0])

    D3 = (D_x[...,2])*(D_y[...,0]*D_z[...,1] - (D_y[...,1]+1)*D_z[...,0])

    D = np.abs(D1-D2+D3)

    return D


if __name__=="__main__":
    
    x = nib.load("001_x.nii.gz")
    y = nib.load("001_y.nii.gz")
    z = nib.load("001_z.nii.gz")
    f_x = x.get_data()
    f_y = y.get_data()
    f_z = z.get_data()
    flow =np.zeros((f_x.shape[0],f_x.shape[1],f_x.shape[2],3),dtype=np.float)
    flow[:,:,:,0]=f_x
    flow[:,:,:,1]=f_y
    flow[:,:,:,2]=f_z

    j1 = Jac(flow)

    x = nib.load("002_x.nii.gz")
    y = nib.load("002_y.nii.gz")
    z = nib.load("002_z.nii.gz")
    f_x = x.get_data()
    f_y = y.get_data()
    f_z = z.get_data()
    flow =np.zeros((f_x.shape[0],f_x.shape[1],f_x.shape[2],3),dtype=np.float)
    flow[:,:,:,0]=f_x
    flow[:,:,:,1]=f_y
    flow[:,:,:,2]=f_z

    j2 = Jac(flow)

    slice_001 = j1[:,:,78,0]
    slice_002 = j2[:,:,78,0]

    labels = ['Brain 001','Brain 002']
    slices = [slice_001,slice_002]
    show_sample_slices(slices,labels,Jac=True, cmap = 'bwr_r')





