import numpy as np
import nibabel as nib
import SimpleITK as sitk

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

'''
flow = sitk.GetImageFromArray(flow, isVector=True)			
flow.SetOrigin((0.0,0.0,0.0))
flow.SetSpacing((0.02580000076188278,0.02580000076188278,0.03999999910593033))
flow.SetDirection((1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0))
#flow= sitk.DisplacementFieldTransform(flow)

map_flow = sitk.DisplacementFieldJacobianDeterminant(flow,True)
'''
j = Jac(flow)

D = Get_Ja(j)

'''
map_flow = sitk.GetArrayFromImage(map_flow)
map_flow = normalize_intensity(map_flow,[0,1])
'''
map = nib.Nifti1Image(D[:,:,:,0],x.affine)
nib.save(map,"jacobian.nii.gz")

#print("Jecobian Shape :"+str(D))

