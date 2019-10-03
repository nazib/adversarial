import tensorflow as tf
import numpy as np
import keras
from keras import losses
from keras.losses import mse, binary_crossentropy
from keras import backend as K


def l13D():
    def loss(y_true, y_pred):

        l1=tf.abs((y_true-y_pred))

        return tf.reduce_mean(l1)+binary_crossentropy(K.flatten(y_true), K.flatten(y_pred))
    return loss


def l23D():
    def loss(y_true, y_pred):
        l2=mse(K.flatten(y_true), K.flatten(y_pred))
        #l2=binary_crossentropy(K.flatten(y_true), K.flatten(y_pred))
        return l2 
    return loss

def triplet():
    def loss(y_true,y_pred):
        #print(K.int_shape(y_true))
        l2=256.0*256.0*96.0*binary_crossentropy(K.flatten(y_true), K.flatten(y_pred))
        return l2
    return loss

def cycle_loss():
    def loss(y_true,y_pred):
        cyc = tf.reduce_mean(tf.abs(y_pred - y_true))
        return cyc
    return loss

def gradientLoss(penalty='l1'):
    def loss(y_true, y_pred):
        
        dy = tf.abs(y_pred[:, 1:, :, :, :] - y_pred[:, :-1, :, :, :])
        dx = tf.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dz = tf.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])

        if (penalty == 'l2'):
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz
        d = tf.reduce_mean(dx)+tf.reduce_mean(dy)+tf.reduce_mean(dz)
        
        #v=tf.reduce_mean(tf.abs(tf.square(y_pred)))
        return d/3.0

    return loss


def gradientLoss2D():
    def loss(y_true, y_pred):
        dy = tf.abs(y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :])
        dx = tf.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])

        dy = dy * dy
        dx = dx * dx

        d = tf.reduce_mean(dx)+tf.reduce_mean(dy)
        return d/2.0

    return loss


def cc3D(win=[9, 9, 9], voxel_weights=None):
    def loss(I, J):
        
        I2 = I*I
        J2 = J*J
        IJ = I*J

        filt = tf.ones([win[0], win[1], win[2], 1, 1])

        I_sum = tf.nn.conv3d(I, filt, [1, 1, 1, 1, 1], "SAME")
        J_sum = tf.nn.conv3d(J, filt, [1, 1, 1, 1, 1], "SAME")
        I2_sum = tf.nn.conv3d(I2, filt, [1, 1, 1, 1, 1], "SAME")
        J2_sum = tf.nn.conv3d(J2, filt, [1, 1, 1, 1, 1], "SAME")
        IJ_sum = tf.nn.conv3d(IJ, filt, [1, 1, 1, 1, 1], "SAME")

        win_size = win[0]*win[1]*win[2]
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross*cross / (I_var*J_var+1e-5)
        #ssim = tf.image.ssim(I, J, max_val=1.0)
        #FCC = tf.pow((1.0-ssim),0.5)*tf.reduce_mean(cc)
        # if(voxel_weights is not None):
        #	cc = cc * voxel_weights

        return -1.0*cc
        
    return loss


def cc2D(win=[9, 9]):
    def loss(I, J):
        I2 = tf.multiply(I, I)
        J2 = tf.multiply(J, J)
        IJ = tf.multiply(I, J)

        sum_filter = tf.ones([win[0], win[1], 1, 1])

        I_sum = tf.nn.conv2d(I, sum_filter, [1, 1, 1, 1], "SAME")
        J_sum = tf.nn.conv2d(J, sum_filter, [1, 1, 1, 1], "SAME")
        I2_sum = tf.nn.conv2d(I2, sum_filter, [1, 1, 1, 1], "SAME")
        J2_sum = tf.nn.conv2d(J2, sum_filter, [1, 1, 1, 1], "SAME")
        IJ_sum = tf.nn.conv2d(IJ, sum_filter, [1, 1, 1, 1], "SAME")

        win_size = win[0]*win[1]

        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross*cross / (I_var*J_var + np.finfo(float).eps)
        return -1.0*tf.reduce_mean(cc)
    return loss

def NMI(bin):
    def loss(y_true, y_pred):
        #x=tf.contrib.gan.losses.wargs.mutual_information_penalty(y_true,y_pred)
        value_range=[0,1]
        [H, xedges, yedges] = tf.py_func(histogram_2d, [y_true, y_pred,bin], [tf.float32,tf.float32, tf.float32])
        pxy=H/tf.reduce_sum(H)
        px=tf.reduce_sum(pxy,axis=1)
        py=tf.reduce_sum(pxy,axis=0)

        Hx=-tf.reduce_sum(px*tf.log(px))
    	Hy=-tf.reduce_sum(py*tf.log(py))
        
        Hx=tf.cond(tf.is_nan(Hx), lambda:tf.assign(Hx,1.0),lambda:Hx)
        Hx=tf.cond(tf.is_inf(Hx), lambda:tf.assign(Hx,1.0),lambda:Hx)

        Hy=tf.cond(tf.is_nan(Hy), lambda:tf.assign(Hy,1.0),lambda:Hy)
        Hy=tf.cond(tf.is_nan(Hy), lambda:tf.assign(Hy,1.0),lambda:Hy)
        
        p_x_y=tf.matmul(px[:,None],py[None,:])
        
        zero = tf.constant(0, dtype=tf.float32)
        idx=tf.where(tf.not_equal(pxy,zero))
        pxy=tf.gather_nd(pxy,idx)
        p_x_y=tf.gather_nd(p_x_y,idx)
        P=tf.pow((1-pxy),0.5)
        nmi=tf.reduce_sum(P*tf.log(pxy/p_x_y))
        nmi=nmi/tf.sqrt(Hx*Hy)
        return -1.0*nmi
    
    def histogram_2d(a,b,bin):
        ar = a.reshape(-1)
        br = b.reshape(-1)
        aux = np.histogram2d(ar, br,bins=bin)
        return aux[0].astype(np.float32), aux[1].astype(np.float32), aux[2].astype(np.float32)

    return loss

def KL_loss():
    def loss(y_true,y_pred):

        z_mean=y_pred[:,0:2]
        z_log_var=y_pred[:,2:4]
        #reconstruction_loss *= image_size * image_size
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        kl_loss = K.mean(kl_loss)
        return kl_loss
    
    return loss

def fake_loss():
    def loss(y_true,y_pred):
        zero=tf.Variable(tf.zeros([1]), name="fake")
        return zero
    return loss

def MI(GRIDS = 20):
    def loss(I,J):
        shape = I.get_shape().as_list()
        print(shape)
        dim = 64*64*64
        I =tf.reshape(I, [-1, dim])
        J =tf.reshape(J, [-1, dim])

        ex = entropy1d(I)
        ey = entropy1d(J)
        exy = entropy2d(I, J)
        return ex + ey - exy
    return loss

def SSIM():
    def loss(y_true,y_pred):
        ssim = tf.image.ssim(y_true, y_pred, max_val=1.0)

        return -1.0*ssim
    return loss

GRIDS = 20
def core_function1d(x, y, grids = GRIDS):
    return tf.maximum((1/(grids - 1)) - tf.abs(tf.subtract(x, y)), 0)

def core_function2d(x1, x2, y1, y2, grids1 = GRIDS, grids2 = GRIDS):
    return core_function1d(x1, y1, grids1) + core_function1d(x2, y2, grids1)

def entropy1d(x, grids = GRIDS):
    shape1 = [x.get_shape().as_list()[0], 1, x.get_shape().as_list()[1]]
    shape2 = [1, grids, 1]

    gx = tf.linspace(0.0, 1.0, grids)

    X = tf.reshape(x, shape1)
    GX = tf.reshape(gx, shape2)

    mapping = core_function1d(GX, X, grids)
    mapping = tf.reduce_sum(mapping, 0)
    mapping = tf.add(mapping, 1e-10)
    mapping_normalized = tf.divide(mapping, tf.reduce_sum(mapping, 0, keepdims = True))

    entropy = tf.negative(tf.reduce_sum(tf.reduce_sum(tf.multiply(mapping_normalized, tf.log(mapping_normalized * grids)), 0)))

    return entropy

def entropy2d(x, y, gridsx = GRIDS, gridsy = GRIDS):
    batch_size = x.get_shape().as_list()[0]
    x_szie = x.get_shape().as_list()[1]
    y_size = y.get_shape().as_list()[1]

    gx = tf.linspace(0.0, 1.0, gridsx)
    gy = tf.linspace(0.0, 1.0, gridsy)

    X = tf.reshape(x, [batch_size, 1, 1, x_szie, 1])
    Y = tf.reshape(y, [batch_size, 1, 1, 1, y_size])

    GX = tf.reshape(gx, [1, gridsx, 1, 1, 1])
    GY = tf.reshape(gy, [1, 1, gridsy, 1, 1])

    mapping = core_function2d(GX, GY, X, Y, gridsx, gridsy)
    mapping = tf.reduce_sum(mapping, 0)
    mapping = tf.add(mapping, 1e-10)
    mapping_normalized = tf.divide(mapping, tf.reduce_sum(mapping, [0, 1], keepdims = True))

    entropy = tf.negative(tf.reduce_sum(tf.reduce_sum(tf.multiply(mapping_normalized, tf.log(mapping_normalized * (gridsx *gridsy))), [0, 1])))

    return entropy


