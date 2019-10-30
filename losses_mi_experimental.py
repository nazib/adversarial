"""
losses for VoxelMorph
"""


# Third party inports
import tensorflow as tf
import keras.backend as K
import numpy as np
#import networks
from keras.losses import mean_squared_error
import pickle



def normalize_percentile(features, percentile, feature_stats, layer=0, twod=False):
    pcs = feature_stats[layer][percentile]
    if not twod:
        features = features / pcs[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
    else:
        features = features / pcs[np.newaxis, np.newaxis, np.newaxis, :]
    return tf.clip_by_value(features, 0, 1)

def mutualInformation(bin_centers,
                      sigma_ratio=0.5,    # sigma for soft MI. If not provided, it will be half of a bin length
                      max_clip=1,
                      crop_background=False, # crop_background should never be true if local_mi is True
                      local_mi=False,
                      patch_size=1):
    if local_mi:
        return localMutualInformation(bin_centers, sigma_ratio, max_clip, patch_size)
    else:
        return globalMutualInformation(bin_centers, sigma_ratio, max_clip, crop_background)

def globalMutualInformation(bin_centers,
                      sigma_ratio=0.5,
                      max_clip=1,
                      crop_background=False):
    """
    Mutual Information for image-image pairs

    This function assumes that y_true and y_pred are both (batch_sizexheightxwidthxdepthxchan)
        
    """

    """ prepare MI. """
    vol_bin_centers = K.variable(bin_centers)
    num_bins = len(bin_centers)
    sigma = np.mean(np.diff(bin_centers))*sigma_ratio

    preterm = K.variable(1 / (2 * np.square(sigma)))

    def mi(y_true, y_pred):
        """ soft mutual info """
        y_pred = K.clip(y_pred, 0, max_clip)
        y_true = K.clip(y_true, 0, max_clip)

        if crop_background:
            # does not support variable batch size
            thresh = 0.0001
            padding_size = 20
            filt = tf.ones([padding_size, padding_size, padding_size, 1, 1])

            smooth = tf.nn.conv3d(y_true, filt, [1, 1, 1, 1, 1], "SAME")
            mask = smooth > thresh
            # mask = K.any(K.stack([y_true > thresh, y_pred > thresh], axis=0), axis=0)
            y_pred = tf.boolean_mask(y_pred, mask)
            y_true = tf.boolean_mask(y_true, mask)
            y_pred = K.expand_dims(K.expand_dims(y_pred, 0), 2)
            y_true = K.expand_dims(K.expand_dims(y_true, 0), 2)

        else:
            # reshape: flatten images into shape (batch_size, heightxwidthxdepthxchan, 1)
            y_true = K.reshape(y_true, (-1, K.prod(K.shape(y_true)[1:])))
            y_true = K.expand_dims(y_true, 2)
            y_pred = K.reshape(y_pred, (-1, K.prod(K.shape(y_pred)[1:])))
            y_pred = K.expand_dims(y_pred, 2)
        
        nb_voxels = tf.cast(K.shape(y_pred)[1], tf.float32)

        # reshape bin centers to be (1, 1, B)
        o = [1, 1, np.prod(vol_bin_centers.get_shape().as_list())]
        vbc = K.reshape(vol_bin_centers, o)
        
        # compute image terms
        I_a = K.exp(- preterm * K.square(y_true  - vbc))
        I_a /= K.sum(I_a, -1, keepdims=True)

        I_b = K.exp(- preterm * K.square(y_pred  - vbc))
        I_b /= K.sum(I_b, -1, keepdims=True)

        # compute probabilities
        I_a_permute = K.permute_dimensions(I_a, (0,2,1))
        pab = K.batch_dot(I_a_permute, I_b)  # should be the right size now, nb_labels x nb_bins
        pab /= nb_voxels
        pa = tf.reduce_mean(I_a, 1, keep_dims=True)
        pb = tf.reduce_mean(I_b, 1, keep_dims=True)
        
        papb = K.batch_dot(K.permute_dimensions(pa, (0,2,1)), pb) + K.epsilon()
        mi = K.sum(K.sum(pab * K.log(pab/papb + K.epsilon()), 1), 1)

        return mi

    def loss(y_true, y_pred):
        return -mi(y_true, y_pred)

    return loss

def localMutualInformation(bin_centers,
                      vol_size,
                      sigma_ratio=0.5,
                      max_clip=1,
                      patch_size=1):
    """
    Mutual Information for image-image pairs
    # vol_size = (160, 192, 224)  

    This function assumes that y_true and y_pred are both (batch_sizexheightxwidthxdepthxchan)
        
    """

    """ prepare MI. """
    vol_bin_centers = K.variable(bin_centers)
    num_bins = len(bin_centers)
    sigma = np.mean(np.diff(bin_centers))*sigma_ratio

    preterm = K.variable(1 / (2 * np.square(sigma)))

    def local_mi(y_true, y_pred):
        y_pred = K.clip(y_pred, 0, max_clip)
        y_true = K.clip(y_true, 0, max_clip)

        # reshape bin centers to be (1, 1, B)
        o = [1, 1, 1, 1, num_bins]
        vbc = K.reshape(vol_bin_centers, o)
        
        # compute padding sizes
        x, y, z = vol_size
        x_r = -x % patch_size
        y_r = -y % patch_size
        z_r = -z % patch_size
        pad_dims = [[0,0]]
        pad_dims.append([x_r//2, x_r - x_r//2])
        pad_dims.append([y_r//2, y_r - y_r//2])
        pad_dims.append([z_r//2, z_r - z_r//2])
        pad_dims.append([0,0])
        padding = tf.constant(pad_dims)

        # compute image terms
        # num channels of y_true and y_pred must be 1
        I_a = K.exp(- preterm * K.square(tf.pad(y_true, padding, 'CONSTANT')  - vbc))
        I_a /= K.sum(I_a, -1, keepdims=True)

        I_b = K.exp(- preterm * K.square(tf.pad(y_pred, padding, 'CONSTANT')  - vbc))
        I_b /= K.sum(I_b, -1, keepdims=True)

        I_a_patch = tf.reshape(I_a, [(x+x_r)//patch_size, patch_size, (y+y_r)//patch_size, patch_size, (z+z_r)//patch_size, patch_size, num_bins])
        I_a_patch = tf.transpose(I_a_patch, [0, 2, 4, 1, 3, 5, 6])
        I_a_patch = tf.reshape(I_a_patch, [-1, patch_size**3, num_bins])

        I_b_patch = tf.reshape(I_b, [(x+x_r)//patch_size, patch_size, (y+y_r)//patch_size, patch_size, (z+z_r)//patch_size, patch_size, num_bins])
        I_b_patch = tf.transpose(I_b_patch, [0, 2, 4, 1, 3, 5, 6])
        I_b_patch = tf.reshape(I_b_patch, [-1, patch_size**3, num_bins])

        # compute probabilities
        I_a_permute = K.permute_dimensions(I_a_patch, (0,2,1))
        pab = K.batch_dot(I_a_permute, I_b_patch)  # should be the right size now, nb_labels x nb_bins
        pab /= patch_size**3
        pa = tf.reduce_mean(I_a_patch, 1, keep_dims=True)
        pb = tf.reduce_mean(I_b_patch, 1, keep_dims=True)
        
        papb = K.batch_dot(K.permute_dimensions(pa, (0,2,1)), pb) + K.epsilon()
        mi = K.mean(K.sum(K.sum(pab * K.log(pab/papb + K.epsilon()), 1), 1))

        return mi

    def loss(y_true, y_pred):
        return -local_mi(y_true, y_pred)

    return loss

