import tensorflow as tf
import numpy as np
import sys

def generator_loss(Dis_fake_src,Dis_fake_tgt):
    forward_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dis_fake_tgt,labels=tf.ones_like(Dis_fake_tgt)))
    inverse_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dis_fake_src,labels=tf.ones_like(Dis_fake_src)))
    loss = forward_loss  + inverse_loss
    return loss

def discriminator_loss(real_output, fake_output):

    real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_output,labels=tf.ones_like(real_output)))
    fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_output,labels=tf.zeros_like (fake_output)))
    total_loss = tf.reduce_mean(real_loss + fake_loss)
    return total_loss

def cc3D(I,J,win=[9, 9, 9]):

    I2 = I * I
    J2 = J * J
    IJ = I * J

    filt = tf.ones([win[0], win[1], win[2], 1, 1])
    I_sum = tf.nn.conv3d(I, filt, [1, 1, 1, 1, 1], "SAME")
    J_sum = tf.nn.conv3d(J, filt, [1, 1, 1, 1, 1], "SAME")
    I2_sum = tf.nn.conv3d(I2, filt, [1, 1, 1, 1, 1], "SAME")
    J2_sum = tf.nn.conv3d(J2, filt, [1, 1, 1, 1, 1], "SAME")
    IJ_sum = tf.nn.conv3d(IJ, filt, [1, 1, 1, 1, 1], "SAME")

    win_size = win[0] * win[1] * win[2]
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    cc = tf.reduce_mean(cross * cross / (I_var * J_var + 1e-5))

    return -1.0*cc

def Cyclic_loss(src_cyc,tgt_cyc, src,tgt, ssim_status):

    if ssim_status == 'off':
        src_cyclic = tf.reduce_mean(tf.abs(src-src_cyc))
        tgt_cyclic = tf.reduce_mean(tf.abs(tgt-tgt_cyc))
    else:
        src_cyclic = (1 - tf.reduce_mean(tf.image.ssim(src_cyc, src, max_val=1.0)[0])) + 0.00005 * tf.reduce_sum(tf.squared_difference(src_cyc, src))
        tgt_cyclic = (1 - tf.reduce_mean(tf.image.ssim(tgt_cyc, tgt, max_val=1.0)[0])) + 0.00005 * tf.reduce_sum(tf.squared_difference(tgt_cyc, tgt))

    return src_cyclic+tgt_cyclic

def Cyclic_lossl2(src_cyc,tgt_cyc, src,tgt):
    src_cyclic = tf.reduce_mean(tf.square(src-src_cyc))
    tgt_cyclic = tf.reduce_mean(tf.square(tgt-tgt_cyc))
    return src_cyclic+tgt_cyclic

def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.log(2. * np.pi)
    
    kl = tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar)),
      axis=raxis)
    kl = kl/(4*64*64*64*3)
    return kl

def simple_kl(mean,logvar):
    kl= 1 + logvar - tf.square(mean) - tf.exp(logvar)
    kl = -0.5 * tf.reduce_sum(kl, 1)
    kl = kl/(4*64*64*64*3)
    return kl

def flow_dist_loss(predict_flow, labelFlow):

    fx = labelFlow[:, :, :, :, 0]
    fy = labelFlow[:, :, :, :, 1]
    fz = labelFlow[:, :, :, :, 2]
    ix = labelFlow[:, :, :, :, 3]
    iy = labelFlow[:, :, :, :, 4]
    iz = labelFlow[:, :, :, :, 5]

    pfx = predict_flow[:, :, :, :, 0]
    pfy = predict_flow[:, :, :, :, 1]
    pfz = predict_flow[:, :, :, :, 2]
    pix = predict_flow[:, :, :, :, 3]
    piy = predict_flow[:, :, :, :, 4]
    piz = predict_flow[:, :, :, :, 5]

    lossx = tf.reduce_mean(tf.abs(fx - pfx))
    lossy = tf.reduce_mean(tf.abs(fy - pfy))
    lossz = tf.reduce_mean(tf.abs(fz - pfz))
    lossix = tf.reduce_mean(tf.abs(ix - pix))
    lossiy = tf.reduce_mean(tf.abs(iy - piy))
    lossiz = tf.reduce_mean(tf.abs(iz - piz))

    return tf.reduce_mean(lossx + lossy + lossz + lossix + lossiy + lossiz)


def KL_loss(model):

    kl_loss_l3F = simple_kl(model.layer3_muF, model.layer3_sigF)
    kl_loss_l3B = simple_kl(model.layer3_muB, model.layer3_sigB)

    total_KL_loss = kl_loss_l3B + kl_loss_l3F
    return tf.reduce_mean(total_KL_loss)


def MI(y_pred,y_true,bin_centers,
                            sigma_ratio=0.5,
                            max_clip=1,
                            crop_background=False):
    """
    Mutual Information for image-image pairs

    This function assumes that y_true and y_pred are both (batch_sizexheightxwidthxdepthxchan)

    """

    """ prepare MI. """
    vol_bin_centers = tf.Variable(bin_centers)
    num_bins = len(bin_centers)
    sigma = np.mean(np.diff(bin_centers)) * sigma_ratio
    preterm = tf.Variable(1 / (2 * np.square(sigma)),dtype=tf.float32)

    """ soft mutual info """
    y_pred = tf.clip_by_value(y_pred, 0, max_clip)
    y_true = tf.clip_by_value(y_true, 0, max_clip)

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
        y_pred = tf.expand_dims(tf.expand_dims(y_pred, 0), 2)
        y_true = tf.expand_dims(tf.expand_dims(y_true, 0), 2)

    else:
        # reshape: flatten images into shape (batch_size, heightxwidthxdepthxchan, 1)
        y_true = tf.reshape(y_true, (-1, tf.reduce_prod(tf.shape(y_true)[1:])))
        y_true = tf.expand_dims(y_true, 2)
        y_pred = tf.reshape(y_pred, (-1, tf.reduce_prod(tf.shape(y_pred)[1:])))
        y_pred = tf.expand_dims(y_pred, 2)

    nb_voxels = tf.cast(tf.shape(y_pred)[1], tf.float32)

    # reshape bin centers to be (1, 1, B)
    o = [1, 1, np.prod(vol_bin_centers.get_shape().as_list())]
    vbc = tf.reshape(vol_bin_centers, o)

    # compute image terms
    I_a = tf.exp(- preterm * tf.square(y_true - vbc))
    I_a /= tf.reduce_sum(I_a, -1, keepdims=True)

    I_b = tf.exp(- preterm * tf.square(y_pred - vbc))
    I_b /= tf.reduce_sum(I_b, -1, keepdims=True)

    # compute probabilities
    I_a_permute = tf.transpose(I_a, (0, 2, 1))
    pab = tf.matmul(I_a_permute, I_b)  # should be the right size now, nb_labels x nb_bins
    pab /= nb_voxels
    pa = tf.reduce_mean(I_a, 1, keepdims=True)
    pb = tf.reduce_mean(I_b, 1, keepdims=True)

    papb = tf.matmul(tf.transpose(pa, (0, 2, 1)), pb) + 1e-8
    mi = tf.reduce_sum(tf.reduce_sum(pab * tf.log(pab / papb + 1e-8), 1), 1)
    return -tf.reduce_mean(mi)








