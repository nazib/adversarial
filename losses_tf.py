import tensorflow as tf
import sys


def generator_loss(fake_tgt,fake_src, src,tgt, Dis_fake_src,Dis_fake_tgt):
    #### Basic GAN loss ###
    #forward_loss = -tf.nn.sigmoid_cross_entropy_with_logits(logits=Dis_fake_tgt,labels=tf.zeros_like(Dis_fake_tgt))
    #inverse_loss = -tf.nn.sigmoid_cross_entropy_with_logits(logits=Dis_fake_src,labels=tf.zeros_like(Dis_fake_src))
    '''
    ### GAN loss with CC ###
    forward_loss = -tf.reduce_mean(tf.log(Dis_fake_tgt))
    inverse_loss = -tf.reduce_mean(tf.log(Dis_fake_src))
    forward_loss_reg =cc3D(fake_tgt,tgt)
    inverse_loss_reg =cc3D(fake_src,src)
    loss = (forward_loss + forward_loss_reg) - (inverse_loss + inverse_loss_reg)
    '''
    ### WGAN Loss ###
    forward_loss = tf.reduce_mean(Dis_fake_tgt)
    inverse_loss = tf.reduce_mean(Dis_fake_src)
    forward_loss_reg = cc3D(fake_tgt, tgt)
    inverse_loss_reg = cc3D(fake_src, src)
    #cycle_loss= Cyclic_loss(fake_tgt,fake_src,src,tgt)
    loss = (forward_loss + forward_loss_reg ) + (inverse_loss + inverse_loss_reg )
    return loss

def discriminator_loss(real_output, fake_output):

    #real_loss = -tf.nn.sigmoid_cross_entropy_with_logits(logits=real_output,labels=tf.zeros_like(real_output))
    #fake_loss = -tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_output,labels=-tf.ones_like(fake_output))
    #total_loss = tf.reduce_mean(real_loss + fake_loss)
    #total_loss = tf.reduce_mean(tf.log(real_output) + tf.log(1. - fake_output))

    ### WGAN LOSS###
    #real_loss = tf.reduce_mean(real_output)
    #fake_loss = tf.reduce_mean(fake_output)
    #total_loss = real_loss - fake_loss
    
    ### Triplet Loss ###
    #real_loss = tf.reduce_mean(real_output)
    #fake_loss = tf.reduce_mean(fake_output)
    #total_loss = tf.maximum((real_loss - fake_loss)+0.005, 0.0)

    ## Hinge loss ####
    real_loss = tf.reduce_mean(tf.nn.relu(1.0 - real_output))
    fake_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake_output))
    total_loss = real_loss + fake_loss
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

def Cyclic_loss(fake_tgt,fake_src, src,tgt):
    src_cyclic = tf.reduce_mean(tf.abs(src-fake_src))
    tgt_cyclic = tf.reduce_mean(tf.abs(tgt-fake_tgt))
    return src_cyclic+tgt_cyclic

def KL_loss():


