import tensorflow as tf
import numpy as np
import sys

def generator_loss(fake_tgt,fake_src, src,tgt):
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
    forward_loss = 0.0#tf.reduce_mean(Dis_fake_tgt)
    inverse_loss = 0.0#tf.reduce_mean(Dis_fake_src)
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

def log_normal_pdf(sample, mean, logvar, raxis=1):
  #log2pi = tf.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar)),
      axis=raxis)

def KL_loss(model):
    '''
    ### For layer 0 Forward and Backward KL loss ###
    dim = model.layer0_f_flow.get_shape().as_list()
    dim = np.prod(dim[1:])
    l0_pdf_Fp = log_normal_pdf(tf.reshape(model.layer0_f_flow,[-1,dim]),0.0,0.0)
    l0_pdf_Fq = log_normal_pdf(tf.reshape(model.layer0_f_flow, [-1,dim]), model.layer0_muF, model.layer0_sigF)
    kl_loss_l0F = l0_pdf_Fp - l0_pdf_Fq

    l0_pdf_Bp = log_normal_pdf(tf.reshape(model.layer0_b_flow, [-1,dim]), 0.0, 0.0)
    l0_pdf_Bq = log_normal_pdf(tf.reshape(model.layer0_b_flow, [-1, dim]), model.layer0_muB, model.layer0_sigB)
    kl_loss_l0B = l0_pdf_Bp - l0_pdf_Bq

    ### For layer 1 Forward and Backward KL loss ###
    dim = model.layer1_f_flow.get_shape().as_list()
    dim = np.prod(dim[1:])
    l1_pdf_Fp = log_normal_pdf(tf.reshape(model.layer1_f_flow, [-1,dim]), 0.0, 0.0)
    l1_pdf_Fq = log_normal_pdf(tf.reshape(model.layer1_f_flow, [-1,dim]), model.layer1_muF, model.layer1_sigF)
    kl_loss_l1F = l1_pdf_Fp - l1_pdf_Fq

    l1_pdf_Bp = log_normal_pdf(tf.reshape(model.layer1_b_flow, [-1, dim]), 0.0, 0.0)
    l1_pdf_Bq = log_normal_pdf(tf.reshape(model.layer1_b_flow, [-1, dim]), model.layer1_muB, model.layer1_sigB)
    kl_loss_l1B = l1_pdf_Bp - l1_pdf_Bq

    ### For layer 2 Forward and Backward KL loss ###
    dim = model.layer2_f_flow.get_shape().as_list()
    dim = np.prod(dim[1:])
    l2_pdf_Fp = log_normal_pdf(tf.reshape(model.layer2_f_flow, [-1, dim]), 0.0, 0.0)
    l2_pdf_Fq = log_normal_pdf(tf.reshape(model.layer2_f_flow, [-1, dim]), model.layer2_muF, model.layer2_sigF)
    kl_loss_l2F = l2_pdf_Fp - l2_pdf_Fq

    l2_pdf_Bp = log_normal_pdf(tf.reshape(model.layer2_b_flow, [-1,dim]), 0.0, 0.0)
    l2_pdf_Bq = log_normal_pdf(tf.reshape(model.layer2_b_flow, [-1,dim]), model.layer2_muB, model.layer2_sigB)
    kl_loss_l2B = l2_pdf_Bp - l2_pdf_Bq
    '''
    ### For layer 3 Forward and Backward KL loss ###
    dim = model.layer3_f_flow.get_shape().as_list()
    dim = np.prod(dim[1:])
    l3_pdf_Fp = log_normal_pdf(tf.reshape(model.layer3_f_flow, [-1,dim]), 0.0, 0.0)
    l3_pdf_Fq = log_normal_pdf(tf.reshape(model.layer3_f_flow, [-1, dim]), model.layer3_muF, model.layer3_sigF)
    kl_loss_l3F = l3_pdf_Fp - l3_pdf_Fq

    l3_pdf_Bp = log_normal_pdf(tf.reshape(model.layer3_b_flow, [-1, dim]), 0.0, 0.0)
    l3_pdf_Bq = log_normal_pdf(tf.reshape(model.layer3_b_flow, [-1, dim]), model.layer3_muB, model.layer3_sigB)
    kl_loss_l3B = l3_pdf_Bp - l3_pdf_Bq
    '''
    total_KL_loss = kl_loss_l0B + kl_loss_l0F+kl_loss_l1B + kl_loss_l1F\
                    +kl_loss_l2B + kl_loss_l2F\
                    + kl_loss_l3B + kl_loss_l3F
    '''
    return tf.reduce_mean(kl_loss_l3F+kl_loss_l3B)







