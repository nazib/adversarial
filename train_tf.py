# python imports
import os
import glob
import sys
import random
import nibabel as nib
# third-party imports
import tensorflow as tf
import numpy as np
import scipy.io as sio
from losses_tf import *
from InverseNet_tf import *
from InverseNet_tfv3 import *
# from datagenerators import example_gen
from measurment import *
from utility import *
import BatchDataReader
from itertools import permutations
from ConfigReader import *
#from AdversarialNet import *
from partition import *
import pdb

def train(config_file):
    config = Configuration()
    config.CreateConfig(config_file)
    config.PrintConfiguration()

    vol_size = (config.patch_size[0], config.patch_size[1], config.patch_size[2])
    batch_size = 4

    training_data = glob.glob(config.base_directory + "/train/*.nii.gz")
    training_data.sort()
    #validation_data = training_data[len(training_data)-3:len(training_data)]
    #training_data = training_data[0:len(training_data)-3]
    random.shuffle(training_data)

    train_dataset_reader = BatchDataReader.BatchDataset(training_data,batch_size,config.Number_of_patch)

    model_dir = config.base_directory + config.Model_name
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    gpu = '/gpu:' + str(config.GPU)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.GPU)
    configurnet = tf.ConfigProto()
    configurnet.gpu_options.allow_growth = True
    configurnet.gpu_options.per_process_gpu_memory_fraction = 0.9
    configurnet.allow_soft_placement = True

    with tf.device(gpu):

        src = tf.placeholder(tf.float32, shape=(None, vol_size[0], vol_size[1], vol_size[2], 1),
                                  name="source")
        tgt = tf.placeholder(tf.float32, shape=(None, vol_size[0], vol_size[1], vol_size[2], 1),
                                  name="target")
        diff = tf.placeholder(tf.float32, shape=(None, vol_size[0], vol_size[1], vol_size[2], 6),
                             name="target")
        if config.ModelType == 'Dense':
            G_net = InverseNetDense_tf(src, tgt, vol_size, batch_size, False)
        else:
            G_net = InverseNet_tf(src, tgt, vol_size, batch_size, False)

        fake_tgt, fake_src, src_cyc, tgt_cyc, Flow = G_net.Build()

        Dis_input_src =  tf.placeholder(tf.float32, shape=(None, vol_size[0], vol_size[1], vol_size[2], 1),
                                  name="DiscriminatorInput_src")
        Dis_input_tgt = tf.placeholder(tf.float32, shape=(None, vol_size[0], vol_size[1], vol_size[2], 1),
                                       name="DiscriminatorInput_tgt")

        Dis_real_tgt = discriminator(Dis_input_tgt, fake_tgt, "target", False)
        Dis_real_src = discriminator(Dis_input_src, fake_src, "source", False)

        Dis_fake_tgt = discriminator(Dis_input_src,fake_tgt,"target",True)

        Dis_fake_src = discriminator(Dis_input_tgt,fake_src,"source",True)

        ## Building GAN ##
        Dis_tgt_loss = discriminator_loss(Dis_real_tgt,Dis_fake_tgt)
        Dis_src_loss = discriminator_loss(Dis_real_src,Dis_fake_src)
        Dis_loss = Dis_tgt_loss + Dis_src_loss

        G_loss = generator_loss(Dis_fake_src,Dis_fake_tgt)
        bins = np.arange(0, 32, 1,dtype=np.float32)

        if config.similarity_loss == 'cc' or config.similarity_loss == 'CC':
            similarity = cc3D(fake_tgt, tgt) + cc3D(fake_src, src)
        if config.similarity_loss == 'mi' or config.similarity_loss == 'MI':
            similarity = MI(fake_tgt, tgt, bins) + MI(fake_src, src, bins)
        if config.cyc_loss == 'on':
            Cyc_loss = Cyclic_loss(src_cyc, tgt_cyc, G_net.src, G_net.tgt, config.ssim_loss)
     
        Dis_optimizer = tf.train.GradientDescentOptimizer(0.00002).minimize(Dis_loss)

        total_loss = 0.01 * G_loss + similarity + Cyc_loss

        G_optimizer = tf.train.AdamOptimizer(0.0001, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(total_loss)

        init = tf.global_variables_initializer()
        sess = tf.Session(config=configurnet)
        sess.run(init)
        writer = tf.summary.FileWriter(model_dir)
        model_saver = tf.train.Saver(max_to_keep=None)

    start = config.iteration_start
    iteration = int(config.Number_of_patch / batch_size)

    if start > 0:
        if start > iteration:
            step_st = start - np.int((start / iteration)) * iteration
            paris_st = np.int(start / np.float(iteration))
        else:
            step_st = start

    # total_pairs = len(train_patch_pairs)*(len(train_patch_pairs)-1)
    x = np.arange(len(training_data))
    total_pairs = list(permutations(x, 2))
    random.shuffle(total_pairs)

    for pairs in list(total_pairs):

        src_im, tgt_im = train_dataset_reader.create_pairs(pairs[0], pairs[1], config.patch_size)

        if config.flow_loss == 'on':
            flow = Generate_deformation(training_data[pairs[0]], (576,576,192), 10)

        print ("{0} Patches are selected".format(config.Number_of_patch))
        s = 0
        for step in range(iteration):

            src_patch = src_im[step]
            tgt_patch = tgt_im[step]
            #diff_patch = np.abs(tgt_patch - src_patch)
            #if start ==0:
            #flow = Generate_deformation(diff_patch, vol_size, 10)
            #flow = np.random.rand(batch_size, vol_size[0], vol_size[1], vol_size[2], 6)
            #Training Discriminator Five times to make Network Stable

            d_tgt,d_src, _ = sess.run([Dis_tgt_loss, Dis_src_loss,Dis_optimizer], feed_dict={src: src_patch, tgt: tgt_patch, Dis_input_tgt: tgt_patch, Dis_input_src: src_patch})

            t_loss, g_loss, cyc_loss, _ = sess.run([total_loss, G_loss, Cyc_loss, G_optimizer],
                                                                  feed_dict={src: src_patch, tgt: tgt_patch, Dis_input_tgt:tgt_patch, Dis_input_src:src_patch})

            train_loss = np.double([t_loss, g_loss, cyc_loss, d_tgt,d_src, 0.0])

            message = "Start:{0} Paris :{1},{2} Step :{3} t_Loss:{4} G_loss: {5} Cyc_loss:{6} D_tgt:{7} D_src:{8}\n"\
                .format(start, pairs[0], pairs[1], step, t_loss, g_loss, cyc_loss,d_tgt,d_src)
            print (message)

            if t_loss >= -0.1:
                file = open("Detect.txt", "a")
                data = "Target:{0} Source:{1} Patch:{2}  iteration: {3} LossValue:{4}".format(training_data[pairs[0]], training_data[pairs[1]],
                                                                              step, start, t_loss)
                file.write(data)
                file.write("\n")
                file.close()

            model_saving_dir = model_dir+"/models"

            if not os.path.isdir(model_saving_dir):
                os.mkdir(model_saving_dir)

            model_saving_name = model_dir.split('/')[-1]

            if start % config.Model_saver == 0:
                if start == 0:
                    model_saver.save(sess,model_saving_dir+'/'+model_saving_name, global_step=start)
                else:
                    model_saver.save(sess, model_saving_dir+'/'+model_saving_name, global_step=start)

            if start % 10== 0:
                write_summary(start, sess, train_loss, model_dir)

            s = s + batch_size
            start = start + 1


def ApplyValidation(validation_data,session, model_dir, config):

    target_image = nib.load(validation_data[2]).get_data()
    patchEx = PatchUtility(config.patch_size,target_image.shape, 0.5, patches=[], image=target_image)
    target_patches = patchEx.extract_patches()

    model_name = model_dir.split('/')[-2]
    saver = tf.train.import_meta_graph(model_dir+'/'+model_name+"-0.meta")
    saver.restore(session, tf.train.latest_checkpoint(model_dir))

    src_place = session.graph.get_tensor_by_name("source:0")
    tgt_place = session.graph.get_tensor_by_name("target:0")
    f_im = session.graph.get_tensor_by_name('Generator/Forward_im:0')
    i_im = session.graph.get_tensor_by_name('Generator/Inverse_im:0')

    cc = 0
    mi = 0

    for j in range(len(validation_data)-1):

        src_im = nib.load(validation_data[j]).get_data()
        patchEx = PatchUtility(config.patch_size, src_im.shape, 0.5, patches=[], image=src_im)
        src_im = patchEx.extract_patches()
        warped_patches =[]

        for k in range(len(src_im)):

            src = src_im[k,:, :, :]
            src = np.reshape(src, (1, config.patch_size[0], config.patch_size[1], config.patch_size[2], 1))
            tgt = target_patches[k, :, :, :]
            tgt = np.reshape(tgt, (1, config.patch_size[0], config.patch_size[1], config.patch_size[2], 1))

            f_image, i_image = session.run([f_im,i_im],feed_dict={src_place: src, tgt_place: tgt})
            f_image = np.reshape(f_image, config.patch_size)
            warped_patches.append(f_image)

        warped_patches = list2array(warped_patches)
        combiner = PatchUtility(config.patch_size, (256,256,96), 0.5, patches=warped_patches, image=[])
        registered_image = combiner.combine_patches()

        cc += volume_cross(registered_image,target_image)
        mi += mutual_info(registered_image,target_image,30)

    cc = cc /2
    mi = mi /2
    return cc, mi

if __name__ == "__main__":
    if (len(sys.argv) != 2):
        print ("Configuration file name is required")
    else:
        train(sys.argv[1])


