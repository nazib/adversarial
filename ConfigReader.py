import configparser
import os
import numpy as np
import h5py
import nibabel as nib


class Configuration:
    resolution =''
    patch_size = 64
    base_directory = ''
    Learning_Rate = 0.0
    Model_name = ''
    Model_saver = 0
    itaration_start = 0
    Regularizer = 0.0
    ModelType = ''
    GPU=''
           
    def __init__(self):
        print ("Initialiazing Config")

    def CreateConfig(self,path):
        if not os.path.exists(path):
            print("configuration file Does Not Exists")
        
        config = configparser.ConfigParser()
        config.read(path)
        self.resolution = config.get('image info','resolution')
        self.patch_size = np.asarray(config.get('image info','patch_size')[1:-1].split(','),dtype=int)#int(config.get('image info','patch_size'))
        self.base_directory = config.get('image info','base_dir')
        self.Number_of_patch = int(config.get('image info', 'Number_of_patches'))

        self.Learning_Rate = float(config.get('Network Settings','Learning_Rate'))
        self.Model_name = config.get('Network Settings','Model_name')
        self.Model_saver = int(config.get('Network Settings','Model_save_iter'))
        self.iteration_start = int(config.get('Network Settings','Start'))
        self.Regularizer = float(config.get('Network Settings','Regularizer'))
        self.ModelType = config.get('Network Settings','Modeltype')
        self.GPU = config.get('Network Settings','gpuid')
        self.similarity_loss = config.get('Network Settings','similarity_loss')
        self.cyc_loss = config.get('Network Settings','cyc_loss')

    def PrintConfiguration(self):
        print(" The network is Running with Following parameters")
        print(" Network Type:"+self.ModelType)
        print(" Model Parameters saved in :"+self.base_directory+"/"+self.Model_name)
        print(" Image Resolution used :"+self.resolution)
        print(" Patch Size: "+str(self.patch_size))
        print(" Learning Rate :"+str(self.Learning_Rate))
        print(" Regularizer Weight: ",str(self.Regularizer))
        print(" Model Svaing in each "+str(self.Model_saver)+" iteration")
        print(" Training iteration starts from :"+str(self.itaration_start))
        print("========================================================================")
    
    def ReadTestConfig(self,path):

        if not os.path.exists(path):
            print("Test configuration file Does Not Exists")
        
        config = configparser.ConfigParser()
        config.read(path)

        ############## Reading Test Configurations ############

        self.TestResolution = config.get('data info','resolution')
        self.TestPatchSize = np.asarray(config.get('data info','patch_size')[1:-1].split(','),dtype=int)
        #int(config.get('data info','patch_size'))
        self.TestDataDir = config.get('data info','test_data_dir')

        self.TestAtlasFile = config.get('data info','atlas')
        self.TestAtlas = h5py.File(self.TestDataDir+self.TestAtlasFile,'r')['moving']
        self.TestSliceNo = np.int(config.get('utility','SliceNo'))
        self.TestModelDir = config.get('Network Settings','ModelDir')
        self.TestModelNumber = config.get('Network Settings','ModelNumber')
        self.TestModelType = config.get('Network Settings','ModelType')
        self.TestGPU = '/gpu:'+config.get('Network Settings','GPUID')
        self.TestImageSize = np.asarray(config.get('utility','ImageSize')[1:-1].split(','),dtype=int)
        self.TestOverlap = float(config.get('utility','Overlap'))
        row = config.get('utility','affine')[1:-1].split(',')
        self.TestAffineMatrix =  np.eye(4,4,dtype=np.float)
        self.TestAffineMatrix[0,0]=row[0]
        self.TestAffineMatrix[1,1]=row[1]
        self.TestAffineMatrix[2,2]=row[2]

        self.TestCubeSave =config.get('utility','CubeSave')
        if self.TestCubeSave == 'on':
            self.TestCubeNumber = int(config.get('utility','CubeNumber'))
        self.TestFlowSave =config.get('utility','FlowSave')








        






