import numpy as np
import nibabel as nib
from itertools import permutations
#import BatchDataReader
import random
import os
import sys
import glob
import h5py
import time
import BatchDataReader

if __name__ == "__main__":

    data = "/home/n9614885/100/DV_aligned/"
    patch = "/home/n9614885/100/patches/"

    if len(sys.argv) != 1:
        print("Please insert data directory and pacth directory")
    else:
        #extract_Hpatches(sys.argv[1], sys.argv[2])
        reader = BatchDataReader.BatchDataset(data,4,patch)
        reader.extract_Hpatches()
        print("Patch xtraction Complete")





