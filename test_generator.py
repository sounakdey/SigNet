#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 2 11:59:15 2016

@author: Anjan Dutta
"""
from scipy import misc
from keras.models import Model

def generate_arrays_from_file(path):
    
    f = open(path)
    for line in f:
        # create numpy arrays of input data
        # and labels, from each line in the file
        x, y, z = line.split(' ')
        img1 = misc.imread(x)
        img2 = misc.imread(y)        
        yield (img1, img2, z)
    f.close()
    
data_file = '/home/anjan/Workspace/SignatureVerification/Datasets/GPDS960_pairs.txt'

Model.fit_generator( generate_arrays_from_file( data_file ), 
                    samples_per_epoch=10000, 
                    nb_epoch=10 )