# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 11:57:34 2017

@author: sounak
"""

import numpy as np
np.random.seed(1337)  # for reproducibility
import getpass as gp
import random
import numpy as np

from keras.preprocessing import image
from fit_data import read_signature_data


dataset = 'GPDS960'
tot_writers = 4000
num_train_writers = 100
num_test_writers = 5
batch_size = 100

usr = gp.getuser() 
# dimensions of our images.
img_width, img_height = 220, 155

dataset=='GPDS960'
image_dir = '/home/' + usr + '/Workspace/Datasets/GPDS960_tot/'
data_file = '/home/' + usr  + '/Workspace/Datasets/GPDS960/'+dataset + '_pairs.txt'

idx_writers = list(range(tot_writers))
        
idx_train_writers = sorted(np.random.choice(idx_writers, num_train_writers, replace=False))
idx_test_writers = sorted(np.random.choice([x for x in idx_writers if x not in idx_train_writers], num_test_writers, replace=False))

size = 996
nsamples = 276 #24C2

def fit(x):
    
    x = np.asarray(x, dtype=np.float32())
    x = np.copy(x)
    std = np.std(x, axis=(0, 1, 2))

    return std

        
def standardize(x, std):
    x /= (std + 1e-7)    
    return x


def load_data():
    idx_train_lines = []
    for iw in idx_train_writers:
        idx_train_lines += list(range(iw * size, (iw + 1) * size))
    
    idx_test_lines = []
    for iw in idx_test_writers:
        idx_test_lines += list(range(iw * size, (iw + 1) * size))
        
    f = open( data_file, 'r' )
    lines = f.readlines()
    f.close()
    
    train_lines = [lines[i] for i in idx_train_lines]
    #test_lines = [lines[i] for i in idx_train_lines]        
    test_lines = [lines[i] for i in idx_test_lines]
    del lines
    
    # for train writers    
    idx_lines = []
    
    lp = []
    lin = []
    
    for iline, line in enumerate(train_lines):            
        
        file1, file2, label = line.split(' ')
        
        label = int(label)
        
        lp += [label]        
        lin += [iline]
        
        if(len(lp) != 0 and len(lp) % size == 0):                
                        
            idx1 = [i for i, x in enumerate(lp) if x == 1]
            idx2 = [i for i, x in enumerate(lp) if x == 0]
            
            idx1 = np.random.choice(idx1, nsamples)
            idx2 = np.random.choice(idx2, nsamples)
            
            idx = [None]*(len(idx1)+len(idx2))
            
            idx[::2] = idx1
            idx[1::2] = idx2
            
            del idx1
            del idx2
            
            idx_lines += [lin[i] for i in idx]
            
            lp = []
            lin = []            
        
    train_lines = [train_lines[i] for i in idx_lines]
    
    just_1 = train_lines[0:][::2]
    just_0 = train_lines[1:][::2]
    random.shuffle(just_1)
    random.shuffle(just_0)
    train_lines= [item for sublist in zip(just_1,just_0) for item in sublist]
    
    # for test writers
    idx_lines = []
    
    lp = []
    lin = []
    
    for iline, line in enumerate(test_lines):            
        
        file1, file2, label = line.split(' ')
        
        label = int(label)
        
        lp += [label]        
        lin += [iline]
        
        if(len(lp) != 0 and len(lp) % size == 0):                
                        
            idx1 = [i for i, x in enumerate(lp) if x == 1]
            idx2 = [i for i, x in enumerate(lp) if x == 0]
            
            idx1 = np.random.choice(idx1, nsamples)
            idx2 = np.random.choice(idx2, nsamples)
            
            idx = [None]*(len(idx1)+len(idx2))
            
            idx[::2] = idx1
            idx[1::2] = idx2
            
            del idx1
            del idx2
            
            idx_lines += [lin[i] for i in idx]
            
            lp = []
            lin = []            
        
    test_lines = [test_lines[i] for i in idx_lines]
    
    just_1 = test_lines[0:][::2]
    just_0 = test_lines[1:][::2]
    random.shuffle(just_1)
    random.shuffle(just_0)
    test_lines= [item for sublist in zip(just_1,just_0) for item in sublist]
    
    height=img_height
    width=img_width
#    input_shape=(height, width, 1)
#    cur_train_index = 0
#    cur_test_index = 0
#    #batch_sz = batch_sz
#    samples_per_train = 2*nsamples*num_train_writers
#    samples_per_test = 2*nsamples*num_test_writers
    # Incase dim_ordering = 'tf'
#    channel_axis = 3
#    row_axis = 1
#    col_axis = 2
    
    
    X_sample = read_signature_data(dataset, 2000, height=img_height, width=img_width)
    std = fit(X_sample)
    
    
    image_pairs = []
    label_pairs = []
    for line in train_lines:
        file1, file2, label = line.split(' ')
        
        img1 = image.load_img(image_dir + file1, grayscale = True,
        target_size=(height, width))
                        
        img1 = image.img_to_array(img1, dim_ordering='tf')
        
        img1 = standardize(img1,std)
    
        
        img2 = image.load_img(image_dir + file2, grayscale = True,
        target_size=(height, width))
        
        img2 = image.img_to_array(img2, dim_ordering='tf')
        
        img2 = standardize(img2,std)
        
#        image_pairs += [[img1, img2]]
        image_pairs += [[np.array(img1), np.array(img2)]]
        label_pairs += [int(label)]
    
#    tr_pairs = [np.array(image_pairs)[:,0], np.array(image_pairs)[:,1]]
    tr_pairs = np.array(image_pairs)
    tr_y = np.array(label_pairs)
        
    image_pairs = []
    label_pairs = []    
    for line in test_lines:
        file1, file2, label = line.split(' ')
        
        img1 = image.load_img(image_dir + file1, grayscale = True,
        target_size=(height, width))
                        
        img1 = image.img_to_array(img1, dim_ordering='tf')
        
        img1 = standardize(img1, std)
    
        
        img2 = image.load_img(image_dir + file2, grayscale = True,
        target_size=(height, width))
        
        img2 = image.img_to_array(img2, dim_ordering='tf')
        
        img2 = standardize(img2,std)
        
#        image_pairs += [[img1, img2]]
        image_pairs += [[np.array(img1), np.array(img2)]]
        label_pairs += [int(label)]
        
#    te_pairs = [np.array(image_pairs)[:,0], np.array(image_pairs)[:,1]]
    te_pairs = np.array(image_pairs)    
    te_y = np.array(label_pairs)
    
    

    return tr_pairs, tr_y, te_pairs, te_y
    
    
