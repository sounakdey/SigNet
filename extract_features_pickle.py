# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 17:13:51 2017

@author: Anjan Dutta
"""
import numpy as np
import getpass as gp
import pickle as pkl
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from tqdm import tqdm
import tensorflow as tf
import keras as K




# Create a session for running Ops on the Graph.
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#sess = tf.Session(config=config)
#K.set_session(sess)

# Parameters
height = 155  #224
width = 220   #224
usr = gp.getuser()
size = 996

image_dir = '/home/' + usr + '/Workspace/Datasets/GPDS960/'
pickle_dir = image_dir + 'pickles/'
data_file = image_dir + 'GPDS960_pairs.txt'

# Open the file
f = open( data_file, 'r' )
lines = f.readlines()
f.close()

#lines = lines[0:10]

# Initialize the VGG19 at the end of block5_pool
base_model = VGG16(weights='imagenet', include_top=True)   
model = Model(input=base_model.input, output=base_model.get_layer('block4_pool').output)
    
feature_pairs = []
label_pairs = []

line_count = 0
    
for line in lines:
#    pass
    file1, file2, label = line.split(' ')
    
    img1 = image.load_img(image_dir + file1, target_size=(height, width))
    x = image.img_to_array(img1)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    features1 = model.predict(x)
    
    img2 = image.load_img(image_dir + file2, target_size=(height, width))
    x = image.img_to_array(img2)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    features2 = model.predict(x)
    
    feature_pairs += [[ features1.ravel(), features2.ravel() ]]
    
    label_pairs += [label]
    
    line_count += 1
    
    if(line_count % size == 0):
        
        idx_writer = int(line_count / size)
        str_idx_writer = str(idx_writer).zfill(4)

        with open(pickle_dir + str_idx_writer + '.pickle', 'wb') as f:
            pkl.dump([feature_pairs, label_pairs], f)
        
        feature_pairs = []
        label_pairs = []