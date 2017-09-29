# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 17:24:39 2017

@author: sounak
"""

import numpy as np
import cv2
np.random.seed(1337)  # for reproducibility
import os
import argparse

from keras.utils.visualize_util import plot

import tensorflow as tf
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense, Dropout, Input, Lambda, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.preprocessing import image
from keras import backend as K
from SignatureDataGenerator import SignatureDataGenerator
import getpass as gp
import sys
from keras.models import Sequential, Model
from keras.optimizers import SGD, RMSprop, Adadelta
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import random
random.seed(1337)

# Create a session for running Ops on the Graph.
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))
    
def create_base_network_signet(input_shape):
    
    seq = Sequential()
    seq.add(Convolution2D(96, 11, 11, activation='relu', name='conv1_1', subsample=(4, 4), input_shape= input_shape, 
                        init='glorot_uniform', dim_ordering='tf'))
    seq.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
    seq.add(MaxPooling2D((3,3), strides=(2, 2)))    
    seq.add(ZeroPadding2D((2, 2), dim_ordering='tf'))
    
    seq.add(Convolution2D(256, 5, 5, activation='relu', name='conv2_1', subsample=(1, 1), init='glorot_uniform',  dim_ordering='tf'))
    seq.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
    seq.add(MaxPooling2D((3,3), strides=(2, 2)))
    seq.add(Dropout(0.3))# added extra
    seq.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    
    seq.add(Convolution2D(384, 3, 3, activation='relu', name='conv3_1', subsample=(1, 1), init='glorot_uniform',  dim_ordering='tf'))
    seq.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    
    seq.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2', subsample=(1, 1), init='glorot_uniform', dim_ordering='tf'))    
    seq.add(MaxPooling2D((3,3), strides=(2, 2)))
    seq.add(Dropout(0.3))# added extra
#    model.add(SpatialPyramidPooling([1, 2, 4]))
    seq.add(Flatten(name='flatten'))
    seq.add(Dense(1024, W_regularizer=l2(0.0005), activation='relu', init='glorot_uniform'))
    seq.add(Dropout(0.5))
    
    seq.add(Dense(128, W_regularizer=l2(0.0005), activation='relu', init='glorot_uniform')) # softmax changed to relu
    
    return seq    
    

def compute_accuracy_roc(predictions, labels):
   '''Compute ROC accuracy with a range of thresholds on distances.
   '''
   dmax = np.max(predictions)
   dmin = np.min(predictions)
   nsame = np.sum(labels == 1)
   ndiff = np.sum(labels == 0)
   
   step = 0.01
   max_acc = 0
   
   for d in np.arange(dmin, dmax+step, step):
       idx1 = predictions.ravel() <= d
       idx2 = predictions.ravel() > d
       
       tpr = float(np.sum(labels[idx1] == 1)) / nsame       
       tnr = float(np.sum(labels[idx2] == 0)) / ndiff
       acc = 0.5 * (tpr + tnr)       
#       print ('ROC', acc, tpr, tnr)
       
       if (acc > max_acc):
           max_acc = acc
           
   return max_acc
    
def read_signature_data(dataset, ntuples, height = 30, width = 100):
    
    usr = gp.getuser()

    image_dir = '/home/' + usr + '/Workspace/SignatureVerification/Datasets/' + dataset + '/'
    data_file = image_dir + dataset + '_pairs.txt'
    
    f = open( data_file, 'r' )
    lines = f.readlines()
    f.close()

    
    
    idx = np.random.choice(list(range(len(lines))), ntuples)
    
    lines = [lines[i] for i in idx]
    
    images = []
    
    for line in lines:
        file1, file2, label = line.split(' ')
                                       
        img1 = image.load_img(image_dir + file1, grayscale = True, 
                target_size=(height, width))
                
        img1 = image.img_to_array(img1, dim_ordering='tf')
                
        images.append(img1)
        
        img2 = image.load_img(image_dir + file1, grayscale = True, 
                target_size=(height, width))
            
        img2 = image.img_to_array(img2, dim_ordering='tf')
                
        images.append(img2)
        
    return np.array(images)
        
dataset = 'GPDS300'
if dataset == 'Bengali':

    tot_writers = 100
    num_train_writers = 80
    num_valid_writers = 10
    
elif dataset == 'Hindi':
    
    tot_writers = 160
    num_train_writers = 100
    num_valid_writers = 10
    
elif dataset == 'GPDS300':

    tot_writers = 300
    num_train_writers = 240
    num_valid_writers = 30
    
elif dataset == 'GPDS960':

    tot_writers = 4000
    num_train_writers = 3200
    num_valid_writers = 400
    
elif dataset == 'CEDAR1':

    tot_writers = 55
    num_train_writers = 45
    num_valid_writers = 5

num_test_writers = tot_writers - (num_train_writers + num_valid_writers)

# parameters
batch_sz = 1
nsamples = 276 
img_height = 155
img_width = 220
featurewise_center = False
featurewise_std_normalization = True
zca_whitening = False
nb_epoch = 20    
input_shape=(img_height, img_width, 1)

# initialize data generator   
datagen = SignatureDataGenerator(dataset, tot_writers, num_train_writers, 
    num_valid_writers, num_test_writers, nsamples, batch_sz, img_height, img_width,
    featurewise_center, featurewise_std_normalization, zca_whitening)

# data fit for std
X_sample = read_signature_data(dataset, int(0.5*tot_writers), height=img_height, width=img_width)
datagen.fit(X_sample)
del X_sample

# network definition
base_network = create_base_network_signet(input_shape)

input_a = Input(shape=(input_shape))
input_b = Input(shape=(input_shape))

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model(input=[input_a, input_b], output=distance)

# compile model
rms = RMSprop(lr=1e-5, rho=0.9, epsilon=1e-08)
adadelta = Adadelta()
model.compile(loss=contrastive_loss, optimizer=adadelta)

   
fname = os.path.join('/home/sounak/Desktop/' , 'weights_'+str(dataset)+'.hdf5')


# load the best weights for test
model.load_weights(fname)


m = model.layers[-2]


l = m.layers[-7]    # 7, 9, 14
gen =datagen.next_train()
gen.next()
example= gen.next()
x_0 = example[0][0]
x_1 = example[0][1]
y   = example[1]
inputs = [K.learning_phase()] + m.inputs
f = K.function(inputs, [l.output])
activation_conv_0 = f([0] + [x_0])[0]
activation_conv_1 = f([0] + [x_1])[0]
#img = x[0,:,:,:]
#img = 255-cv2.convertScaleAbs(np.tile(img,(1,1,3)), alpha=255)
#cv2.imwrite('/home/sounak/Documents/stash/sign2.jpg', img)
#sys.exit()
#print activation_conv.shape
print 'images same class? ',y
energy=np.zeros((256,))
for i in xrange(activation_conv_0.shape[3]):
    energy[i]=np.sum(activation_conv_0[0,:,:,i])
    filter_activation = cv2.convertScaleAbs(cv2.resize(activation_conv_0[0,:,:,i],(220,155)), alpha=255)
    img = x_0[0,:,:,:]
    img = cv2.convertScaleAbs(np.tile(img,(1,1,3)), alpha=128)
    img[:,:,0:3]/=2
    mask= cv2.convertScaleAbs(cv2.resize(activation_conv_0[0,:,:,i],(220,155)), alpha=255)
    img[:,:,0] = mask
#    img[:,:,1] += mask
    img[:,:,2] += mask
#    cv2.imwrite('/home/sounak/Documents/stash/filter'+str(i)+'.jpg', img)
#    cv2.imwrite('/home/sounak/Documents/stash/4_conv_resized/filter'+str(i)+'.jpg', filter_activation)
print 'Top 10 higher energy activations id for image1 ',np.argsort(energy)[-10:]
energy=np.zeros((256,))
for i in xrange(activation_conv_1.shape[3]):
    energy[i]=np.sum(activation_conv_1[0,:,:,i])
    filter_activation = cv2.convertScaleAbs(cv2.resize(activation_conv_1[0,:,:,i],(220,155)), alpha=255)
    img = x_1[0,:,:,:]
    img = cv2.convertScaleAbs(np.tile(img,(1,1,3)), alpha=128)
    img[:,:,0:3]/=2
    mask= cv2.convertScaleAbs(cv2.resize(activation_conv_1[0,:,:,i],(220,155)), alpha=255)
    img[:,:,0] = mask
#    img[:,:,1] += mask
    img[:,:,2] += mask
#    cv2.imwrite('/home/sounak/Documents/stash/filter'+str(i)+'.jpg', img)
#    cv2.imwrite('/home/sounak/Documents/stash/4_conv_sign2_resized/filter'+str(i)+'.jpg', filter_activation)
print 'Top 10 higher energy activations id for image2 ',np.argsort(energy)[-10:]