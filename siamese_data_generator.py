# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 12:31:05 2017

@author: sounak
"""

'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs
So that we have 1000 training examples for each class, and 400 validation examples for each class.
In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''

import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import tensorflow as tf
from keras import backend as K
from fit_data import read_data_gpds960

# dimensions of our images.
img_width, img_height = 220, 155

train_data_dir = '/home/sounak/Workspace/Datasets/GPDS960/'
validation_data_dir = '/home/sounak/Workspace/Datasets/GPDS960/'
nb_train_samples = 47574#162000#216000#47574
nb_validation_samples = 800
nb_epoch = 50
batch_size=100

# Create a session for running Ops on the Graph.
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

# build the IJCNN 2016 network
model = Sequential()

#model.add(ZeroPadding2D((0, 0), input_shape=(img_height, img_width, 1)))
model.add(Convolution2D(96, 11, 11, activation='relu', name='conv1_1', subsample=(4, 4),input_shape=(img_height, img_width, 1), 
                        init='glorot_uniform', dim_ordering='tf'))
model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
model.add(MaxPooling2D((3,3), strides=(2, 2)))

model.add(ZeroPadding2D((2, 2), dim_ordering='tf'))
model.add(Convolution2D(256, 5, 5, activation='relu', name='conv2_1', subsample=(1, 1), init='glorot_uniform',  dim_ordering='tf'))
model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
model.add(MaxPooling2D((3,3), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
model.add(Convolution2D(384, 3, 3, activation='relu', name='conv3_1', subsample=(1, 1), init='glorot_uniform',  dim_ordering='tf'))
model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2', subsample=(1, 1), init='glorot_uniform', dim_ordering='tf'))

model.add(MaxPooling2D((3,3), strides=(2, 2)))
model.add(Flatten(input_shape=model.output_shape[1:]))
model.add(Dense(4096, W_regularizer=l2(0.0005), activation='relu', init='glorot_uniform'))
model.add(Dropout(0.5))
model.add(Dense(881, W_regularizer=l2(0.0005), activation='softmax', init='glorot_uniform'))
#model.add(Dense(3000, W_regularizer=l2(0.0005), activation='softmax', init='glorot_uniform'))
#################################################################################
### built the ISVC 2016 network####################
#model = Sequential()
#model.add(ZeroPadding2D((1, 1), input_shape=(img_height, img_width, 1)))
#model.add(Convolution2D(96, 11, 11, activation='relu', name='conv1_1', subsample=(3, 3), 
#                        init='glorot_uniform', dim_ordering='tf'))
#model.add(ZeroPadding2D((1, 1),  dim_ordering='tf'))
#model.add(Convolution2D(96, 3, 3, activation='relu', name='conv1_2', subsample=(1, 1), 
#                        init='glorot_uniform', dim_ordering='tf'))
#model.add(ZeroPadding2D((1, 1),  dim_ordering='tf'))
#model.add(MaxPooling2D((2,2), strides=(2, 2)))    
##~~~~~~~
#model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
#model.add(Convolution2D(128, 5, 5, activation='relu', name='conv2_1', subsample=(1, 1), 
#                        init='glorot_uniform', dim_ordering='tf'))
#model.add(ZeroPadding2D((1, 1),  dim_ordering='tf'))
#model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2', subsample=(1, 1), 
#                        init='glorot_uniform', dim_ordering='tf'))
#model.add(ZeroPadding2D((1, 1),  dim_ordering='tf'))
#model.add(MaxPooling2D((2,2), strides=(2, 2)))                      
##~~~~~~
#model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
#model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1', subsample=(1, 1), 
#                        init='glorot_uniform', dim_ordering='tf'))
#model.add(ZeroPadding2D((1, 1),  dim_ordering='tf'))
#model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2', subsample=(1, 1), 
#                        init='glorot_uniform', dim_ordering='tf'))
#model.add(ZeroPadding2D((1, 1),  dim_ordering='tf'))
#model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3', subsample=(1, 1), 
#                        init='glorot_uniform', dim_ordering='tf'))
#model.add(MaxPooling2D((2,2), strides=(2, 2))) 
##~~~~~~
#model.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
#model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1', subsample=(1, 1), 
#                        init='glorot_uniform', dim_ordering='tf'))
#model.add(ZeroPadding2D((1, 1),  dim_ordering='tf'))
#model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2', subsample=(1, 1), 
#                        init='glorot_uniform', dim_ordering='tf'))
#model.add(ZeroPadding2D((1, 1),  dim_ordering='tf'))
#model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3', subsample=(1, 1), 
#                        init='glorot_uniform', dim_ordering='tf'))
#model.add(MaxPooling2D((2,2), strides=(2, 2)))
#
#model.add(Flatten(input_shape=model.output_shape[1:]))
#model.add(Dense(2048, W_regularizer=l2(0.0005), activation='relu', init='glorot_uniform'))
#model.add(Dropout(0.5))
##model.add(Dense(881, W_regularizer=l2(0.0005), activation='softmax', init='glorot_uniform'))
#model.add(Dense(3000, W_regularizer=l2(0.0005), activation='softmax', init='glorot_uniform'))

###################################################################################
# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True, decay=0.005),
              metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
#        rescale=1./255,
        dim_ordering='tf',
        featurewise_std_normalization=True)

test_datagen = ImageDataGenerator(
#        rescale=1./255,
        dim_ordering='tf',
        featurewise_std_normalization=True)

X_sample = read_data_gpds960()      
train_datagen.fit(X_sample)
test_datagen.fit(X_sample)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        seed=666,
        color_mode ='grayscale',
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        seed=666,
        color_mode='grayscale',
        class_mode='categorical')

# fine-tune the model
model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples)
