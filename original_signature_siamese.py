# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 13:27:43 2017

@author: sounak
"""
from __future__ import absolute_import
from __future__ import print_function
'''Train a Siamese MLP on pairs of digits from the MNIST dataset.
It follows Hadsell-et-al.'06 [1] by computing the Euclidean distance on the
output of the shared network and by optimizing the contrastive loss (see paper
for mode details).
[1] "Dimensionality Reduction by Learning an Invariant Mapping"
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
Gets to 99.5% test accuracy after 20 epochs.
3 seconds per epoch on a Titan X GPU
'''

import numpy as np
np.random.seed(1337)  # for reproducibility

import random

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda, Flatten
from keras.optimizers import RMSprop
from keras import backend as K
from keras.regularizers import l2
import tensorflow as tf

from keras.layers import Convolution2D, MaxPooling2D
import signature_data as sd


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


def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(10)]) - 1
    for d in range(10):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, 10)
            dn = (d + inc) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


def create_base_network(input_dim):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    seq.add(Dense(128, input_shape=(input_dim,), activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(128, activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(128, activation='relu'))
    return seq

def create_base_network2(input_shape):
    
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    seq.add(Convolution2D(20, 5, 5,  activation='relu', border_mode='same', input_shape=input_shape, name='block1_conv1'))
    seq.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block1_pool', dim_ordering='tf'))
    
    seq.add(Convolution2D(50, 5, 5, activation='relu', border_mode='same', name='block2_conv1'))
    seq.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block2_pool', dim_ordering='tf'))
    
    seq.add(Flatten(name='flatten'))
    seq.add(Dense(512,  W_regularizer=l2(0.0005), activation='relu', name='fc1'))
    seq.add(Dropout(0.3))
    seq.add(Dense(256, W_regularizer=l2(0.0005), activation='relu', name='fc2'))
    seq.add(Dropout(0.3))
    seq.add(Dense(128,  W_regularizer=l2(0.0005),activation='relu', name='fc3'))
    
    return seq
    
def create_base_network3(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
           
    seq.add(Flatten(name='flatten', input_shape=input_shape))
    
    seq.add(Dense(128,  activation='relu', name='fc1'))
    seq.add(Dropout(0.1))
    seq.add(Dense(128, activation='relu', name='fc2'))
    seq.add(Dropout(0.1))
    seq.add(Dense(128, activation='relu', name='fc3'))
    
    return seq

def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()

def main():
    
    # Create a session for running Ops on the Graph.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)
    # the data, shuffled and split between train and test sets
#    (X_train, y_train), (X_test, y_test) = mnist.load_data()
#    X_train = X_train.reshape(60000, 784)
#    X_test = X_test.reshape(10000, 784)
#    X_train = X_train.astype('float32')
#    X_test = X_test.astype('float32')
#    X_train /= 255
#    X_test /= 255
#    input_dim = 784
    width, height = 220, 155
    nb_epoch = 20
    input_shape = (height, width, 1)
    
    # create training+test positive and negative pairs
#    digit_indices = [np.where(y_train == i)[0] for i in range(10)]
#    tr_pairs, tr_y = create_pairs(X_train, digit_indices)
#    
#    digit_indices = [np.where(y_test == i)[0] for i in range(10)]
#    te_pairs, te_y = create_pairs(X_test, digit_indices)
    
    tr_pairs, tr_y, te_pairs, te_y = sd.load_data()
    
    # network definition
    base_network = create_base_network2(input_shape)
    
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    
    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    
    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    
    model = Model(input=[input_a, input_b], output=distance)
    
    # train
    rms = RMSprop(lr = 0.0001, decay = 1e-4)
    model.compile(loss=contrastive_loss, optimizer=rms)
    model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
              validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y),
              batch_size=128,
              nb_epoch=nb_epoch)
    
    # compute final accuracy on training and test sets
    pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    tr_acc = compute_accuracy(pred, tr_y)
    pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
    te_acc = compute_accuracy(pred, te_y)
    
    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
    
if __name__ == "__main__":
    main()