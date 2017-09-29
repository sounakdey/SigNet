"""
Created on Fri Oct 28 14:11:45 2016
@author: Anjan Dutta (adutta@cvc.uab.es), Sounak Dey (sdey@cvc.uab.es)
"""
###############################################################################
# Train a Siamese MLP on pairs of digits from the MNIST dataset.
# It follows Hadsell-et-al.'06 [1] by computing the Euclidean distance on the
# output of the shared network and by optimizing the contrastive loss (see paper
# for mode details).
# [1] "Dimensionality Reduction by Learning an Invariant Mapping"
#    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
# Gets to 99.5% test accuracy after 20 epochs.
# 3 seconds per epoch on a Titan X GPU
###############################################################################

from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import random
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import RMSprop
from keras import backend as K
import tensorflow
from tensorflow.python.ops import control_flow_ops 
from read_data_set import generate_data_gpds960
tensorflow.python.control_flow_ops = control_flow_ops

np.random.seed(1337)  # for reproducibility

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
#    margin = 1
#    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))
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
            z1, z2 = digit_indices[d][i], digit_indices[d][i+1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, 10)
            dn = (d + inc) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)    
    
def create_base_network( input_dim ):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    seq.add( Dense( 128, input_shape=( input_dim, ), activation='relu' ) )
    seq.add( Dropout( 0.1 ) )
    seq.add( Dense( 128, activation='relu' ) )
    seq.add( Dropout( 0.1 ) )
    seq.add( Dense( 128, activation='relu' ) )
    
    return seq

def create_base_network_mod( height, width ):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    
#    seq.add(ZeroPadding2D((1,1), input_shape=( 1, height, width )))    
    seq.add(Convolution2D(64, 3, 3,  activation='relu', border_mode='same', input_shape=(1, height, width), name='block1_conv1'))
    seq.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block1_pool', dim_ordering='th'))
    
#    seq.add(ZeroPadding2D((1,1)))
    seq.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1'))
    seq.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block2_pool', dim_ordering='th'))
    
#    seq.add(ZeroPadding2D((1,1)))
    seq.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1'))
    seq.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block3_pool', dim_ordering='th'))
    
#    seq.add(ZeroPadding2D((1,1)))
    seq.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1'))
    seq.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block4_pool', dim_ordering='th'))
    
    seq.add(Flatten(name='flatten'))
    
    seq.add(Dense(1024, activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(512, activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(256, activation='relu'))
    
    return seq


def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()

# the data, shuffled and split between train and test sets
#(X_train, y_train), (X_test, y_test) = mnist.load_data()
#X_train = X_train.reshape(60000, 784)
#X_test = X_test.reshape(10000, 784)
#X_train = X_train.astype('float32')
#X_test = X_test.astype('float32')
#X_train /= 255
#X_test /= 255

height = 30
width = 100
input_dim = height*width
n1 = 24
n2 = 30
batch_size = n1*(n1-1)/2+n1*n2
nb_epoch_train = 100

# create training+test positive and negative pairs
#digit_indices = [np.where(y_train == i)[0] for i in range(10)]
#tr_pairs, tr_y = create_pairs(X_train, digit_indices)

#digit_indices = [np.where(y_test == i)[0] for i in range(10)]
#te_pairs, te_y = create_pairs(X_test, digit_indices)

# network definition
base_network = create_base_network( input_dim )

input_a = Input( shape = ( input_dim, ))
input_b = Input( shape = ( input_dim, ))

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network( input_a )
processed_b = base_network( input_b )

distance = Lambda( euclidean_distance, output_shape = eucl_dist_output_shape )( [processed_a, processed_b] )

model = Model( input=[input_a, input_b], output=distance )

rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms)

# training

model.fit_generator( generate_data_gpds960( height, width, batch_size ), 
    samples_per_epoch = batch_size, nb_epoch = nb_epoch_train )

# testing on training set
acc = model.evaluate_generator( generate_data_gpds960( height, width, batch_size ),
    val_samples = batch_size )

print('* Accuracy on test set: %0.2f%%' % (100 * acc))

#    validation_data = None, class_weight = None, nb_worker = 1 )

#tr_acc = compute_accuracy( pred, labels[9:10] )
    
# testing on test set
#pred = model.predict_generator( generate_data_gpds960( height, width ), 
#    samples_per_epoch = 996, nb_epoch = nb_epoch, verbose = 2, callbacks = [],
#    validation_data = None, class_weight = None, nb_worker = 1 )

#te_acc = compute_accuracy( pred, labels[11:12] )

#print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
#print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
    
# train

#model.compile( loss=contrastive_loss, optimizer=SGD, metrics = ['accuracy'] )
#model.fit( [train_pairs[:, 0], train_pairs[:, 1]], train_labels,
#          validation_data=( [test_pairs[:, 0], test_pairs[:, 1]], test_labels ),
#          batch_size=32, nb_epoch=nb_epoch )

# compute final accuracy on training and test sets
#pred = model.predict( [ train_pairs[:, 0], train_pairs[:, 1] ] )
#tr_acc = compute_accuracy( pred, train_labels )
#pred = model.predict( [ test_pairs[:, 0], test_pairs[:, 1] ] )
#te_acc = compute_accuracy( pred, test_labels )
#
#print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
#print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))