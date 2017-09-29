# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 19:28:34 2016

@author: sounak
"""
from __future__ import absolute_import
from __future__ import print_function


'''Train a Siamese MLP on pairs of digits from the MNIST/Signature dataset.
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
import pandas as pd
#from keras.datasets import mnist
#from keras.datasets import signature                                            # signature
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda
from keras.optimizers import RMSprop
from keras import backend as K
from scipy import misc
import os
import tensorflow
from tensorflow.python.ops import control_flow_ops
tensorflow.python.control_flow_ops = control_flow_ops


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
            z1, z2 = digit_indices[d][i], digit_indices[d][i+1]
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


def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()
    
def parse_data_file_gpds960():

    data_file = '/home/adutta/Workspace/SignatureVerification/Datasets/GPDS960/GPDS960_pairs.txt'
    
    annotations_train = pd.read_table(data_file, sep=' ', header=None,names=['image1', 'image2', 'label'])
    small_hack = pd.read_table(data_file, sep='/', header=None,names=['writers', 'garbage1', 'garbage2'])
    filenames1 = list(annotations_train['image1'].values)
    filenames2 = list(annotations_train['image2'].values)
    labels = annotations_train['label'].values
    labels[labels==2]=0
    writers = list(small_hack['writers'].values)
    
    
    return filenames1, filenames2, labels, np.asarray(writers)
    
def generate_data_gpds960(filenames1, filenames2, labels, writers, num_of_writers, batch_size, height, width): 
    i = 0
    while 1:    
        image_dir = '/home/adutta/Workspace/SignatureVerification/Datasets/GPDS960/'    
        unique_writers = list( np.unique( writers ) )    
        for uw in unique_writers:
            indices = [ i for num, x in enumerate( writers ) if x == num_of_writers ]
            
            subset_filenames1 = [ filenames1[i] for i in indices ]
            subset_filenames2 = [ filenames2[i] for i in indices ]
            subset_labels = [ labels[i] for i in indices ]
            image_pairs = []
            label_pairs = []
            for f in range( 0, len( subset_filenames1 )):            
                file1 = subset_filenames1[f]
                file2 = subset_filenames2[f]
                img1 = np.transpose(np.reshape(np.invert( misc.imresize( misc.imread( image_dir + file1 ),
                                                [height, width] ) ), -1))
                img2 = np.transpose(np.reshape(np.invert( misc.imresize( misc.imread( image_dir + file2 ),
                                                [height, width] ) ), -1))          
                image_pairs += [[img1, img2]]
                label_pairs += subset_labels[f]
    
            yield( img1[i:i+batch_size], img2[i:i+batch_size], label_pairs[i:i+batch_size] )
#            if i+batch_size < samples_per_epoch:
#                i=0
#            else:
#                i+=batch_size

def signature_load_data(filenames1, filenames2, labels, writers):
    train_writers = 3
    test_writers = 3
    path = '/home/adutta/Workspace/SignatureVerification/Datasets/GPDS960/'
    index_train = np.asarray(np.where(writers <= train_writers))    
    index_test =  np.asarray(np.where((writers <= (train_writers+test_writers)) & (writers > train_writers)))
    pairs = []
    y = []
    
    for ind in index_train[0,:]:
        pair = []
        fname1 = os.path.join(path, filenames1[ind])
        fname2 = os.path.join(path, filenames2[ind])
        img1 = misc.imread(fname1)
        img_resized1 = misc.imresize(img1,(30, 100))
        img_reshaped1 = np.transpose(np.reshape(img_resized1, -1))
        pair.append(img_reshaped1)
        img2 = misc.imread(fname2)
        img_resized2 = misc.imresize(img2,(30, 100))
        img_reshaped2 = np.transpose(np.reshape(img_resized2, -1))
        pair.append(img_reshaped2)
        y.append(labels[ind])
        pairs.append(pair)
    tr_pairs = np.array(pairs)
    tr_y = np.array(y)
    pairs = []
    y = []
    for ind in index_test[0,:]:
        pair = []
        fname1 = os.path.join(path, filenames1[ind])
        fname2 = os.path.join(path, filenames2[ind])
        img1 = misc.imread(fname1)
        img_resized1 = misc.imresize(img1,(30, 100))
        img_reshaped1 = np.transpose(np.reshape(img_resized1, -1))
        pair.append(img_reshaped1)
        img2 = misc.imread(fname2)
        img_resized2 = misc.imresize(img2,(30, 100))
        img_reshaped2 = np.transpose(np.reshape(img_resized2, -1))
        pair.append(img_reshaped2)
        y.append(labels[ind])
        pairs.append(pair)
    te_pairs = np.array(pairs)
    te_y = np.array(y)
    tr_pairs = tr_pairs.astype('float32')
    te_pairs = te_pairs.astype('float32')
    tr_pairs /= 255
    te_pairs /= 255
    return tr_pairs, tr_y, te_pairs, te_y        
    
#def signature_load_data():
#    path = '/home/sounak/Documents/Datasets/GPDS960/'
#    train_writers = 1
#    test_writers = 1
#    with open('/home/sounak/Documents/Datasets/GPDS960_pairs.txt', 'r') as fp:
#        lines = fp.readlines()
#        
#    pairs = []
#    y = []
#    for nn, l in enumerate(lines):
#        #print nn
#        parts = l.split(' ')
#        if int(parts[0].split('/')[0]) <= train_writers:
#            pair = []
#            for num, part in enumerate(parts[:-1]):
#                
##                if int(parts[0].split('/')[0])<=99:
##                    part = part[1:]
#                
#                filename = os.path.join(path, part)
#                img = misc.imread(filename)
#                img_resized = misc.imresize(img,(300, 1000))
#                img_reshaped = np.transpose(np.reshape(img_resized, -1))
#                pair.append(img_reshaped)
#            pairs.append(pair)
#            y.append(int(parts[-1]))
#        
#    tr_pairs = np.array(pairs)
#    tr_y = np.array(y)
#    
#    
#    pairs = []
#    y = []
#    for nn, l in enumerate(lines):
#        #print nn
#        parts = l.split(' ')
#        if int(parts[0].split('/')[0]) <= train_writers+test_writers and int(parts[0].split('/')[0]) > train_writers:
#            pair = []
#            for num, part in enumerate(parts[:-1]):
#                
##                if int(parts[0].split('/')[0])<=99:
##                    part = part[1:]
#                filename = os.path.join(path, part)
#                img = misc.imread(filename)
#                img_resized = misc.imresize(img,(300, 1000))
#                img_reshaped = np.transpose(np.reshape(img_resized, -1))
#                pair.append(img_reshaped)
#            pairs.append(pair)
#            y.append(int(parts[-1]))
#        
#    te_pairs = np.array(pairs)
#    te_y = np.array(y)
#    
#    tr_pairs /= 255
#    te_pairs /= 255
#    
#    return tr_pairs, tr_y, te_pairs, te_y    


filenames1, filenames2, labels, writers = parse_data_file_gpds960()

#tr_pairs, tr_y = generate_data_gpds960(filenames1, filenames2, labels, writers, 300, 1000)
# the data, shuffled and split between train and test sets
tr_pairs, tr_y, te_pairs, te_y = signature_load_data(filenames1, filenames2, labels, writers)                          # for signature
#(X_train, y_train), (X_test, y_test) = mnist.load_data()
#X_train = X_train.reshape(60000, 784)
#X_test = X_test.reshape(10000, 784)
#X_train = X_train.astype('float32')
#X_test = X_test.astype('float32')
#X_train /= 255
#X_test /= 255
#input_dim = 784
height = 30
width = 100
input_dim = height*width                                                              # for signature
nb_epoch = 10
samples_per_epoch = 996
batch_size = 50

# configuring the memory usage in GPU
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.5)
#sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
#K.set_session(sess)


# create training+test positive and negative pairs
#digit_indices = [np.where(y_train == i)[0] for i in range(10)]
#tr_pairs, tr_y = create_pairs(X_train, digit_indices)

#digit_indices = [np.where(y_test == i)[0] for i in range(10)]
#te_pairs, te_y = create_pairs(X_test, digit_indices)

# network definition
base_network = create_base_network(input_dim)

input_a = Input(shape=(input_dim,))
input_b = Input(shape=(input_dim,))

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model(input=[input_a, input_b], output=distance)

# train
rms = RMSprop()
#sgd = SGD()
model.compile(loss=contrastive_loss, optimizer=rms)

#unique_writers = list(np.unique(writers)) 
#tr_writers = unique_writers[0:10]
#samples_per_epoch = (996*len(tr_writers))

#model.fit_generator(generate_data_gpds960(filenames1, filenames2, labels, writers, tr_writers, batch_size, height, width), samples_per_epoch, nb_epoch)
model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y),
          batch_size=50,
          nb_epoch=nb_epoch)

# compute final accuracy on training and test sets
#tr_writers = unique_writers[9:10]
#samples_per_epoch = (996*len(tr_writers))
#pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
##pred = model.predict_generator(generate_data_gpds960(filenames1, filenames2, labels, writers, tr_writers, batch_size, height, width), samples_per_epoch,  max_q_size=10, nb_worker=1, pickle_safe=False)
#tr_acc = compute_accuracy(pred, labels[9:10])
##pred = model.predict_generator(generate_data_gpds960(filenames1, filenames2, labels, writers, unique_writers[11:12], batch_size, height, width), samples_per_epoch=image_pairs.shape[0],  max_q_size=10, nb_worker=1, pickle_safe=False)
#pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
#te_acc = compute_accuracy(pred, labels[11:12])
pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = compute_accuracy(pred, tr_y)
pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
te_acc = compute_accuracy(pred, te_y)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))