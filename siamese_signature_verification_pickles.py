"""
Created on Fri Oct 28 14:11:45 2016
@author: Anjan Dutta (adutta@cvc.uab.es), Sounak Dey (sdey@cvc.uab.es)
"""
###############################################################################
# Train a Siamese MLP on pairs of signatures from the GPDS960 dataset.
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
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop, SGD, Adagrad, Adadelta, Adam
from keras import backend as K
from keras.applications.vgg16 import VGG16
import tensorflow
from tensorflow.python.ops import control_flow_ops
tensorflow.python.control_flow_ops = control_flow_ops
from load_signature_pickles import read_pickles

from sklearn.svm import SVC
from sklearn.metrics import  accuracy_score
from sklearn.preprocessing import normalize

np.random.seed(1337)  # for reproducibility

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def euclidean_distance_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)
    
def cosine_distance(vects):
    x, y = vects
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return -K.mean(x * y, axis=-1, keepdims=True)

def cosine_distance_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)
    
def absolute_difference(vectors):
    x, y = vectors
    return abs(x - y)
    
def absolute_difference_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))  
#    return K.mean((1 - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0)))
    
def create_base_network1( input_dim ):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    seq.add( Dense( 1024, input_shape=( input_dim, ), activation='relu' ) )
    seq.add( Dropout( 0.2 ) )
    seq.add( Dense( 1024, activation='tanh' ) )
    seq.add( Dropout( 0.2 ) )
    seq.add( Dense( 1024, activation='tanh' ) )
    seq.add( Dropout( 0.2 ) )
    seq.add( Dense( 512, activation='tanh' ) )
    
    return seq

def create_base_network2(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    
    seq.add(Convolution2D(20, 5, 5,  activation='relu', border_mode='same', input_shape=input_shape, name='block1_conv1'))
    seq.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block1_pool', dim_ordering='th'))
    
    seq.add(Convolution2D(50, 5, 5, activation='relu', border_mode='same', name='block2_conv1'))
    seq.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block2_pool', dim_ordering='th'))
    
#    seq.add(Convolution2D(128, 4, 4, activation='relu', border_mode='same', name='block3_conv1'))
#    seq.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block3_pool', dim_ordering='th'))        
    
#    seq.add(Convolution2D(256, 4, 4, activation='relu', border_mode='same', name='block4_conv1'))
#    seq.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block4_pool', dim_ordering='th'))
    
#    seq.add(Dropout(0.1))
       
    seq.add(Flatten(name='flatten'))
    
    seq.add(Dense(256, activation='relu', name='fc1'))
#    seq.add(Dropout(0.1))
    seq.add(Dense(128, activation='relu', name='fc2'))
#    seq.add(Dropout(0.1))
    seq.add(Dense(128, activation='relu', name='fc3'))
    
    return seq
    
def create_base_network3(input_shape):
    
    base_model = VGG16(weights='imagenet')
    model_vgg16_conv = Model(input=base_model.input, output=base_model.get_layer('block2_pool').output)

    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
#    model_vgg16_conv.summary()
    
    input = Input(shape=input_shape, name = 'image_input')
    
    output_vgg16_conv = model_vgg16_conv(input)
    
    #Add the fully-connected layers 
    seq = Flatten(name='flatten')(output_vgg16_conv)
#    seq = Dense(4096, activation='relu', name='fc1')(seq)
#    seq = Dense(4096, activation='relu', name='fc2')(seq)
#    seq = Dense(1000, activation='relu', name='fc3')(seq)
    
    return seq    
    
def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()

# some parameters
height=30
width=100
input_dim=25088
n1=24 # number of genuine signatures per ID
n2=30 # number of forged signatures per ID
del n1, n2
batch_size=512
nb_epoch=10
input_shape=(height, width, 1)

num_train_writers = 10
num_test_writers = 10

# network definition 1
base_network = create_base_network1( input_dim )
input_a = Input( shape = ( input_dim, ))
input_b = Input( shape = ( input_dim, ))

# network definition 2
#base_network=create_base_network2(input_shape)
#input_a=Input(shape=input_shape)
#input_b=Input(shape=input_shape)

# network definition 3
#base_network=create_base_network3(input_shape)
#input_a=Input(shape=input_shape)
#input_b=Input(shape=input_shape)

# because we re-use the same instance `base_network`, the weights of the network
# will be shared across the two branches
processed_a=base_network(input_a)
processed_b=base_network(input_b)

# distance measure
distance=Lambda(euclidean_distance, output_shape=
                euclidean_distance_output_shape)([processed_a, processed_b])

# merging the input and getting the distance
model=Model(input=[input_a, input_b], output=distance)

# optimizer
#sgd=SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
#rms = RMSprop()
rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
#adagrad=Adagrad(lr=0.001, epsilon=1e-08, decay=0.0)
#adadelta=Adadelta()
#adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#model.compile(optimizer=rms, loss=contrastive_loss)
#model.compile(optimizer=rms, loss=categorical_crossentropy)
model.compile(optimizer=rms, loss=contrastive_loss)

num_train_writers = 2
num_test_writers = 2

idx_writers = list(range(288))

print('Loading training data...', end="", flush=True)
       
idx_train_writers = sorted(np.random.choice(idx_writers, num_train_writers, replace=False))
train_pairs, train_labels = read_pickles(idx_train_writers)

abs_diff = []

for i in range(len(train_pairs)):
    fi1 = train_pairs[i][0]
    fi2 = train_pairs[i][1]
    
    fi1 = fi1 / sum(fi1)
    fi2 = fi2 / sum(fi2)
    
    abs_diff += [sum(abs(fi1 - fi2))]
    
    

print('Done.')

print("Normalizing the features...")
train_pairs = normalize(train_pairs, norm="l2", axis=2)
print("Done")

print('Loading test data...', end="", flush=True)

idx_test_writers = sorted(np.random.choice([x for x in idx_writers if x not in idx_train_writers], num_test_writers, replace=False))
test_pairs, test_labels = read_pickles(idx_test_writers)

print('Done.')

# training with batch generator
#model.fit_generator(generate_data_gpds960(height, width), 
#    samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch, verbose=1,
#    validation_data=validation_data)

# validation_data

#print('Training SVM')
#clf = SVC()
#clf.fit( , train_labels)
#SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
#    max_iter=-1, probability=False, random_state=None, shrinking=True,
#    tol=0.001, verbose=False)
#    
#print('Testing SVM')
#
#pred_labels = clf.predict()

#print(accuracy_score(test_labels, pred_labels))


print('Training...')

#model.fit([train_pairs[:,0], train_pairs[0,1]], train_labels,          
#         validation_data=(test_pairs, test_labels),
#         batch_size=batch_size, nb_epoch=nb_epoch)

model.fit([train_pairs[:, 0], train_pairs[:, 1]], train_labels,
          validation_data=([test_pairs[:, 0], test_pairs[:, 1]], test_labels),
          batch_size=batch_size, nb_epoch=nb_epoch)
          
print('Training done.')

# compute final accuracy on training and test sets
print('Testing...', end="", flush=True)
pred = model.predict([train_pairs[:, 0], train_pairs[:, 1]])
tr_acc = compute_accuracy(pred, train_labels)
pred = model.predict([test_pairs[:, 0], test_pairs[:, 1]])
te_acc = compute_accuracy(pred, test_labels)
print('Done.')

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))