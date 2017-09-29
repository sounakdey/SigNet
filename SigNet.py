# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 15:45:17 2017

@author: Anjan Dutta
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 12:24:27 2017

@author: adutta
"""

import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.layers import Dense, Dropout, Input, Lambda, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.preprocessing import image
from keras import backend as K
from SignatureDataGenerator import SignatureDataGenerator
import getpass as gp
from keras.models import Sequential, Model
from keras.optimizers import SGD, RMSprop
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import random
random.seed(1337)

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

def create_base_network1(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    seq.add(Flatten(name='flatten', input_shape=input_shape))
    seq.add(Dense(128, activation='relu'))
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
    seq.add(Dropout(0.3))
    seq.add(Convolution2D(50, 5, 5, activation='relu', border_mode='same', name='block2_conv1'))
    seq.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block2_pool', dim_ordering='tf'))
    seq.add(Dropout(0.3))
    seq.add(Flatten(name='flatten'))
    seq.add(Dense(512,  W_regularizer=l2(0.0005), activation='relu', name='fc1'))
    seq.add(Dropout(0.5))
    seq.add(Dense(256, W_regularizer=l2(0.0005), activation='relu', name='fc2'))
    seq.add(Dropout(0.5))
    seq.add(Dense(256,  W_regularizer=l2(0.0005),activation='relu', name='fc3'))
    
    return seq
    
def create_base_network3(input_shape):    
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    seq.add(Convolution2D(20, 11, 11,  activation='relu', border_mode='same', input_shape=input_shape, name='block1_conv1', subsample=(1, 1), init='glorot_uniform'))
    seq.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block1_pool', dim_ordering='tf'))
    
    seq.add(Convolution2D(50, 5, 5, activation='relu', border_mode='same', name='block2_conv1', subsample=(1, 1), init='glorot_uniform'))
    seq.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block2_pool', dim_ordering='tf'))
    
    seq.add(Flatten(name='flatten'))
    seq.add(Dense(128,  W_regularizer=l2(0.0005), activation='relu', name='fc1'))
    seq.add(Dropout(0.1))
    seq.add(Dense(128, W_regularizer=l2(0.0005), activation='relu', name='fc2'))
    seq.add(Dropout(0.1))
    seq.add(Dense(128, W_regularizer=l2(0.0005), activation='relu', name='fc3'))
    
    return seq
    
def create_base_network_ijcnn(input_shape):
    
    seq = Sequential()
    seq.add(Convolution2D(96, 11, 11, activation='relu', name='conv1_1', subsample=(4, 4),input_shape= input_shape, 
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
    
def create_base_network_vgg16(input_shape):

    seq = Sequential()
    seq.add(ZeroPadding2D((1,1),input_shape=input_shape))
    seq.add(Convolution2D(64, 3, 3, activation='relu'))
    seq.add(ZeroPadding2D((1,1)))
    seq.add(Convolution2D(64, 3, 3, activation='relu'))
    seq.add(MaxPooling2D((2,2), strides=(2,2)))

    seq.add(ZeroPadding2D((1,1)))
    seq.add(Convolution2D(128, 3, 3, activation='relu'))
    seq.add(ZeroPadding2D((1,1)))
    seq.add(Convolution2D(128, 3, 3, activation='relu'))
    seq.add(MaxPooling2D((2,2), strides=(2,2)))

    seq.add(ZeroPadding2D((1,1)))
    seq.add(Convolution2D(256, 3, 3, activation='relu'))
    seq.add(ZeroPadding2D((1,1)))
    seq.add(Convolution2D(256, 3, 3, activation='relu'))
    seq.add(ZeroPadding2D((1,1)))
    seq.add(Convolution2D(256, 3, 3, activation='relu'))
    seq.add(MaxPooling2D((2,2), strides=(2,2)))

    seq.add(ZeroPadding2D((1,1)))
    seq.add(Convolution2D(512, 3, 3, activation='relu'))
    seq.add(ZeroPadding2D((1,1)))
    seq.add(Convolution2D(512, 3, 3, activation='relu'))
    seq.add(ZeroPadding2D((1,1)))
    seq.add(Convolution2D(512, 3, 3, activation='relu'))
    seq.add(MaxPooling2D((2,2), strides=(2,2)))

    seq.add(ZeroPadding2D((1,1)))
    seq.add(Convolution2D(512, 3, 3, activation='relu'))
    seq.add(ZeroPadding2D((1,1)))
    seq.add(Convolution2D(512, 3, 3, activation='relu'))
    seq.add(ZeroPadding2D((1,1)))
    seq.add(Convolution2D(512, 3, 3, activation='relu'))
    seq.add(MaxPooling2D((2,2), strides=(2,2)))

    seq.add(Flatten())
    seq.add(Dense(4096, activation='relu'))
    seq.add(Dropout(0.5))
    seq.add(Dense(4096, activation='relu'))
    seq.add(Dropout(0.5))
    seq.add(Dense(1000, activation='relu'))
    
    return seq

def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    predictions = (predictions - np.min(predictions)) / (np.max(predictions) -np.min(predictions))
    predictions = (predictions < 0.5) * 1.0
    return float(np.sum(labels==predictions.ravel()))/len(labels)
    
def compute_accuracy(predictions, labels):
   '''Compute classification accuracy with a fixed threshold on distances.
   '''
   labels=labels.tolist()
   labels_same = [i for i,x in enumerate(labels) if x==1] 
   labels_diff = [i for i,x in enumerate(labels) if x==0]     
   predict_same = predictions[labels_same]
   predict_diff = predictions[labels_diff]
   TA=[]
   FA=[]
   
   for d in np.arange(min(predictions),max(predictions),0.1):
       x=float(np.sum(predict_same<=d))/float(len(labels_same))
       y=float(np.sum(predict_diff<=d))/float(len(labels_diff))
#        print x,y,d
       TA.append(x)
       FA.append(y)

   return (TA, FA)
    
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
   
def compute_accuracy_roc2(predictions, labels):
   '''Compute ROC accuracy with a fixed threshold on distances.
   '''
   dmax = np.max(predictions)
   dmin = np.min(predictions)
   print ('ROC2', dmax, dmin)
   step = 0.1
   max_val = 0
   max_d = float('inf')
   for d in np.arange(dmin, dmax+step, step):        
       tpr = labels[predictions.ravel() <= d].mean()
       print ('ROC2', tpr)
       if (tpr > max_val):
           max_val = tpr
           max_d = d    
   return labels[predictions.ravel() < max_d].mean() 
    
def read_signature_data(dataset, ntuples, height = 30, width = 100):
    
    usr = gp.getuser()

    image_dir = '/home/' + usr + '/Workspace/Datasets/Signatures_Inverted/' + dataset + '/'
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
        
def main():
    
    # dataset specifications
    dataset = 'CEDAR1'
    
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
        num_train_writers = 200
        num_valid_writers = 10
        
    elif dataset == 'GPDS960':
    
        tot_writers = 4000
        num_train_writers = 1950
        num_valid_writers = 10
        
    elif dataset == 'CEDAR1':
    
        tot_writers = 55
        num_train_writers = 45
        num_valid_writers = 5
    
    num_test_writers = tot_writers - (num_train_writers + num_valid_writers)
    
    # parameters
    batch_sz = 128
    nsamples = 8 
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
    X_sample = read_signature_data(dataset, round(0.5*tot_writers), height=img_height, width=img_width)
    datagen.fit(X_sample)
    del X_sample
    
    # network definition
    base_network = create_base_network_ijcnn(input_shape)
    
    input_a = Input(shape=(input_shape))
    input_b = Input(shape=(input_shape))
    
    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    
    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    
    model = Model(input=[input_a, input_b], output=distance)
    
    # train model
    rms = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-08)
    model.compile(loss=contrastive_loss, optimizer=rms)
        
    model.fit_generator(generator=datagen.next_train(), samples_per_epoch=datagen.samples_per_train, nb_epoch=nb_epoch, validation_data=datagen.next_valid(), nb_val_samples=int(datagen.samples_per_valid))
        
#    model.fit_generator(generator=datagen.next_train(), samples_per_epoch=datagen.samples_per_train, nb_epoch=nb_epoch)

    tr_pred = model.predict_generator(generator=datagen.next_train(), val_samples=int(datagen.samples_per_train))
    te_pred = model.predict_generator(generator=datagen.next_test(), val_samples=int(datagen.samples_per_test))
    
    tr_acc = compute_accuracy(tr_pred, datagen.train_labels)
    te_acc = compute_accuracy(te_pred, datagen.test_labels)
    
    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
    
if __name__ == "__main__":
    main()
