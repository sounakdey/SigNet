# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 15:02:09 2017

@author: adutta
"""

from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.layers import Dense, Dropout, Input, Lambda
from keras.models import Model
from keras.optimizers import RMSprop, SGD, Adagrad, Adadelta, Adam
import numpy as np
from keras.applications.imagenet_utils import decode_predictions
from read_data_set import read_data_gpds960


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
#    return K.mean((1 - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0)))

def create_base_network1(input_dim):
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


base_model = VGG19(weights='imagenet',include_top=True)

# let's visualize layer names and layer indices:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)
   
model = Model(input=base_model.input, output=base_model.get_layer('block5_pool').output)



############################3 Loading the dataset #####################################
height=3
width=10
input_dim=height*width
n1=24 # number of genuine signatures per ID
n2=30 # number of forged signatures per ID
del n1, n2
batch_size=512
nb_epoch=10
input_shape=(height, width, 3)

num_train_writers = 1
num_test_writers = 1

print('Loading training data...', end="", flush=True)
idx_writers = list(range(4000))
idx_train_writers = sorted(np.random.choice(idx_writers, num_train_writers, replace=False))
train_pairs, train_labels = read_data_gpds960(idx_train_writers, height=height, width=width)
print('Done.')

print('Loading test data...', end="", flush=True)
idx_test_writers = sorted(np.random.choice([x for x in idx_writers if x not in idx_train_writers], num_test_writers, replace=False))
test_pairs, test_labels = read_data_gpds960(idx_test_writers, height=height, width=width)
#print('Done.')
#####################################################################################

feature_pairs = []
for pair in train_pairs:
    img_a = pair[0]
    img_b = pair[1]
    input_a = np.expand_dims(img_a, axis=0)
    input_a = preprocess_input(img_a)
    input_b = np.expand_dims(img_b, axis=0)
    input_b = preprocess_input(img_b)
    features_a = model.predict(input_a) # The size of the input image requires to (1,224,224,3) for the 
    features_b = model.predict(input_b) # the model to predict a feature. It is getting a (1, 3, 10, 3)
    feature_pairs += [[ features_a.ravel(), features_b.ravel() ]]
    input_dim = len(features_a.ravel())
#####################################################################################

# network definition
base_network = create_base_network1(input_dim)

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
model.compile(loss=contrastive_loss, optimizer=rms, metrics=['accuracy'])
model.fit([feature_pairs[:, 0], feature_pairs[:, 1]], train_labels,
          validation_data=([test_pairs[:, 0], test_pairs[:, 1]], test_labels),
          batch_size=128, nb_epoch=nb_epoch)

#img_path = '/home/adutta/Workspace/Datasets/GPDS960/0001/c-001-01.jpg'
#img = image.load_img(img_path, target_size=(224, 224))
#x = image.img_to_array(img)
#x = np.expand_dims(x, axis=0)
#x = preprocess_input(x)
#
#features = base_model.predict(x)
#print('Predicted:', decode_predictions(features, top=3)[0])