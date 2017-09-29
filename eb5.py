from datasets.Esposalles_bigrams import EsposallesDataset,prevlabels
from layers.SpatialPyramidPooling import SPP
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.layers.core import Dense,Activation,Dropout
from keras.utils import np_utils
from keras.layers import merge,Input
from keras.models import Model
import numpy as np
import re

def buildModel():
    inputimage=Input(shape=(1,50,50))
    inputlabel=Input(shape=(7,))
    x=Convolution2D(32,3,3,border_mode='same',input_shape=(1,50,50),activation='relu')(inputimage)
    x=Convolution2D(32,3,3,border_mode='same',activation='relu')(x)
    x=MaxPooling2D(pool_size=(2,2))(x)
    x=Convolution2D(64,3,3,border_mode='same',activation='relu')(x)
    x=Convolution2D(64,3,3,border_mode='same',activation='relu')(x)
    x=MaxPooling2D(pool_size=(2,2))(x)
    x=Convolution2D(128,3,3,border_mode='same',activation='relu')(x)
    x=Convolution2D(128,3,3,border_mode='same',activation='relu')(x)
    x=Convolution2D(256,3,3,border_mode='same',activation='relu')(x)
    x=SPP([(4,4),(2,2),(1,1)])(x)
    y=Dropout(0.5)(inputlabel)
    y=Dense(64,activation='relu')(y)
    x = merge([x,y], mode='concat')
    x=Dense(2048,activation='relu')(x)
    x=Dropout(0.5)(x)
    x=Dense(512,activation='relu')(x)
    x=Dropout(0.5)(x)
    outputs=Dense(6,activation='softmax')(x)
    m = Model(input=[inputimage, inputlabel], output=[outputs])
    optimizer=SGD(lr=0.0001,momentum=0.9,nesterov=True,decay=0.000001)
    m.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['acc'])

    return m

def trainModel(m,prefix):
    E=EsposallesDataset('train_regs.txt')
    for iters in xrange(100):
        accs=[]
        losses=[]
        for j in xrange (len(E.shuffled_examples)):
           x,y,z=E.get_example()
           y=np_utils.to_categorical(y,6)
           z=np_utils.to_categorical([prevlabels[z]],7)
           l,a=m.train_on_batch([x,z],y,class_weight=E.class_weights)
           accs.append(a)
           losses.append(l)
        print np.mean(losses),np.mean(accs)
        m.save_weights('./saved_weights/'+prefix+'Esposalles.h5',overwrite=True)
    
def evaluateModel(m,prefix,true_previous_label=False):
    E=EsposallesDataset('test_regs.txt')
    m.load_weights('./saved_weights/'+prefix+'Esposalles.h5')
    accs=[]
    losses=[]
    confmat=np.zeros((6,6),dtype='int32')
    previous_label=np.asarray([[0.,0.,0.,0.,0.,0.,1.]],dtype='float32') #Start register
    prevpag,pag,prevreg,reg=(0,0,0,0)
    for j in xrange (len(E.shuffled_examples)):
        x,y,z=E.get_example();
        if true_previous_label:
           previous_label=np_utils.to_categorical([prevlabels[z]],7)
        else:
             match=re.search('^[^/]*/Volum_069_Registres_([0-9]{4})_Reg([0-9]{2})_Line([0-9]{2})_Pos([0-9]{2})$',z);
             pag,reg,lin,pos=int(match.group(1)),int(match.group(2)),int(match.group(3)),int(match.group(4))
             if prevpag != pag or prevreg != reg:
                previous_lab2el=np.asarray([[0.,0.,0.,0.,0.,0.,1.]],dtype='float32') #Start register
        predicted_label=m.predict([x,previous_label],verbose=0)
        confmat[np.argmax(predicted_label),y[0]]+=1
        y=np_utils.to_categorical(y,6)
        l,a=m.evaluate([x,previous_label],y,verbose=0)
        accs.append(a)
        losses.append(l)
        if not true_previous_label:
           previous_label=np.hstack((predicted_label==np.max(predicted_label),[[0.]]))
           prevpag,prevreg=pag,reg
    print 'accuracy,loss:',np.mean(accs),np.mean(losses)
    print 'confmat:'
    print confmat

m=buildModel()
trainModel(m,'eb5')
evaluateModel(m,'eb5',True)
evaluateModel(m,'eb5',False)
