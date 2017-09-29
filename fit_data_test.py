# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 12:42:20 2017

@author: sounak
"""

import numpy as np
import getpass as gp
from scipy import misc
from keras.preprocessing import image
    
def read_signature_data(dataset, ntuples, height = 30, width = 100):
    
    usr = gp.getuser()

    image_dir = '/home/' + usr + '/Workspace/SignatureVerification/Datasets/' + dataset + '/'
#    image_dir = '/home/' + usr + '/Workspace/Datasets/GPDS960_tot/'
#    image_dir = '/home/' + usr + '/Workspace/Datasets/GPDS960_tot/'
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
    
    
def read_data_gpds960():
    
    num_train_writers = 100
    idx_writers_range = list(range(100))

    idx_writers = sorted(np.random.choice(idx_writers_range, num_train_writers, replace=False))
    
    nsamples = 10; height = 155; width = 220 ;
    shape = (height, width, 1)
#    shape = -1
    usr = gp.getuser()
    size = 996
    dataset = 'GPDS960'
    image_dir = '/home/' + usr + '/Workspace/SignatureVerification/Datasets/' + dataset + '/'
    data_file = image_dir + dataset + '_pairs.txt'
    
    list_lines = []
    for iw in idx_writers:
        list_lines += list( range( iw * size, ( iw + 1 ) * size ) )
        
    f = open( data_file, 'r' )
    lines = f.readlines()
    f.close()
    
    lines = [ lines[i] for i in list_lines ]
    
    list_lines = []
    
    lp = []
    lin = []
    
    for iline, line in enumerate(lines):
        
        file1, file2, label = line.split(' ')
        
        label = int(label)
        
        lp += [label]        
        lin += [iline]
        
        if( len(lp) != 0 and len(lp) % size == 0 ):
                        
            idx1 = [ i for i, x in enumerate(lp) if x == 1 ]
            idx2 = [ i for i, x in enumerate(lp) if x == 0 ]
            
            idx1 = np.random.choice( idx1, nsamples )
            idx2 = np.random.choice( idx2, nsamples )
            
            idx = [None]*(len(idx1)+len(idx2))
            
            idx[::2] = idx1
            idx[1::2] = idx2
            
            del idx1
            del idx2
            
            list_lines += [lin[i] for i in idx]
            
            lp = []
            lin = []
            
            
    lines = [ lines[i] for i in list_lines ]
    
#    image_pairs = []
#    label_pairs = []
    images =[]
               
    for line in lines:
                    
        file1, file2, label = line.split(' ')
        img1 = np.reshape(misc.imresize( misc.imread( image_dir
                + file1 ), [ height, width ] ), shape)
                
#        img1 = np.reshape( np.tile( np.invert( misc.imresize( misc.imread( image_dir
#                + file1 ), [ height, width ] ) ), 3 ), shape )
        img1 = img1.astype('float32')    
#        img1 = img1 / np.max( img1 )        
#        img1 = img1.tolist()
        
        img2 = np.reshape( misc.imresize( misc.imread( image_dir
                + file1 ), [ height, width ] ), shape )
#        img2 = np.reshape( np.tile( np.invert( misc.imresize( misc.imread( image_dir
#                + file2 ), [ height, width ] ) ), 3 ), shape )
                
        img2 = img2.astype('float32')
#        img2 = img2 / np.max( img2 )        
#        img2 = img2.tolist()        
        
#        label = int(label)
    
#        image_pairs += [[ img1, img2 ]]
        images.append(img1)
        images.append(img2)
#        label_pairs += [ label ]
        
#    Editing for shuffling 
    return np.array(images)        

    
#def main():
#    dataset = 'GPDS960'
#    ntuples = 1000
#    x = read_signature_data(dataset, ntuples)
#    
#if __name__ == "__main__":
#    main()