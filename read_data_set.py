# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 13:31:20 2016

@author: Anjan Dutta (adutta@cvc.uab.es), Sounak Dey (sdey@cvc.uab.es)
"""
import numpy as np
import getpass as gp
from scipy import misc
from keras.preprocessing import image


def generate_data_gpds960( height, width ):
       
    size = 996
    usr = gp.getuser()

    data_file = '/home/' + usr + '/Workspace/SignatureVerification/Datasets/GPDS960/GPDS960_pairs.txt'
    image_dir = '/home/' + usr + '/Workspace/SignatureVerification/Datasets/GPDS960/'
    
    f = open( data_file, 'r' )
    lines = f.readlines()
    f.close()    
    
    line_count = 0
    
    image_pairs = []
    label_pairs = []
    
    while True:
        for line in lines:            
            file1, file2, label = line.split(' ')
#            img1 = np.reshape( np.invert( misc.imresize( misc.imread( image_dir
#                    + file1 ), [height, width] ) ), -1 )
#            img1 = np.reshape(np.invert(misc.imresize(misc.imread(image_dir + 
#                                 file1), (height, width))), (1, height, width))
            img1 = np.invert(misc.imresize(misc.imread(image_dir + file1), 
                                           (height, width)))
            img1 = img1 / np.max( img1 )
#            img2 = np.reshape( np.invert( misc.imresize( misc.imread( image_dir
#                    + file2 ), [height, width] ) ), -1 )
            img2 = np.invert(misc.imresize(misc.imread(image_dir + file2), 
                                           (height, width)))
            img2 = img2 / np.max(img2)
            label = int(label)
            
#            print('%d' % (label))
            
            image_pairs += [[img1, img2]]
            label_pairs += [label]
            
            line_count += 1
                            
            if(line_count % size == 0):
                idx1 = [i for i, x in enumerate(label_pairs) if x == 1]
                idx2 = [i for i, x in enumerate(label_pairs) if x == 0]
#                idx1 = np.where(label_pairs == 1)[0]
#                idx2 = np.where(label_pairs == 0)[0]
                nones = len(idx2)
#                print('nzeros: %d' % len(idx2))
#                print('nones: %d' % len(idx1))
                idx2 = np.random.choice(idx2, nones)
                idx = np.concatenate((idx1, idx2), axis = 0)
                yield( [np.array(image_pairs)[idx,0], np.array(image_pairs)[idx,1]], 
                        np.array(label_pairs)[idx,])
                image_pairs = []
                label_pairs = []
                
def read_data_gpds960( idx_writers, shape, nsamples = 10, height = 30, width = 100 ):
    
#    shape = (height, width, 1)
#    shape = -1
    usr = gp.getuser()
    size = 996
    dataset = 'GPDS960'
    image_dir = '/home/' + usr + '/Workspace/Datasets/' + dataset + '/'
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
    
    image_pairs = []
    label_pairs = []
               
    for line in lines:
                    
        file1, file2, label = line.split(' ')
        img1 = np.reshape( np.invert( misc.imresize( misc.imread( image_dir
                + file1 ), [ height, width ] ) ), shape )
                
#        img1 = np.reshape( np.tile( np.invert( misc.imresize( misc.imread( image_dir
#                + file1 ), [ height, width ] ) ), 3 ), shape )
        img1 = img1.astype('float16')    
        img1 = img1 / np.max( img1 )        
        img1 = img1.tolist()
        
        img2 = np.reshape( np.invert( misc.imresize( misc.imread( image_dir
                + file1 ), [ height, width ] ) ), shape )
#        img2 = np.reshape( np.tile( np.invert( misc.imresize( misc.imread( image_dir
#                + file2 ), [ height, width ] ) ), 3 ), shape )
                
        img2 = img2.astype('float16')
        img2 = img2 / np.max( img2 )        
        img2 = img2.tolist()        
        
        label = int(label)
    
        image_pairs += [[ img1, img2 ]]
        label_pairs += [ label ]
        
#    Editing for shuffling 
            
    return np.array( image_pairs ), np.array( label_pairs )
    
def read_data_cedar( idx_writers, shape, nsamples = 20, height = 30, width = 100 ):
    
#    shape = (height, width, 1)
#    shape = -1
    usr = gp.getuser()
    size = 852
    dataset = 'CEDAR1'
    image_dir = '/home/' + usr + '/Workspace/Datasets/' + dataset + '/'
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
    
    image_pairs = []
    label_pairs = []
               
    for line in lines:
        
        print(line)
                    
        file1, file2, label = line.split(' ')
        
        img1 = np.reshape(np.invert(image.load_img(image_dir+file1, 
                                        target_size=(height,width))), shape)
                        
        img1 = img1.astype('float16')    
        img1 = img1 / np.max( img1 )        
        img1 = img1.tolist()
        
        img2 = np.reshape(np.invert(image.load_img(image_dir+file2, 
                                        target_size=(height,width))), shape)
                
        img2 = img2.astype('float16')
        img2 = img2 / np.max( img2 )        
        img2 = img2.tolist()        
        
        label = int(label)
    
        image_pairs += [[ img1, img2 ]]
        label_pairs += [ label ]
        
#    Editing for shuffling 
            
    return np.array( image_pairs ), np.array( label_pairs )

def trim(im):
#    global_thresh = threshold_otsu(im)
    global_thresh = 200
    binary_global = im > global_thresh
    
    ind_hor = np.nonzero(np.sum(binary_global == 0, axis = 1))
    x_min = np.min(ind_hor) - 5
    x_max = np.max(ind_hor) + 5
        
    ind_ver = np.nonzero(np.sum(binary_global == 0, axis = 0))
    y_min = np.min(ind_ver) - 5
    y_max = np.max(ind_ver) + 5
        
    return im[x_min:x_max,y_min:y_max]