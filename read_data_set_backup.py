# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 13:31:20 2016

@author: Anjan Dutta (adutta@cvc.uab.es), Sounak Dey (sdey@cvc.uab.es)
"""
import numpy as np
import pandas as pd
from scipy import misc
import random as rnd

def diff(first, second):
    
    second = set(second)
    return [item for item in first if item not in second]

def divide_train_test_set1( data_file, percent_train_set ):
    
    f1 = open( data_file, 'r' )
    lines = f1.readlines()
    f1.close()
    
    filenames = []
    labels = []
    writers = []
    
    for line in lines:
        filename, label = line.split( ' ' )
        filenames.append( filename )
        writer = filename.split('/')[0]
        writers.append( int( writer ) )
        labels.append( int( label ) )
    
    unique_labels = np.unique( labels )
    unique_writers = np.unique( writers )
    
    train_set = []
    test_set = []
    
    for uw in unique_writers:
        indices1 = [ i for i, x in enumerate( writers ) if x == uw ]
        labels_ = [ labels[i] for i in indices1 ]
        for ul in unique_labels:
            indices2 = [ i for i, x in enumerate( labels_ ) if x == ul ]
            indices = [ indices1[i] for i in indices2 ]            
            train_set.extend( rnd.sample( indices, int( len( indices )*percent_train_set ) ) )
            test_set.extend( diff( indices, train_set ) )
            
    train_set = sorted( train_set )
    test_set = sorted( test_set )
    
    train_images = [ filenames[i] for i in train_set ]
    test_images = [ filenames[i] for i in test_set ]
    train_labels = [ labels[i] for i in train_set ]
    test_labels = [ labels[i] for i in test_set ]
    
    return train_images, test_images, train_labels, test_labels
    
def divide_train_test_set2( data_file, percent_train_set, percent_test_set ):
    
    if percent_train_set + percent_test_set > 1.0:
        print( "Erroneous percent of train and test set." )
        exit()
    
    f1 = open( data_file, 'r' )
    lines = f1.readlines()
    f1.close()
    
    filenames1 = []
    filenames2 = []
    labels = []
    writers = []
    
    for line in lines:
        filename1, filename2, label = line.split( ' ' )
        filenames1.append( filename1 )
        filenames2.append( filename2 )
        writer = filename1.split('/')[0]
        writers.append( int( writer ) )
        labels.append( int( label ) )
    
    unique_writers = list( np.unique( writers ) )
    
    num_train_writers = int( len( unique_writers ) * percent_train_set )
    num_test_writers = int( len( unique_writers ) * percent_test_set )
    
##  Filter in the percent_train_set
    train_writers = rnd.sample( unique_writers, num_train_writers )
    all_writers_minus_train_writers = diff( unique_writers, train_writers )
    test_writers = rnd.sample( all_writers_minus_train_writers, num_test_writers )
    
    train_set = []
    
    for tw in train_writers:
        indices = [ i for i, x in enumerate( writers ) if x == tw ]
        train_set.extend( indices )
        
    test_set = []

    for tw in test_writers:
        indices = [ i for i, x in enumerate( writers ) if x == tw ]
        test_set.extend( indices )
            
    train_set = sorted( train_set )
    test_set = sorted( test_set )
    
    train_files1 = [ filenames1[i] for i in train_set ]
    train_files2 = [ filenames2[i] for i in train_set ]
    train_labels = [ labels[i] for i in train_set ]
    test_files1 = [ filenames1[i] for i in test_set ]
    test_files2 = [ filenames2[i] for i in test_set ]   
    test_labels = [ labels[i] for i in test_set ]    
    
    return (train_files1, train_files2, train_labels, test_files1, test_files2, test_labels)
    
def load_data_gpds960( percent_train_set, percent_test_set, height, width ):
    
    image_dir = '/home/anjan/Workspace/SignatureVerification/Datasets/GPDS960/'
    data_file = '/home/anjan/Workspace/SignatureVerification/Datasets/GPDS960_pairs.txt'
    
    (train_files1, train_files2, train_labels, test_files1, test_files2, test_labels) = divide_train_test_set2( data_file, percent_train_set, percent_test_set )
    num_train = len( train_labels )
    train_pairs = []
        
    for i in range( 0, num_train ):
        file1 = train_files1[i]
        file2 = train_files2[i]
        train_img1 = np.transpose( np.reshape( np.invert( misc.imresize(
            misc.imread( image_dir + file1 ), [height, width] ) ), -1 ) ) / 255
        train_img1 = train_img1.astype( 'float32' )
        train_img2 = np.transpose( np.reshape( np.invert( misc.imresize( 
            misc.imread( image_dir + file2 ), [height, width] ) ), -1 ) ) / 255
        train_img2 = train_img2.astype( 'float32' )
        train_pairs += [[ train_img1, train_img2 ]]
                
    train_pairs = np.array( train_pairs )
    train_labels = np.array( train_labels )
        
    num_test = len( test_labels )
    
    test_pairs = []
        
    for i in range( 0, num_test ):
        file1 = test_files1[i]
        file2 = test_files2[i]
        test_img1 = np.transpose( np.reshape( np.invert( misc.imresize(
            misc.imread( image_dir + file1 ), [height, width] ) ), -1 ) ) / 255
        test_img1 = test_img1.astype( 'float32' )
        test_img2 = np.transpose( np.reshape( np.invert( misc.imresize( 
            misc.imread( image_dir + file2 ), [height, width] ) ), -1 ) ) / 255
        test_img2 = test_img2.astype( 'float32' )        
        test_pairs += [[ test_img1, test_img2 ]]
    
    test_pairs = np.array( test_pairs )
    test_labels = np.array( test_labels )
        
    return train_pairs, train_labels, test_pairs, test_labels
    
def parse_data_file_gpds960():

    data_file = '/home/anjan/Workspace/SignatureVerification/Datasets/GPDS960_pairs.txt'    
    annotations_train = pd.read_table(data_file, sep=' ', header=None,names=['image1', 'image2', 'label'])
    small_hack = pd.read_table(data_file, sep='/', header=None,names=['writers', 'garbage1', 'garbage2'])
    filenames1 = list(annotations_train['image1'].values)
    filenames2 = list(annotations_train['image2'].values)
    labels = annotations_train['label'].values
    writers = list(small_hack['writers'].values)
            
    return filenames1, filenames2, labels, writers
    
#def generate_data_gpds960(filenames1, filenames2, labels, writers, num_of_writers, batch_size, height, width): 
#    i = 0
#    while 1:    
#        image_dir = '/home/sounak/Documents/Datasets/GPDS960/'    
#        unique_writers = list( np.unique( writers ) )    
#        for uw in unique_writers:
#            indices = [ i for i, x in enumerate( writers ) if x == num_of_writers ]
#            
#            subset_filenames1 = [ filenames1[i] for i in indices ]
#            subset_filenames2 = [ filenames2[i] for i in indices ]
#            subset_labels = [ labels[i] for i in indices ]
#            image_pairs = []
#            label_pairs = []
#            for f in range( 0, len( subset_filenames1 )):            
#                file1 = subset_filenames1[f]
#                file2 = subset_filenames2[f]
#                img1 = np.transpose(np.reshape(np.invert( misc.imresize( misc.imread( image_dir + file1 ),
#                                                [height, width] ) ), -1))
#                img2 = np.transpose(np.reshape(np.invert( misc.imresize( misc.imread( image_dir + file2 ),
#                                                [height, width] ) ), -1))          
#                image_pairs += [[img1, img2]]
#                label_pairs += subset_labels[f]
#    
#            yield( image_pairs[i:i+batch_size], label_pairs[i:i+batch_size] )
#            if i+batch_size > len(subset_filenames1):
#                i=0
#            else:
#                i+=batch_size
    
#def generate_data_gpds960(filenames1, filenames2, labels, writers, height, width):    
#    while True:        
#        image_dir = '/home/anjan/Workspace/SignatureVerification/Datasets/GPDS960/'
#        unique_writers = list( np.unique( writers ) )    
#        for uw in unique_writers:
#            indices = [ i for i, x in enumerate( writers ) if x == uw ]
#            subset_filenames1 = [ filenames1[i] for i in indices ]
#            subset_filenames2 = [ filenames2[i] for i in indices ]
#            subset_labels = [ labels[i] for i in indices ]
#            image_pairs = []
#            label_pairs = []
#            for f in range( 0, len( subset_filenames1 )):            
#                file1 = subset_filenames1[f]
#                file2 = subset_filenames2[f]
#                img1 = np.reshape( np.invert( misc.imresize( misc.imread( 
#                                image_dir + file1 ), [height, width] ) ), -1 )
#                img2 = np.reshape( np.invert( misc.imresize( misc.imread(
#                                image_dir + file2 ), [height, width] ) ), -1 )
#                image_pairs += [[img1, img2]]
#                label_pairs += [subset_labels[f]]
#    
#            yield( [np.array( image_pairs )[:,0], np.array( image_pairs )[:,1]], np.array( label_pairs ) )
#            yield( np.array( label_pairs ) )

def generate_data_gpds960( height, width, batch_size ):
    
    data_file = '/home/anjan/Workspace/SignatureVerification/Datasets/GPDS960_pairs.txt'
    image_dir = '/home/anjan/Workspace/SignatureVerification/Datasets/GPDS960/'
    
    f = open( data_file, 'r' )
    lines = f.readlines()
    f.close()    
    num_element = height*width
    
    line_count = 0
    
    image_pairs = []    
    label_pairs = []
    
    while True:
        for line in lines:            
            file1, file2, label = line.split( ' ' )
            img1 = np.reshape( np.invert( misc.imresize( misc.imread( image_dir
                    + file1 ), [height, width] ) ), -1 ) / 255
            img2 = np.reshape( np.invert( misc.imresize( misc.imread( image_dir
                    + file2 ), [height, width] ) ), -1 ) / 255
            label = int( label )
            
            image_pairs += [[img1, img2]]
            label_pairs += [label]
            
            line_count += 1
                            
            if( line_count % batch_size == 0):
                yield( [np.array( image_pairs )[:,0], np.array( image_pairs )[:,1]], np.array( label_pairs ) )
                image_pairs = []
                label_pairs = []               