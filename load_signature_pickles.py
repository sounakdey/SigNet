# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 13:11:10 2017

@author: adutta
"""

import os
import pickle
import numpy as np

def read_pickles(idx):

    dir_pickles = '/home/adutta/Workspace/Datasets/GPDS960/pickles'
    
    files_pickles = os.listdir(dir_pickles)
    
    x_ = []
    l_ = []
    
    for i in idx:
        
        f = files_pickles[i]
        
        try:
            
            with open(os.path.join(dir_pickles, f), 'rb') as fp:
                features_pairs, label_pairs = pickle.load(fp)
        #        unpickler = pickle.Unpickler(fp)
        #        features_pairs, label_pairs = unpickler.load(fp)
                
            label_pairs = [int(s) for s in label_pairs]
            
            idx1 = [ i for i, x in enumerate(label_pairs) if x == 1 ]
            idx2 = [ i for i, x in enumerate(label_pairs) if x == 0 ]
            
            nones = len(idx1)
            idx2 = np.random.choice( idx2, nones )
                    
            idx = [None]*(len(idx1)+len(idx2))
                    
            idx[::2] = idx1
            idx[1::2] = idx2
            
            del idx1
            del idx2
            
            x_ += [features_pairs[i] for i in idx]
            l_ += [label_pairs[i] for i in idx]
        
        except Exception:
            print(f)
            pass   
        
    return np.asarray(x_,dtype=np.float64), np.asarray(l_,dtype=np.int64)
 
#def main_read():
#    
#    num_train_writers = 5
#    num_test_writers = 5
#    idx_writers = list(range(288))
#           
#    idx_train_writers = sorted(np.random.choice(idx_writers, num_train_writers, replace=False))
#    idx_test_writers = sorted(np.random.choice([x for x in idx_writers if x not in idx_train_writers], num_test_writers, replace=False))
#    
#    x_tr, l_tr = read_pickles(idx_train_writers)
#    x_te, l_te = read_pickles(idx_test_writers)



