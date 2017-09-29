# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 17:03:35 2017

@author: adutta
"""

import os
from scipy import misc
import numpy as np

if __name__ == "__main__":
    
    dir_path = '/home/adutta/Workspace/Datasets/Bengali'    
    tot_mean = 0
    tot_std = 0
    img_count = 0
    
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if os.path.join(root, file).endswith(('.jpg','.tif','.png','.bmp')):
                im = np.invert(misc.imread(os.path.join(root, file)))
                                
                tot_mean += np.mean(im)
                tot_std += np.std(im)
                img_count += 1
                
    print('The mean of the dataset is: %3f'%(tot_mean/img_count))
    print('The std of the dataset is: %3f'%(tot_std/img_count))
                
#                print(os.path.join(root, file))
        
        
#    for path, subdir, _ in os.walk(dir_path):
#        for isubdir in subdir:
#            for ifile in os.path.join(path,isubdir):
#                print(ifile)
#             
#            img = misc.imread(subfolders)
#            tot_mean+= np.mean(img)
#            tot_std+= np.std(img)
#            img_count += 1
#       
#    print('The mean of the dataset is: %3f'%(tot_mean/img_count))
#    print('The mean of the dataset is: %3f'%(tot_std/img_count))