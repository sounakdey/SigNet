#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 14:15:19 2016

@author: Anjan Dutta (adutta@cvc.uab.es)
"""
import getpass as gp
import os
import glob

usr = gp.getuser()

#dataset = 'CEDAR1'
#ext = '*_.tif'
#
#image_dir = '/home/' + usr + '/Workspace/Datasets/' + dataset + '/'
#data_file = image_dir + dataset + '_pairs.txt'
#
#fp = open( data_file, 'w' )
#
## For CEDAR
#
#idx_signers=list(range(55))
#
#for isigner in idx_signers:
#    ext = '*_'+str(isigner+1)+'_*.png'
#    file_names = sorted( glob.glob1( image_dir, ext ) )
#    for i in range( 0, len( file_names ) ):
#        for j in range( i+1, len( file_names ) ):
#            if( file_names[i][0] == 'f' and file_names[j][0] == 'o'):
#                string = file_names[j] + ' ' + file_names[i] + ' ' + '0'
#                print( string, file = fp )
#            elif( file_names[i][0] == 'o' and file_names[j][0] == 'o'):
#                string = file_names[j] + ' ' + file_names[i] + ' ' + '1'
#                print( string, file = fp )
#                
#fp.close()

# For GPDS300, GPDS960

dataset = 'GPDS300'
ext = '*.bmp'

image_dir = '/home/' + usr + '/Workspace/Datasets/' + dataset + '/'
data_file = image_dir + dataset + '_pairs.txt'

subdir_names = sorted( os.listdir( image_dir ) )

fp = open( data_file, 'w' )

for sd in subdir_names:
    file_names = sorted( glob.glob1( image_dir + sd , ext ) )
    for i in range( 0, len( file_names ) ):
        for j in range( i+1, len( file_names ) ):
            if( file_names[i][0:2] == 'c-' and file_names[j][0:2] == 'c-'):
                string = sd + '/' + file_names[i] + ' ' + sd + '/' + \
                        file_names[j] + ' ' + '1' + '\n'
                fp.write(string)
#                 print( string, file = fp )      # for Python3.0+
            elif( file_names[i][0:2] == 'c-' and file_names[j][0:2] == 'cf'):
                string = sd + '/' + file_names[i] + ' ' + sd + '/' + \
                        file_names[j] + ' ' + '0' +'\n'
                fp.write(string)
                # print( string, file = fp )   # for Python3.0+
fp.close()
# For Bengali, Hindi
#
#dataset = 'Bengali'
#ext = '*.tif'
#
#image_dir = '/home/' + usr + '/Workspace/Datasets/' + dataset + '/'
#data_file = image_dir + dataset + '_pairs.txt'
#
#subdir_names = sorted( os.listdir( image_dir ) )
#
#fp = open( data_file, 'w' )
#
#for sd in subdir_names:
#    file_names = sorted( glob.glob1( image_dir + sd , ext ) )
#    for i in range( 0, len( file_names ) ):
#        for j in range( i+1, len( file_names ) ):
#            if( file_names[i].split('-')[3] == 'F' and file_names[j].split('-')[3] == 'G'):
#                string = sd + '/' + file_names[j] + ' ' + sd + '/' + \
#                        file_names[i] + ' ' + '0'
#                print( string, file = fp )
#            elif( file_names[i].split('-')[3] == 'G' and file_names[j].split('-')[3] == 'G'):
#                string = sd + '/' + file_names[j] + ' ' + sd + '/' + \
#                        file_names[i] + ' ' + '1'
#                print( string, file = fp )
#
#fp.close()