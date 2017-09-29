# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 13:07:29 2017

@author: sounak
"""

import pandas as pd
import numpy as np
import scipy.spatial as ssp
from collections import Counter

def semantics2vector():
    word_topics = open('/home/sounak/Documents/python_learning/lexicon_oxford_word_semantics.txt','r')

    lines = word_topics.readlines()
    temp_list = []
    # Creating the list from the text file
    for i in lines:
        gg = i.split('   ')
        temp_list.append(gg)
        
    # Flattening the list
    flat_temp_list = [item for sublist in temp_list for item in sublist]
    # List if semantics with now path weights
    flat_list_no_path_num = [item.split('_')[0] for item in flat_temp_list]
    # Creating the histogram of the semantics
    word_counts = Counter(flat_list_no_path_num)
    df = pd.DataFrame.from_dict(word_counts, orient='index')
    # Sorting the histogram 
    sorted_df = df.sort_values(by=[0], ascending=[False])
    # First few semantics
    first_128 = sorted_df[0:127]
    # Get the wordlist # This is not used in anything :)
    words = [sublist[0].split('_')[0] for sublist in temp_list]
    # Create the vector of the length of the number of semantics predetermined
    fin_lista_semantics = list(first_128.index)
    
    # The initialisation
    vector_semantics = np.zeros((len(lines),len(fin_lista_semantics)))
    
    # Making the vectors
    for num, sublist in enumerate(temp_list):
        for item in sublist:
            if item.split('_')[1] != '1' and (item.split('_')[0] in fin_lista_semantics):
                vector_semantics[num][fin_lista_semantics.index(item.split('_')[0])] = 1
    
    # For using the words with only with the semantics # Trying to trim the previous vector
    good_rows = np.asarray(np.nonzero(vector_semantics.sum(axis=1) > 0)).ravel()
    semantic2vec = vector_semantics[good_rows,:]
    
    return semantic2vec
    
if __name__ == '__main__':
    semantic2vec = semantics2vector()
    # Convert to binary
    semantic2vec = np.asarray(semantic2vec, dtype=np.int8)
    # Hamming distance calculation
    length = len(semantic2vec)             # Initialization
    ham_dist = np.zeros((length,length))
#    ham_dist = np.count_nonzero(semantic2vec!=semantic2vec)
    for i in range(length):
        for j in range(length):
            print ssp.distance.hamming(semantic2vec[i,:], semantic2vec[j,:])
            ham_dist[i,j] = ssp.distance.hamming(semantic2vec[i,:], semantic2vec[j,:])
            
    print max(ham_dist)   
    
#    hamming_distance = ssp.distance.hamming(semantic2vec, semantic2vec)