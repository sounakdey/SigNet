# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 09:53:45 2017

@author: sounak
"""

import cv2
import numpy as np
import os




# Main Function    
if __name__ == "__main__":
    
    # Nominal Values : Colors, Fonts, Background   
    
    height = 100
    width = 500
    bg_dir = ''
    words_file = '/home/sounak/Documents/Datasets/word_lists/bengaliSuman.txt'
    
    # Setting the random seed
    seed = np.array([[0,0,0,0,1,3,3,7]], dtype='uint8') # Random seed '1337'
    
    
    # For the script
    script = cv2.text.CV_TEXT_SYNTHESIZER_SCRIPT_BENGALI
    

    
    # Colors
    colorImg = cv2.imread('1000_color_clusters.png',cv2.IMREAD_COLOR)
    
    # Creating the Text synthesiser instant
    text_synth = cv2.text.TextSynthesizer_create(100, 500, script)
    text_synth.setRandomSeed(seed)
    text_synth.setColorClusters(colorImg)
    
    # Background Images (Adding)
    onlyfiles = [f for f in os.listdir(bg_dir) if os.path.isfile(os.paht.join(bg_dir, f))]
    for fp in onlyfiles:
        bg_img = cv2.imread(fp)
        text_synth.addBgSampleImage(bg_img)
        
    # Generate the images
    data = [line for line in open(words_file).read().split('\n') if len(line)>0]
    for text in data:
        synth_img = text_synth.generateSample(text)
        filename = 'hehe.jpg'
        cv2.imwrite(filename,synth_img)
        