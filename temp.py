# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import os
#import matplotlib.pyplot as plt
import matplotlib.image as img

def fetch_df(the_path_in):
    the_path_in = os.path.abspath(the_path_in)
    the_dirs = os.listdir(the_path_in)
    the_df_out = pd.DataFrame()
    for dir_name in the_dirs:
        the_filenames = os.listdir(the_path_in + dir_name)
        for word in the_filenames[0:100]:
            f = img.imread(the_path_in + dir_name + '/' + word, "r", encoding='ISO-8859-1')
            tmp = pd.DataFrame(f.shape, columns=['data'])
            tmp['label'] = word
            the_df_out = the_df_out.append(tmp, ignore_index=True)
            f.close()
            
    return(the_df_out)

#%%
    
fetch_df("./Desktop/Resized Example/")
    
#the_dirs = os.listdir("./Users/xinyuzhang/Desktop/Resized Example/")