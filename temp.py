# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

import pandas as pd
import os
#import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
from PIL import Image

class MTeX(object):
    
    def get_img(self, in_path, out_path):
        in_path = os.path.abspath(in_path)
        the_dirs = os.listdir(in_path)
        the_df_out = pd.DataFrame()
        out_files = os.listdir(out_path)
        for in_name, out_name in zip(the_dirs, out_files):
            the_filenames = os.listdir(in_path + "/" + in_name)
            for word in the_filenames[0:100]:
                img = Image.open(in_path + '/' + in_name + '/' + word)
                img = img.resize((32,32))
                img.save(out_path + "/" + out_name + "/" + word)
    
    def fetch_df(self, the_path_in):
        the_path_in = os.path.abspath(the_path_in)
        the_dirs = os.listdir(the_path_in)
        the_df_out = pd.DataFrame()
        for dir_name in the_dirs:
            the_filenames = os.listdir(the_path_in + "/" + dir_name)
            for word in the_filenames[0:100]:
                f = np.array(img.imread(the_path_in + '/' + dir_name + '/' + word))
                datacol = pd.DataFrame([[self.chonkify(f)]], columns=['data'])
                datacol['label'] = dir_name
                the_df_out = the_df_out.append(datacol, ignore_index=True)
            
        return(the_df_out)
    
    def chonkify(self, tmp):
        chonk = []

        for i in range(0,len(tmp),4):
            for j in range(0,len(tmp),4):
                chonk.append(tmp[i:i+4, j:j+4])

        temp = []
        for x,lst in enumerate(chonk):
            summer = []
            for y,line in enumerate(lst):
                for z,num in enumerate(line):
                    if num >= 170:
                        #chonk[x][y][z] = 0
                        summer.append(0)
                    else:
                        #chonk[x][y][z] = 1
                        summer.append(1)
                
            temp.append(sum(summer))

        return(temp)

#%%
MTeX = MTeX()

#%%

MTeX.get_img("C:/Users/swagj/Documents/GitHub/MTeX/ScienceDirect_Data/", "C:/Users/swagj/Documents/GitHub/MTeX/Resized Example/")

data = MTeX.fetch_df(the_path_in = r"C:\Users\swagj\Documents\GitHub\MTeX\Resized Example")

data.to_csv(r"C:\Users\swagj\Documents\GitHub\MTeX\SD_Data.csv")
