# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

import pandas as pd
import os
#import matplotlib.pyplot as plt
#import matplotlib.image as img
import numpy as np
from PIL import Image
from sklearn.datasets import load_digits
import seaborn as sns; sns.set()
import cv2 as cv
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV


class MTeX(object):
    
    def __init__(self, user_input):
        self.user_input = user_input
    
    def ML_Call(self, data):
        data_dict = data.to_dict()
        Xtrain, Xtest, ytrain, ytest = train_test_split(data_dict['data'], data_dict['target'],
                                                random_state=0)
        #model = RandomForestClassifier(n_estimators=1000)
        #model.fit(Xtrain, ytrain)
        #ypred = model.predict(Xtest)
        #accuracy = metrics.classification_report(ypred, ytest))
        param_grid = {"max_depth": [10, 50, 100],
              "n_estimators": [16, 32, 64],
              "random_state": [1234]}

        grid = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=10)

        grid.fit(Xtrain, ytrain)

        #print("best mean cross-validation score: {:.3f}".format(grid.best_score_))
        #print("best parameters: {}".format(grid.best_params_))
        #print("test-set score (accuracy): {:.3f}".format(grid.score(Xtest, ytest)))

        modelGrid = RandomForestClassifier(**grid.best_params_).fit(Xtrain, ytrain)
        return(modelGrid)
    
    def get_img(self, in_path, out_path):
        in_path = os.path.abspath(in_path)
        the_dirs = os.listdir(in_path)
        out_files = os.listdir(out_path)
        for in_name, out_name in zip(the_dirs, out_files):
            the_filenames = os.listdir(in_path + "/" + in_name)
            for word in the_filenames[0:100]:
                img = Image.open(in_path + '/' + in_name + '/' + word)
                img = img.resize((32,32))
                img.save(out_path + "/" + out_name + "/" + word)
                
    def contour_resize(self, in_path, out_path):
        in_path = os.path.abspath(in_path)
        the_dirs = os.listdir(in_path)
        for word in the_dirs:
            img = Image.open(in_path + '/' + word)
            img = img.resize((32,32))
            img.save(out_path + "/" + word)
    
    def fetch_df(self, the_path_in):
        the_path_in = os.path.abspath(the_path_in)
        the_dirs = os.listdir(the_path_in)
        the_df_out = pd.DataFrame()
        for dir_name in the_dirs:
            the_filenames = os.listdir(the_path_in + "/" + dir_name)
            for word in the_filenames[0:100]:
                f = np.array(cv.imread(the_path_in + '/' + dir_name + '/' + word))
                f = cv.cvtColor(f, cv.COLOR_BGR2GRAY)
                datacol = pd.DataFrame([[self.chonkify(f)]], columns=['data'])
                datacol['target'] = dir_name
                the_df_out = the_df_out.append(datacol, ignore_index=True)
            
        digits = load_digits()
        empty_col = np.empty(len(digits.target), dtype=np.object)
        for i in range(len(digits.target)):
            empty_col[i] = digits.data[i]
        
        for x,d in enumerate(empty_col):
            empty_col[x] = d.tolist()
            for y in range(len(d)):
                d[y] = int(d[y])

        test = pd.DataFrame.from_dict({"target":digits.target, "data":empty_col})
        the_df_out = pd.concat([test, the_df_out], sort = True, ignore_index = True)
        return(the_df_out)
    
    def fetch_contour(self, the_path_in):
        the_path_in = os.path.abspath(the_path_in)
        the_dirs = os.listdir(the_path_in)
        #the_df_out = pd.DataFrame()
        the_df_out = []
        for word in the_dirs:
            f = np.array(cv.imread(the_path_in + '/' + word))
            f = cv.cvtColor(f, cv.COLOR_BGR2GRAY)
            #temp_col = pd.DataFrame([[self.chonkify(f)]], columns=['data'])
            #the_df_out = the_df_out.append(temp_col, ignore_index = True)
            the_df_out.append(self.chonkify(f))
            
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
    
    def prepro(self, folder):
        img = cv.imread(self.user_input) 
        threshold = 100

        src_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        src_gray = cv.blur(src_gray, (3,3))

        canny_output = cv.Canny(src_gray, threshold, threshold * 2)
        contours0, hierarchy = cv.findContours(canny_output, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        #contours0, hierarchy = cv.findContours(canny_output, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
        contours = [cv.approxPolyDP(cnt, 3, True) for cnt in contours0]

        for i in range(len(contours)):
            clone = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8)
            clone.fill(255)
            if cv.contourArea(contours[i]) > 200:
                cv.drawContours(clone, contours, i, (0,0,0), cv.FILLED)
                [x, y, w, h] = cv.boundingRect(contours[i])
                crop_img = clone[int(math.floor(y-h*0.5)):y+h*2, int(math.floor(x-w*0.5)):x+w*2]
                cv.imwrite(os.path.abspath("./Documents/GitHub/MTeX/{}/contour{}.jpg".format(folder, i)), crop_img)

        cv.waitKey(0)
        

