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
from sklearn.datasets import load_digits
#import scipy.io as sio
import cv2 as cv
import math


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
    
    def prepro(self, img_file):
        img = cv.imread(img_file) 
        threshold = 100

        src_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        src_gray = cv.blur(src_gray, (3,3))

        canny_output = cv.Canny(src_gray, threshold, threshold * 2)
        contours0, hierarchy = cv.findContours(canny_output, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = [cv.approxPolyDP(cnt, 3, True) for cnt in contours0]

        for i in range(len(contours)):
            clone = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8)
            clone.fill(255)
            if cv.contourArea(contours[i]) > 200:
                cv.drawContours(clone, contours, i, (0,0,0), cv.FILLED)
                [x, y, w, h] = cv.boundingRect(contours[i])
                crop_img = clone[int(math.floor(y-h*0.5)):y+h*2, int(math.floor(x-w*0.5)):x+w*2]
                cv.imwrite("./contours/contour{}.jpg".format(i), crop_img)

        cv.waitKey(0)
        
#%%
MTeX = MTeX()

#%%

MTeX.get_img("C:/Users/swagj/Documents/GitHub/MTeX/ScienceDirect_Data/", "C:/Users/swagj/Documents/GitHub/MTeX/Resized Example/")

#%%

data = MTeX.fetch_df(the_path_in = r"C:\Users\swagj\Documents\GitHub\MTeX\Resized Example")

data_dict = data.to_dict()

data.to_csv(r"C:\Users\swagj\Documents\GitHub\MTeX\SD_Data.csv")

print(type(data.data[0]), type(data.data[2000]), type(data.data[0][0]), type(data.data[2000][0]))

#%%

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.metrics import confusion_matrix

#%%

Xtrain, Xtest, ytrain, ytest = train_test_split(data_dict['data'], data_dict['target'],
                                                random_state=0)
model = RandomForestClassifier(n_estimators=1000)
model.fit(Xtrain, ytrain)
ypred = model.predict(Xtest)
print(metrics.classification_report(ypred, ytest))

#%%
import matplotlib.image as img
from scipy.ndimage import sobel
import numpy as np

user_input = img.imread(r'C:/Users/swagj/Documents/GitHub/MTeX/IMG_0151.JPG')

sx = sobel(user_input, axis=0, mode='constant')
sy = sobel(user_input, axis=1, mode='constant')
sob = np.hypot(sx, sy)

#%%

import cv2
import numpy as np

# load image
img = cv2.imread(r'C:/Users/swagj/Documents/GitHub/MTeX/IMG_0151.JPG') 
rsz_img = cv2.resize(img, None, fx=0.25, fy=0.25) # resize since image is huge
gray = cv2.cvtColor(rsz_img, cv2.COLOR_BGR2GRAY) # convert to grayscale

# threshold to get just the signature
retval, thresh_gray = cv2.threshold(gray, thresh=100, maxval=255, type=cv2.THRESH_BINARY)

# find where the signature is and make a cropped region
points = np.argwhere(thresh_gray==0) # find where the black pixels are
points = np.fliplr(points) # store them in x,y coordinates instead of row,col indices
x, y, w, h = cv2.boundingRect(points) # create a rectangle around those points
x, y, w, h = x-10, y-10, w+20, h+20 # make the box a little bigger
crop = gray[y:y+h, x:x+w] # create a cropped region of the gray image

# get the thresholded crop
retval, thresh_crop = cv2.threshold(crop, thresh=200, maxval=255, type=cv2.THRESH_BINARY)

# display
cv2.imshow("Cropped and thresholded image", thresh_crop) 
cv2.waitKey(0)

#%%
import cv2 as cv
import numpy as np
import matplotlib.image as img
import math

img = cv.imread(r'C:/Users/swagj/Documents/GitHub/MTeX/IMG_0151.JPG') 
threshold = 100

src_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
src_gray = cv.blur(src_gray, (3,3))

canny_output = cv.Canny(src_gray, threshold, threshold * 2)
contours0, hierarchy = cv.findContours(canny_output, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contours = [cv.approxPolyDP(cnt, 3, True) for cnt in contours0]

drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

for i in range(len(contours)):
    clone = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8)
    clone.fill(255)
    if cv.contourArea(contours[i]) > 200:
        cv.drawContours(clone, contours, i, (0,0,0), cv.FILLED)
        [x, y, w, h] = cv.boundingRect(contours[i])
        #cv.rectangle(clone,(int(math.floor(x-w*0.5)),int(math.floor(y-h*0.5))),(x+w*2,y+h*2), (255,0,0), 2)
        crop_img = clone[int(math.floor(y-h*0.5)):y+h*2, int(math.floor(x-w*0.5)):x+w*2]
        cv.imwrite("./contours/contour{}.jpg".format(i), crop_img)

#cv.imshow('Contours', drawing)

cv.waitKey()