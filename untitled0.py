# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 15:15:10 2020

@author: swagj
"""


from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import random as rng

rng.seed(12345)

def thresh_callback(val):
    threshold = val

    # Detect edges using Canny
    canny_output = cv.Canny(src_gray, threshold, threshold * 2)

    # Find contours
    contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Draw contours
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv.drawContours(drawing, contours, i, color, 2, cv.LINE_8, hierarchy, 0)

    # Show in a window
    cv.imshow('Contours', drawing)

# Load source image
parser = argparse.ArgumentParser(description='Code for Finding contours in your image tutorial.')
parser.add_argument('--input', help='Path to input image.', default=r'C:/Users/swagj/Documents/GitHub/MTeX/IMG_0151.JPG')
args = parser.parse_args()

src = cv.imread(cv.samples.findFile(args.input))
if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)

# Convert image to gray and blur it
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
src_gray = cv.blur(src_gray, (3,3))

# Create Window
source_window = 'Source'
cv.namedWindow(source_window)
cv.imshow(source_window, src)
max_thresh = 255
thresh = 100 # initial threshold
cv.createTrackbar('Canny Thresh:', source_window, thresh, max_thresh, thresh_callback)
thresh_callback(thresh)

cv.waitKey()

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