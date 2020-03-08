# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 08:54:10 2020

@author: swagj
"""

from flask import Flask, request, render_template
import MTeX

app = Flask(__name__)

MTeX = MTeX()

MTeX.get_img("C:/Users/swagj/Documents/GitHub/MTeX/ScienceDirect_Data/", "C:/Users/swagj/Documents/GitHub/MTeX/Resized Example/")
data = MTeX.fetch_df(the_path_in = r"C:\Users\swagj\Documents\GitHub\MTeX\Resized Example")
modelGrid = MTeX.ML_Call(data)

if request.method == "POST":
    if request.files:
        image = request.files.get('image', '')
        MTeXpred = MTeX(image)
        MTeXpred.prepro(folder = "test_contour")
        MTeXpred.contour_resize("C:/Users/swagj/Documents/GitHub/MTeX/test", "C:/Users/swagj/Documents/GitHub/MTeX/resize_contour")
        contour_df = MTeXpred.fetch_contour(r'C:\Users\swagj\Documents\GitHub\MTeX\resize_contour')
        result = (modelGrid.predict(contour_df))
        
        
        