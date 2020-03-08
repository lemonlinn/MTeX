# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 08:54:10 2020

@author: swagj
"""

from flask import Flask, request, render_template
import MTeX
import json
import os


app = Flask(__name__)

MTeX = MTeX()

MTeX.get_img("./Documents/GitHub/MTeX/ScienceDirect_Data/", "./Documents/GitHub/MTeX/Resized Example/")
data = MTeX.fetch_df(the_path_in = r".\Documents\GitHub\MTeX\Resized Example")
modelGrid = MTeX.ML_Call(data)

@app.route('/')
def index():
    return render_template("index.html")
    
@app.route('/inputs', methods = ['POST'])
def inputs():
    if request.method == "POST":
        if request.files:
            image = request.files.get('image', '')
            MTeXpred = MTeX(image)
            MTeXpred.prepro(folder = "test_contour")
            MTeXpred.contour_resize("./Documents/GitHub/MTeX/test_contour", "C:/Users/swagj/Documents/GitHub/MTeX/resize_contour")
            contour_df = MTeXpred.fetch_contour(r'.\Documents\GitHub\MTeX\resize_contour')
            result = (modelGrid.predict(contour_df))
            my_json = json.dumps({"prediction":result.tolist()})
            files = os.listdir("./Documents/GitHub/MTeX/resize_contour/")
            for f in files:
                os.remove("./Documents/GitHub/MTeX/resize_contour/" + f)
            files = os.listdir("./Documents/GitHub/MTeX/test_contour/")
            for f in files:
                os.remove("./Documents/GitHub/MTeX/test_contour/" + f)
            
        return render_template("index.html", info = my_json)
    return render_template("index.html", info = my_json)

if __name__ == '__main__':
    app.run(port = 5000, debug = True)
        
        
        