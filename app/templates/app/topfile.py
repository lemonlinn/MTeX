# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 08:56:06 2020

@author: swagj
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.metrics import confusion_matrix
from Documents.GitHub.MTeX.app.templates.app.classobj import MTeX

#%%

MTeX = MTeX(r"C:/Users/swagj/Documents/GitHub/MTeX/test/easy_test.jpg")

#%%

MTeX.get_img("C:/Users/swagj/Documents/GitHub/MTeX/ScienceDirect_Data/", "C:/Users/swagj/Documents/GitHub/MTeX/Resized Example/")

#%%

data = MTeX.fetch_df(the_path_in = r"C:\Users\swagj\Documents\GitHub\MTeX\Resized Example")

modelGrid, ytest, ypred = MTeX.ML_Call(data)

#%%

#data_dict = data.to_dict()

#data.to_csv(r"C:\Users\swagj\Documents\GitHub\MTeX\SD_Data.csv")



#param_grid = {"max_depth": [10, 50, 100],
              #"n_estimators": [16, 32, 64],
              #"random_state": [1234]}

#grid = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=10)

#grid.fit(Xtrain, ytrain)

#print("best mean cross-validation score: {:.3f}".format(grid.best_score_))
#print("best parameters: {}".format(grid.best_params_))
#print("test-set score (accuracy): {:.3f}".format(grid.score(Xtest, ytest)))

#modelGrid = RandomForestClassifier(**grid.best_params_).fit(Xtrain, ytrain)



#%%

MTeX.prepro(folder = "test_contour")

MTeX.contour_resize("C:/Users/swagj/Documents/GitHub/MTeX/test_contour", "C:/Users/swagj/Documents/GitHub/MTeX/resize_contour")

contour_df = MTeX.fetch_contour(r'C:\Users\swagj\Documents\GitHub\MTeX\resize_contour')

#%%

result = (modelGrid.predict(contour_df))

#my_json = json.dumps({"prediction":result.tolist()})

mat = confusion_matrix(ytest, ypred)
hm = sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label');

plt.show(block = False)
hm.get_figure().savefig(r"C:\Users\swagj\Documents\GitHub\MTeX\heatmap.png")
print(result)

#%%
files = os.listdir("C:/Users/swagj/Documents/GitHub/MTeX/resize_contour/")
for f in files:
    os.remove("C:/Users/swagj/Documents/GitHub/MTeX/resize_contour/" + f)
  
#%%
files = os.listdir("C:/Users/swagj/Documents/GitHub/MTeX/test_contour/")
for f in files:
    os.remove("C:/Users/swagj/Documents/GitHub/MTeX/test_contour/" + f)


