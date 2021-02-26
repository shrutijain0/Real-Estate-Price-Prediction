# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 12:41:44 2021

@author: milin
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

df=pd.read_csv('EPP cleaned.csv',index_col=False)

df1=df.drop(['RESALE','UNDER_CONSTRUCTION','RERA','Unnamed: 0','BHK_OR_RK'],axis=1)


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn import linear_model

X=df1.drop('TARGET(PRICE_IN_LACS)',axis=1)
y=df1['TARGET(PRICE_IN_LACS)']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

models = {"Linear Regression" : LinearRegression(),
          "Lasso Regression" : Lasso(),
          "Ridge Regression" : Ridge(),
          "ElasticNet Regression" : ElasticNet()}

def fit_and_score(models, X_train, X_test, y_train, y_test) : 

    np.random.seed(42)
    model_scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        model_scores[name] = model.score(X_test, y_test)
    return model_scores

model_scores = fit_and_score(models= models,
                             X_train = X_train,
                             X_test = X_test,
                             y_train = y_train,
                             y_test = y_test)

model_scores

model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
y_pred

modelscore=model.score(X_test,y_test)
modelscore
