# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 13:52:12 2021

@author: milin
"""

import streamlit as st
import pandas as pd
from sklearn.linear_model import  Lasso


st.write("""
# House Price Prediction
This app predicts the **India's House Price**!
""")
st.write('---')

df=pd.read_csv('EPP cleaned1.csv',index_col=False)
df1=df.drop(['RESALE','RERA','Unnamed: 0','BHK_OR_RK'],axis=1)

X=df1.drop('TARGET(PRICE_IN_LACS)',axis=1)
y=df1['TARGET(PRICE_IN_LACS)']


BHK_NO=st.number_input('Enter a BHK NO')
SQUARE_FT=st.number_input('Enter a Square FT')
UNDER_CONSTRUCTION=st.number_input('Enter 0 if underconstruction or 1 if not')


data={'BHK_NO':BHK_NO,'SQUARE_FT':SQUARE_FT,'UNDER_CONSTRUCTION':UNDER_CONSTRUCTION}
features = pd.DataFrame(data, index=[0])


st.header('Your Specified Inputs')
st.write(features)
st.write('---')

model = Lasso()
model.fit(X, y)
# Apply Model to Make Prediction
prediction = model.predict(features)



st.header('Prediction of House Price')
st.write(prediction)
st.write('---')
