# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 14:46:30 2020

@author: Rahul3.Tiwari
"""

import pandas as pd
#mport numpy as np
import pickle

dta_st = pd.read_csv('C:\\Users\\rahul3.tiwari\\Desktop\\salaries.csv')
X = dta_st.drop(['Salary'],axis=1)
y = dta_st['Salary']



from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X,y)

pickle.dump(reg, open('model-salary.pkl','wb'))

model = pickle.load(open('model-salary.pkl','rb'))
print(model.predict([[3.5]]))


