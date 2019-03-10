# -*- coding: utf-8 -*-

"""
Created on Wed Sep  5 18:40:46 2018

@author: sbodur
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df1=pd.read_csv("lineer-regression_dataset.csv",sep= ";")

plt.scatter(df1.deneyim,df1.maas)
plt.xlabel("deneyim")
plt.ylabel("maas")
plt.show()

#%% sklearn

from sklearn.linear_model import LinearRegression

#linear regression model
linear_reg=LinearRegression()

x=df1.deneyim.values.reshape(-1,1)   # When this part execute ,show the like (14,)
y=df1.maas.values.reshape(-1,1)      # We cant see y value so use the reshape method

linear_reg.fit(x,y)

#%% Prediction
b0=linear_reg.predict(0)

print("b0 : ", b0)

b0_=linear_reg.intercept_
print("b0_ : ", b0_)       #intercept

b1=linear_reg.coef_
print("b1 : ",b1)           #slope

print(linear_reg.predict(11))

#visualize
array=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1)
plt.scatter(x,y)
plt.show()

y_head=linear_reg.predict(array)
plt.plot(array,y_head,color="red")
