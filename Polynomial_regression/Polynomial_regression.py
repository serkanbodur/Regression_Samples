# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
@author sbodur
"""

import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("Polynomial_regression.csv",sep=";")

y=df.araba_max_hiz.values.reshape(-1,1)
x=df.araba_fiyat.values.reshape(-1,1)

plt.scatter(x,y)
plt.ylabel("araba_max_hiz")
plt.xlabel("araba_fiyat")
plt.show()

#linear regression
#%%
from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(x,y)
#%%

y_head=lr.predict(x)

plt.plot(x,y_head,color="blue",label="Linear")
plt.show()

print("10000k lik araba",lr.predict(10000))

# %%
#polynomial regression = y = b0+b1*x+b2*x^2+b3*x^3

from sklearn.preprocessing import PolynomialFeatures

polynomial_regression=PolynomialFeatures(degree=2)
x_polynomial=polynomial_regression.fit_transform(x)  #x i x kareye cevir

#if you increase degree your result come to true value
# %% Fit
linear_regression2=LinearRegression()
linear_regression2.fit(x_polynomial,y)

# %%
y_head2=linear_regression2.predict(x_polynomial)
plt.plot(x,y_head2,color="black",label="Polynomial")
plt.legend()
plt.show()


