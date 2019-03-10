# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 22:46:49 2018

@author: sbodur
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("random-forest-regression-dataset.csv",sep=";",header=None)

x=df.iloc[:,0].values.reshape(-1,1)
y=df.iloc[:,1].values.reshape(-1,1)

#%%
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=100,random_state=42)

rf.fit(x,y)
print("7.5 level of price = ",rf.predict(7.5))

x_=np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head=rf.predict(x_)

#visualize
plt.scatter(x,y,color="blue")
plt.plot(x_,y_head,color="green")
plt.xlabel("tribun level")
plt.ylabel("price")
plt.show()