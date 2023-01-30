#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

wdf = pd.read_csv("/home/pranjal/Documents/datasets/mtcars.csv")
#display(wdf)
wdf.head()



# In[28]:


# Data visualization: 
    # Check if Linear regression is applicable --> By plotting it

plt.scatter(wdf.wt, wdf.mpg, s = 15, c = "blue") # scatter Plot --> dots/points
plt.xlabel("wt", fontsize=15)
plt.ylabel("mpg", fontsize=15)
plt.xticks(fontsize=13, rotation=0)
plt.yticks(fontsize=13, rotation=0)


# In[29]:


from sklearn.linear_model import LinearRegression
features_df = wdf[['wt']]
target_val   = wdf.mpg

display(features_df.head())
display(target_val.head())


# In[31]:


features_df = wdf[['wt']]
target_val  = wdf.mpg

from sklearn.metrics import mean_absolute_error

reg_model = LinearRegression()


reg_model.fit(features_df, target_val)
print(f"\nIntercept: {reg_model.intercept_}, coeffecient: {reg_model.coef_}")


prediction = reg_model.predict(wdf[['wt']])


error = mean_absolute_error(target_val, prediction)
print(f"\nMean Absolute Error = {error}\n")


plt.scatter(wdf.wt, wdf.mpg, s = 25, c = "blue", alpha = 0.7, label = "Data")
plt.xlabel("Temperature", fontsize=15)
plt.ylabel("Humidity", fontsize=15)
plt.xticks(fontsize=13, rotation=0)
plt.yticks(fontsize=13, rotation=0)
plt.title(f"Model parameters:\nItrcpt = {reg_model.intercept_}\nCoeff = {reg_model.coef_}",\
          loc='left', fontsize=15)


plt.plot(wdf.wt, prediction, c = "red", lw = 1, alpha = 0.5, label = "Predictions")

plt.legend(fontsize=12, loc = 0) # loc -> location [0, 10]


# In[35]:


features_df = wdf[['wt']]
target_val  = wdf.mpg

from sklearn.metrics import mean_absolute_error

reg_model = LinearRegression()


reg_model.fit(features_df, target_val)
print(f"\nIntercept: {reg_model.intercept_}, coeffecient: {reg_model.coef_}")


prediction = reg_model.predict(wdf[['wt']])


error = mean_absolute_error(target_val, prediction)
print(f"\nMean Absolute Error = {error}\n")


plt.scatter(wdf.wt, wdf.mpg, s = 25, c = "blue", alpha = 0.7, label = "Data")
plt.xlabel("Temperature", fontsize=15)
plt.ylabel("Humidity", fontsize=15)
plt.xticks(fontsize=13, rotation=0)
plt.yticks(fontsize=13, rotation=0)
plt.title(f"Model parameters:\nItrcpt = {reg_model.intercept_}\nCoeff = {reg_model.coef_}",\
          loc='left', fontsize=15)


plt.plot(wdf.wt, prediction, c = "red", lw = 1, alpha = 0.5, label = "Predictions")



# ====================================================================================
# Prediction at a given value
wt1       = 3.20
pred1       = reg_model.predict([[wt1]])

print(f"\nPredicted value of mpg at wt {wt1} [K] is: {pred1}")
print(f"\nPredicted value of mpg at wt {wt1} [K] is: {pred1[0]}")

plt.scatter(wt1, pred1, color ='r', label = "predicted value")
plt.vlines(x = wt1, ymin = 0.0, ymax = pred1, color = 'y')
plt.hlines(y = pred1, xmin = 0.0, xmax = wt1, color = 'y')

# ====================================================================================

plt.legend(fontsize=12, loc = 0) # loc -> location [0, 10]
#plt.xlim(-7,22)

