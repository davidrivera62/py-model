#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import itertools
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings
warnings.filterwarnings('ignore')


# In[35]:


arroz = pd.read_csv("data_real.csv")
arroz['Date'] = pd.to_datetime(arroz['Date'])
arroz = arroz.set_index('Date')


# In[36]:


mod = sm.load('arroz.pickle')


# In[39]:


pred= mod.get_forecast(steps = 4+6)
pred_ci = pred.conf_int()


# In[40]:


#Producing and visualizing forecasts

pred_uc = mod.get_forecast(steps=4+6)
pred_ci = pred_uc.conf_int()

ax = arroz.plot(marker='o',label='observed',figsize=(14, 7))
pred_uc.predicted_mean.tail(4).plot(marker='*',ax=ax, label='Forecast',)
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Fecha')
ax.set_ylabel('Fedearroz Price')

plt.legend()
plt.savefig('forecasting_real.png')


# In[41]:


Arroz_Forecast = pd.DataFrame(pred_uc.predicted_mean.tail(4))
Arroz_Forecast.columns = ['Price']


# In[42]:


compare_real = pd.DataFrame(index=Arroz_Forecast.index, columns=['Date'])
compare_real['Date'] = Arroz_Forecast.index
compare_real['Forecast'] = Arroz_Forecast['Price']
compare_real['Real'] = arroz
compare_real["Error"] = compare_real['Real'] - compare_real['Forecast']
compare_real.to_csv (r'compare_real.csv', index = None, header=True)


# In[28]:


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[29]:


d = {'RMSE':[sqrt(mean_squared_error(compare_real['Real'],compare_real['Forecast']))],
     'MAPE':[mean_absolute_percentage_error(compare_real['Real'],compare_real['Forecast'])]}


# In[30]:


Error_real = pd.DataFrame(data = d)
Error_real.to_csv (r'Error.csv', index = None, header=True)


# In[ ]:




