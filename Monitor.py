#!/usr/bin/env python
# coding: utf-8

# In[10]:


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


# In[17]:


mod = sm.load('arroz.pickle')


# In[18]:


compare_real = pd.read_csv("compare_real.csv")
compare_real['Date'] = pd.to_datetime(compare_real['Date'])
compare_real = compare_real.set_index('Date')
compare_real['Alarma Signo'] = ""
compare_real['Alarma Error'] = ""


# In[19]:


for x in range(4,len(compare_real)+1):
    if compare_real['Error'].iloc[x-4] > 0 and compare_real['Error'].iloc[x-3] > 0 and compare_real['Error'].iloc[x-2] > 0 and compare_real['Error'].iloc[x-1] > 0 or compare_real['Error'].iloc[x-4] < 0 and compare_real['Error'].iloc[x-3] < 0 and compare_real['Error'].iloc[x-2] < 0 and compare_real['Error'].iloc[x-1] < 0:
        compare_real['Alarma Signo'].iloc[x-1] = "Alarma"   
    


# In[20]:


compare_real['Alarma Error'] = np.where(compare_real['Error'] / (3.5*np.std(mod.forecasts_error))>1,'Alarma','OK')


# In[23]:


compare_real['Date'] =  compare_real.index
compare_real = compare_real[['Date','Forecast','Real','Error','Alarma Signo','Alarma Error']]
compare_real.to_csv (r'alarmas.csv', index = None, header=True)


# In[ ]:




