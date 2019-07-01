#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import itertools
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings
import requests
from bs4 import BeautifulSoup
warnings.filterwarnings('ignore')
from PIL import Image


# In[19]:


mod = sm.load('arroz.pickle')


# In[20]:


def alarmas(comparacion):
    comparacion['Alarma Signo'] = ""
    comparacion['Alarma Error'] = ""
    
    for x in range(4,len(comparacion)+1):
        if comparacion['Error'].iloc[x-4] > 0 and comparacion['Error'].iloc[x-3] > 0 and comparacion['Error'].iloc[x-2] > 0 and comparacion['Error'].iloc[x-1] > 0 or comparacion['Error'].iloc[x-4] < 0 and comparacion['Error'].iloc[x-3] < 0 and comparacion['Error'].iloc[x-2] < 0 and comparacion['Error'].iloc[x-1] < 0:
            comparacion['Alarma Signo'].iloc[x-1] = "Alarma"
        
    comparacion['Alarma Error'] = np.where(comparacion['Error'] / (3.5*np.std(mod.forecasts_error))>1,'Alarma','OK')
    
    return comparacion


# In[21]:


compare_real = pd.read_csv("compare_real.csv")
compare_real['Date'] = pd.to_datetime(compare_real['Date'])
compare_real = compare_real.set_index('Date')


# In[22]:


alarmas = alarmas(compare_real)


# In[23]:


alarmas['Date'] =  alarmas.index
alarmas = alarmas[['Date','Forecast','Real','Error','Alarma Signo','Alarma Error']]
alarmas.to_csv (r'alarmas.csv', index = None, header=True)


# In[ ]:


if 'Alarma' in compare_real['Alarma Signo'].tolist():
    sys.exit(1)
else:
    sys.exit(0)

