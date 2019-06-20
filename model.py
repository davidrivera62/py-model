#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


arroz = pd.read_csv("dataprep.csv")
arroz['Date'] = pd.to_datetime(arroz['Date'])
arroz = arroz.set_index('Date')


# In[4]:


fig = plt.figure()
sns.set()

f, axs = plt.subplots(2, 2, figsize=(15, 7))

plt.subplot(121)
sns.lineplot(x=arroz.index,
             y="Price",
             data=arroz)
plt.title('Fedearroz Rice Price')
plt.ylabel('Price ($)')

plt.subplot(122)
sns.distplot(arroz.Price, kde=False, color="b")
plt.title('Fedearroz Rice Price Distribution')
plt.ylabel('Price Frequency')

plt.suptitle('Fedearroz Rice Price Analysis', fontsize=16)
plt.savefig('TSoriginal.png')

from pylab import rcParams
rcParams['figure.figsize'] = 18, 8

decomposition = sm.tsa.seasonal_decompose(arroz['Price'], model='additive')
fig = decomposition.plot()
plt.savefig('decomposition.png')

# In[5]:


# n peridos a pronosticar
n = 6

train, test = arroz[:len(arroz)-n], arroz[len(arroz)-n:]


# In[6]:


list_aic = []
list_param = []
list_param_seasonal = []

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(train,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            list_aic.append(results.aic)
            list_param.append(param)
            list_param_seasonal.append(param_seasonal)

        except:
            continue


# In[7]:


print('ARIMA{}x{}12 - AIC:{}'.format(list_param[list_aic.index(min(list_aic))],
                                     list_param_seasonal[list_aic.index(min(list_aic))],
                                     min(list_aic)))


# In[8]:


mod = sm.tsa.statespace.SARIMAX(train,order=(list_param[list_aic.index(min(list_aic))][0],
                                             list_param[list_aic.index(min(list_aic))][1],
                                             list_param[list_aic.index(min(list_aic))][2]),
                                seasonal_order=(list_param_seasonal[list_aic.index(min(list_aic))][0],
                                                list_param_seasonal[list_aic.index(min(list_aic))][1],
                                                list_param_seasonal[list_aic.index(min(list_aic))][2],
                                                12),
                                 enforce_stationarity=False, enforce_invertibility=False,).fit()
mod.summary()


# In[9]:


results.plot_diagnostics(figsize=(16, 8))
plt.savefig('diagnosis.png')


# In[10]:


d = {'AIC':[results.aic],
     'BIC':[results.bic]}
aicbic = pd.DataFrame(data=d)
aicbic.to_csv (r'aicbic.csv', index=None, header=True)


# In[11]:


# Data Fitted

pred = results.get_prediction(start=pd.to_datetime(train.index[0]), dynamic=False)
pred_ci = pred.conf_int()


# In[12]:


ax = train.plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))

ax.set_xlabel('Date')
ax.set_ylabel('Fedearroz Price')
plt.legend()

plt.savefig('datafitted.png')
plt.show()


# In[13]:


# Producing and visualizing forecasts

pred_uc = results.get_forecast(steps=n)
pred_ci = pred_uc.conf_int()

ax = arroz.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Fedearroz Price')

plt.legend()

plt.savefig('forecasting.png')


# In[14]:


Arroz_Forecast = pd.DataFrame(pred_uc.predicted_mean)
Arroz_Forecast.columns = ['Price']


# In[15]:


comparacion = pd.DataFrame(index=Arroz_Forecast.index, columns=['Date'])
comparacion['Date'] = Arroz_Forecast.index
comparacion['Forecast'] = Arroz_Forecast['Price']
comparacion['Real'] = arroz[-n:]
comparacion["Error"] = comparacion['Real'] - comparacion['Forecast']
comparacion.to_csv(r'comparacion.csv', index = None, header=True)


# In[16]:


comparacion


# In[17]:


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[18]:


d = {'RMSE': [sqrt(mean_squared_error(comparacion['Real'],
                                     comparacion['Forecast']))],
     'MAPE':[mean_absolute_percentage_error(comparacion['Real'],
                                            comparacion['Forecast'])]}


# In[19]:


Error = pd.DataFrame(data=d)
Error.to_csv(r'Error.csv', index=None, header=True)


# In[20]:


np.std(results.forecasts_error)


# In[21]:


Big_Error = pd.DataFrame(index=Arroz_Forecast.index, columns=['ratio'])


# In[22]:


Big_Error['ratio'] = comparacion['Error'] / (3.5*np.std(results.forecasts_error))
