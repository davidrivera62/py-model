#!/usr/bin/env python
# coding: utf-8

# In[4]:


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


# In[5]:


mod = sm.load('arroz.pickle')


# In[6]:


def alarmas(comparacion):
    comparacion['Alarma Signo'] = ""
    comparacion['Alarma Error'] = ""

    for x in range(4,len(comparacion)+1):
        if comparacion['Error'].iloc[x-4] > 0 and comparacion['Error'].iloc[x-3] > 0 and comparacion['Error'].iloc[x-2] > 0 and comparacion['Error'].iloc[x-1] > 0 or comparacion['Error'].iloc[x-4] < 0 and comparacion['Error'].iloc[x-3] < 0 and comparacion['Error'].iloc[x-2] < 0 and comparacion['Error'].iloc[x-1] < 0:
            comparacion['Alarma Signo'].iloc[x-1] = "Alarma"

    comparacion['Alarma Error'] = np.where(comparacion['Error'] / (3.5*np.std(mod.forecasts_error))>1,'Alarma','OK')

    return comparacion


# In[7]:


def Dataprep(n):
    # Web Scraping - https://pythonprogramminglanguage.com/web-scraping-with-pandas-and-beautifulsoup/
    res = requests.get("http://www.fedearroz.com.co/new/precios.php")
    soup = BeautifulSoup(res.content,'lxml')
    table = soup.find_all('table')[0]
    df = pd.read_html(str(table))

    #Data Wrangling
    arroz=df[0]

    arroz['Mes'] = ['1','2','3','4','5','6','7','8','9','10','11','12']
    arroz = pd.melt(arroz, id_vars=['Mes'],var_name='Year',value_name='Price')
    arroz = arroz.rename(columns={'Mes': 'Month'})
    arroz['Year'] = arroz['Year'].astype(int)
    arroz['Month'] = arroz['Month'].astype(int)
    arroz = arroz.dropna()

    arroz['Date']=pd.to_datetime((arroz.Year*10000+arroz.Month*100+1).apply(str),format='%Y%m%d')

    arroz=arroz[['Date','Price']]
    arroz['Price']=arroz['Price']/1000
    arroz = arroz.set_index('Date')
    arroz.drop(arroz.tail(1).index,inplace=True)

    #(Base: diciembre 2014=100)
    IPP = pd.read_csv('IPP.csv',sep=';',decimal=',')
    IPP['Date'] = pd.to_datetime(IPP['Año(aaaa)-Mes(mm)'])
    IPP = IPP.set_index('Date').dropna()

    arroz['Price'] = arroz['Price']*IPP['Factor']
    #arroz['Date'] = arroz.index
    #arroz = arroz[['Date','Price']]

    train, test = arroz[:len(arroz)-n], arroz[len(arroz)-n:]

    return train, test


# In[8]:


def modelrice(train):
    list_aic= []
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

                #print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))

                list_aic.append(results.aic)
                list_param.append(param)
                list_param_seasonal.append(param_seasonal)

            except:
                continue

    mod = sm.tsa.statespace.SARIMAX(train,order=(list_param[list_aic.index(min(list_aic))][0],
                                             list_param[list_aic.index(min(list_aic))][1],
                                             list_param[list_aic.index(min(list_aic))][2]),
                                seasonal_order=(list_param_seasonal[list_aic.index(min(list_aic))][0],
                                                list_param_seasonal[list_aic.index(min(list_aic))][1],
                                                list_param_seasonal[list_aic.index(min(list_aic))][2],
                                                12),
                                 enforce_stationarity=False, enforce_invertibility=False,).fit()

    return mod


# In[9]:


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[10]:


n = 6
train, test = Dataprep(n)
mod = modelrice(train)

plt.close('all')
plt.rc('figure', figsize=(12, 7))
plt.text(0.01, 0.05, str(mod.summary()), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
plt.axis('off')
plt.tight_layout()

### salida
plt.savefig('Model_Results_new.png')

#Producing and visualizing forecasts

pred_uc = mod.get_forecast(steps=n)
pred_ci = pred_uc.conf_int()

ax = test.plot(marker='o',label='observed',figsize=(14, 7))
pred_uc.predicted_mean.plot(marker='*',ax=ax, label='Forecast',)
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Fecha')
ax.set_ylabel('Fedearroz Price')

plt.legend()

### salida
plt.savefig('forecasting_real_new.png')

Arroz_Forecast = pd.DataFrame(pred_uc.predicted_mean)
Arroz_Forecast.columns = ['Price']

comparacion = pd.DataFrame(index=Arroz_Forecast.index, columns=['Date'])
comparacion['Date'] = Arroz_Forecast.index
comparacion['Forecast'] = Arroz_Forecast['Price']
comparacion['Real'] = test
comparacion["Error"] = comparacion['Real'] - comparacion['Forecast']

d = {'RMSE':[sqrt(mean_squared_error(comparacion['Real'],comparacion['Forecast']))],
     'MAPE':[mean_absolute_percentage_error(comparacion['Real'],comparacion['Forecast'])]}

Error = pd.DataFrame(data = d)
### salida
Error.to_csv (r'Error_real_new.csv', index = None, header=True)

comparacion['Alarma Signo'] = ""
comparacion['Alarma Error'] = ""

for x in range(4,len(comparacion)+1):
    if comparacion['Error'].iloc[x-4] > 0 and comparacion['Error'].iloc[x-3] > 0 and comparacion['Error'].iloc[x-2] > 0 and comparacion['Error'].iloc[x-1] > 0 or comparacion['Error'].iloc[x-4] < 0 and comparacion['Error'].iloc[x-3] < 0 and comparacion['Error'].iloc[x-2] < 0 and comparacion['Error'].iloc[x-1] < 0:
        comparacion['Alarma Signo'].iloc[x-1] = "Alarma"

### salida
comparacion['Alarma Error'] = np.where(comparacion['Error'] / (3.5*np.std(mod.forecasts_error))>1,'Alarma','OK')

comparacion.to_csv (r'comparacion_real_new.csv', index = None, header=True)


# In[ ]:
