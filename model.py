#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 11:18:39 2019

@author: drivera
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import itertools
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

arroz = pd.read_csv("dataprep.csv")
arroz['Date'] = pd.to_datetime(arroz['Date'])
arroz = arroz.set_index('Date')

fig = plt.figure()
sns.set()

f, axs = plt.subplots(2,2,figsize=(15,7))

plt.subplot(121)
sns.lineplot(x=arroz.index, 
             y="Price", 
             data=arroz)
plt.title('Fedearroz Rice Price')
plt.ylabel('Price ($)')

plt.subplot(122)
sns.distplot(arroz.Price,
                    kde=False,
                    color="b")
plt.title('Fedearroz Rice Price Distribution')
plt.ylabel('Price Frequency')

plt.suptitle('Fedearroz Rice Price Analysis', fontsize=16)
plt.savefig('TSoriginal.png')

from pylab import rcParams
rcParams['figure.figsize'] = 18, 8

decomposition = sm.tsa.seasonal_decompose(arroz['Price'], model='additive')
fig = decomposition.plot()
plt.savefig('decomposition.png')

train, test = arroz[:len(arroz)-12], arroz[len(arroz)-12:]

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
            
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue
        
mod = sm.tsa.statespace.SARIMAX(train,
                                order=(0, 1, 1),
                                seasonal_order=(0, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()

results.plot_diagnostics(figsize=(16, 8))
plt.savefig('diagnosis.png')

#Data Fitted

pred = results.get_prediction(start=pd.to_datetime(train.index[0]), dynamic=False)
pred_ci = pred.conf_int()

ax = train.plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))

#ax.fill_between(pred_ci.index,
#                pred_ci.iloc[:, 0],
#                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')
ax.set_ylabel('Fedearroz Price')
plt.legend()

plt.savefig('datafitted.png')

#Data Forecasted

#Producing and visualizing forecasts

pred_uc = results.get_forecast(steps=12)
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

Arroz_Forecast = pd.DataFrame(pred_uc.predicted_mean)
Arroz_Forecast.columns = ['Price']
#print(Arroz_Forecast)
#print(arroz[-12:])

comparacion = Arroz_Forecast
#comparacion = arroz[-12:]
comparacion.columns = ["Forecast"]
comparacion['Real'] = arroz[-12:]
comparacion["Error"] = comparacion['Real'] - comparacion['Forecast']

comparacion.to_csv (r'comparacion.csv', index = None, header=True)

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

d = {'RMSE':[sqrt(mean_squared_error(comparacion['Real'],comparacion['Forecast']))],
     'MAPE':[mean_absolute_percentage_error(comparacion['Real'],comparacion['Forecast'])]}
Error = pd.DataFrame(data = d)

Error.to_csv (r'Error.csv', index = None, header=True)