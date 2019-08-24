#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import requests
from bs4 import BeautifulSoup


# In[13]:


# Web Scraping - https://pythonprogramminglanguage.com/web-scraping-with-pandas-and-beautifulsoup/
res = requests.get("http://www.fedearroz.com.co/new/precios.php")
soup = BeautifulSoup(res.content,'lxml')
table = soup.find_all('table')[0]
df = pd.read_html(str(table))


# In[14]:


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
arroz.drop(arroz.tail(5).index,inplace=True)


# In[15]:


arroz.to_csv (r'dataprep.csv', index = None, header=True)


# In[16]:


oni = pd.read_csv('ONI.csv',sep=';')
oni['Date'] = pd.to_datetime(oni['Date'])
oni['Date'].dt.strftime('%Y-%m-%d')
oni = oni.set_index('Date').dropna()
oni.drop(oni.tail(3).index,inplace=True) # drop last n rows


# In[17]:


precipitaciones = pd.read_csv('Precipitaciones.csv',sep=';',decimal=',')
precipitaciones['Date'] = pd.to_datetime(precipitaciones['Date'])
precipitaciones['Date'].dt.strftime('%Y-%m-%d')
precipitaciones = precipitaciones.set_index('Date').dropna()


# In[18]:


trm = pd.read_csv('TRM.csv')
trm['Date'] = pd.to_datetime(trm['Date'])
trm['Date'].dt.strftime('%Y-%m-%d')
trm = trm.set_index('Date').dropna()
trm.drop(trm.tail(4).index,inplace=True) # drop last n row


# In[19]:


#(Base: diciembre 2014=100)
IPP = pd.read_csv('IPP.csv',sep=';',decimal=',')
IPP['Date'] = pd.to_datetime(IPP['Año(aaaa)-Mes(mm)'])
IPP = IPP.set_index('Date').dropna()
IPP.drop(IPP.tail(2).index,inplace=True) # drop last n row


# In[20]:


exog= precipitaciones
exog['oni'] = oni['Oni 3.4 NOAA']
exog['TRM'] = trm['TRM']
exog['Date'] = exog.index
exog = exog[['Date','Montería','Neiva','Ibagué','Villavicencio','oni','TRM']]


# In[21]:


arroz['Price'] = arroz['Price']*IPP['Factor']
arroz['Date'] = arroz.index
arroz = arroz[['Date','Price']]


# In[22]:


arroz.to_csv (r'dataprep.csv', index = None, header=True)
exog.to_csv (r'exog.csv', index = None, header=True)


# In[ ]:
