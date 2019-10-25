#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import requests
from bs4 import BeautifulSoup


# In[19]:


# Web Scraping - https://pythonprogramminglanguage.com/web-scraping-with-pandas-and-beautifulsoup/
res = requests.get("http://www.fedearroz.com.co/new/precios.php")
soup = BeautifulSoup(res.content,'lxml')
table = soup.find_all('table')[0]
df = pd.read_html(str(table))


# In[20]:


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
arroz = arroz.tail(7)
arroz.drop(arroz.tail(1).index,inplace=True)


# In[21]:


#(Base: diciembre 2014=100)
IPP = pd.read_csv('IPP.csv',sep=';',decimal=',')
IPP['Date'] = pd.to_datetime(IPP['Año(aaaa)-Mes(mm)'])
IPP = IPP.set_index('Date').dropna()
#IPP = IPP.tail(2)


# In[22]:


arroz['Price'] = arroz['Price']*IPP['Factor']
arroz['Date'] = arroz.index
arroz = arroz[['Date','Price']]


# In[23]:


arroz.to_csv (r'data_real.csv', index = None, header=True)


# In[ ]:




