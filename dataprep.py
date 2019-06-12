#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 11:02:57 2019

@author: drivera
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup

#Web Scraping - https://pythonprogramminglanguage.com/web-scraping-with-pandas-and-beautifulsoup/
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

#arroz = arroz.set_index('Date')

arroz.to_csv (r'dataprep.csv', index = None, header=True)