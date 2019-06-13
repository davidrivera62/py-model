#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 14:57:45 2019

@author: drivera
"""
import pandas as pd


aicbic = pd.read_csv("aicbic.csv")

aic = aicbic['AIC'][0]
bic = aicbic['BIC'][0]

if aic < 710 and bic < 720:
    file1 = open("r_AIC.txt","w") 
    L = ["Yes"]
    file1.writelines(L)
    file1.close() 
else:
    file1 = open("r_AIC.txt","w") 
    L = ["No"]
    file1.writelines(L)
    file1.close()
