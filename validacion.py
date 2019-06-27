#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 14:57:45 2019

@author: drivera
"""
import pandas as pd
import sys


aicbic = pd.read_csv("aicbic.csv")

aic = aicbic['AIC'][0]
bic = aicbic['BIC'][0]

if aic < 738 and bic < 744:
    file1 = open("r_AIC.txt","w")
    L = ["Yes"]
    file1.writelines(L)
    file1.close()
    sys.exit(0)
else:
    file1 = open("r_AIC.txt","w")
    L = ["No"]
    file1.writelines(L)
    file1.close()
    sys.exit(1)
