#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 18:08:00 2024

@author: ichitakushouta
"""
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ETFs
etfs = ['SPY', 'XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLU', 'XLB']

# (a) Download and clean historical price data for each ETF
start_date = '2010-01-01'
end_date = datetime.now().strftime('%Y-%m-%d')
df = yf.download(etfs, start=start_date, end=end_date)['Adj Close']
df.dropna()

#Show the summary of the data and check if there is null value
print('(a)')
print(df.describe())
print('How many null values?:')
print(df.isnull().sum().sum())


