#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 14:52:59 2024

@author: ichitakushouta
"""


import csv
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from pandas_datareader import data
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

df_spy = pd.read_csv('HistoricalPrices SPY.csv')
df_xlb = pd.read_csv('HistoricalPrices XLB.csv')
df_xle = pd.read_csv('HistoricalPrices XLE.csv')
df_xlf = pd.read_csv('HistoricalPrices XLF.csv')
df_xli = pd.read_csv('HistoricalPrices XLI.csv')
df_xlk = pd.read_csv('HistoricalPrices XLK.csv')
df_xlp = pd.read_csv('HistoricalPrices XLP.csv')
df_xlu = pd.read_csv('HistoricalPrices XLU.csv')
df_xlv = pd.read_csv('HistoricalPrices XLV.csv')
df_xly = pd.read_csv('xlyprice.csv')

##SPY
#clean first column
df_spy.columns = ['Date','Open', 'High', 'Low', 'Close', 'Volume']

#set date column as date time
df_spy['Date'] = pd.to_datetime(df_spy['Date'])

#edit table to ascending order for date
df_spy = df_spy.sort_values('Date')

#make a new column for daily returns
df_spy['Daily Return'] = df_spy['Close'].pct_change()

#set a variable for trading days per year
trading_days_per_year = 252

#calculate annualied return
annualized_return_spy = ((1 + df_spy['Daily Return'].mean()) ** trading_days_per_year) - 1

#calculate standard deviation 
annualized_std_spy = df_spy['Daily Return'].std() * np.sqrt(trading_days_per_year)

##XLB
#clean first column
df_xlb.columns = ['Date','Open', 'High', 'Low', 'Close', 'Volume']

#set date column as date time
df_xlb['Date'] = pd.to_datetime(df_xlb['Date'])

#edit table to ascending order for date
df_xlb = df_xlb.sort_values('Date')

#make a new column for daily returns
df_xlb['Daily Return'] = df_xlb['Close'].pct_change()

#set a variable for trading days per year
trading_days_per_year = 252

#calculate annualied return
annualized_return_xlb = ((1 + df_xlb['Daily Return'].mean()) ** trading_days_per_year) - 1

#calculate standard deviation 
annualized_std_xlb = df_xlb['Daily Return'].std() * np.sqrt(trading_days_per_year)

##XLE
#clean first column
df_xle.columns = ['Date','Open', 'High', 'Low', 'Close', 'Volume']

#set date column as date time
df_xle['Date'] = pd.to_datetime(df_xle['Date'])

#edit table to ascending order for date
df_xle = df_xle.sort_values('Date')

#make a new column for daily returns
df_xle['Daily Return'] = df_xle['Close'].pct_change()

#set a variable for trading days per year
trading_days_per_year = 252

#calculate annualied return
annualized_return_xle = ((1 + df_xle['Daily Return'].mean()) ** trading_days_per_year) - 1

#calculate standard deviation 
annualized_std_xle = df_xle['Daily Return'].std() * np.sqrt(trading_days_per_year)

##XLF
#clean first column
df_xlf.columns = ['Date','Open', 'High', 'Low', 'Close', 'Volume']

#set date column as date time
df_xlf['Date'] = pd.to_datetime(df_xlf['Date'])

#edit table to ascending order for date
df_xlf = df_xlf.sort_values('Date')

#make a new column for daily returns
df_xlf['Daily Return'] = df_xlf['Close'].pct_change()

#set a variable for trading days per year
trading_days_per_year = 252

#calculate annualied return
annualized_return_xlf = ((1 + df_xlf['Daily Return'].mean()) ** trading_days_per_year) - 1

#calculate standard deviation 
annualized_std_xlf = df_xlf['Daily Return'].std() * np.sqrt(trading_days_per_year)

##XLI
#clean first column
df_xli.columns = ['Date','Open', 'High', 'Low', 'Close', 'Volume']

#set date column as date time
df_xli['Date'] = pd.to_datetime(df_xli['Date'])

#edit table to ascending order for date
df_xli = df_xli.sort_values('Date')

#make a new column for daily returns
df_xli['Daily Return'] = df_xli['Close'].pct_change()

#set a variable for trading days per year
trading_days_per_year = 252

#calculate annualied return
annualized_return_xli = ((1 + df_xli['Daily Return'].mean()) ** trading_days_per_year) - 1

#calculate standard deviation 
annualized_std_xli = df_xli['Daily Return'].std() * np.sqrt(trading_days_per_year)

##XLK
#clean first column
df_xlk.columns = ['Date','Open', 'High', 'Low', 'Close', 'Volume']

#set date column as date time
df_xlk['Date'] = pd.to_datetime(df_xlk['Date'])

#edit table to ascending order for date
df_xlk = df_xlk.sort_values('Date')

#make a new column for daily returns
df_xlk['Daily Return'] = df_xlk['Close'].pct_change()

#set a variable for trading days per year
trading_days_per_year = 252

#calculate annualied return
annualized_return_xlk = ((1 + df_xlk['Daily Return'].mean()) ** trading_days_per_year) - 1

#calculate standard deviation 
annualized_std_xlk = df_xlk['Daily Return'].std() * np.sqrt(trading_days_per_year)
 
##XLP
#clean first column
df_xlp.columns = ['Date','Open', 'High', 'Low', 'Close', 'Volume']

#set date column as date time
df_xlp['Date'] = pd.to_datetime(df_xlp['Date'])

#edit table to ascending order for date
df_xlp = df_xlp.sort_values('Date')

#make a new column for daily returns
df_xlp['Daily Return'] = df_xlp['Close'].pct_change()

#set a variable for trading days per year
trading_days_per_year = 252

#calculate annualied return
annualized_return_xlp = ((1 + df_xlp['Daily Return'].mean()) ** trading_days_per_year) - 1

#calculate standard deviation 
annualized_std_xlp = df_xlp['Daily Return'].std() * np.sqrt(trading_days_per_year)

##XLU
#clean first column
df_xlu.columns = ['Date','Open', 'High', 'Low', 'Close', 'Volume']

#set date column as date time
df_xlu['Date'] = pd.to_datetime(df_xlu['Date'])

#edit table to ascending order for date
df_xlu = df_xlu.sort_values('Date')

#make a new column for daily returns
df_xlu['Daily Return'] = df_xlu['Close'].pct_change()

#set a variable for trading days per year
trading_days_per_year = 252

#calculate annualied return
annualized_return_xlu = ((1 + df_xlu['Daily Return'].mean()) ** trading_days_per_year) - 1

#calculate standard deviation 
annualized_std_xlu = df_xlu['Daily Return'].std() * np.sqrt(trading_days_per_year)

##XLV
#clean first column
df_xlv.columns = ['Date','Open', 'High', 'Low', 'Close', 'Volume']

#set date column as date time
df_xlv['Date'] = pd.to_datetime(df_xlv['Date'])

#edit table to ascending order for date
df_xlv = df_xlv.sort_values('Date')

#make a new column for daily returns
df_xlv['Daily Return'] = df_xlv['Close'].pct_change()

#set a variable for trading days per year
trading_days_per_year = 252

#calculate annualied return
annualized_return_xlv = ((1 + df_xlv['Daily Return'].mean()) ** trading_days_per_year) - 1

#calculate standard deviation 
annualized_std_xlv = df_xlv['Daily Return'].std() * np.sqrt(trading_days_per_year)

##XLY
#clean first column
df_xly.columns = ['Date','Open', 'High', 'Low', 'Close', 'Volume']

#set date column as date time
df_xly['Date'] = pd.to_datetime(df_xly['Date'])

#edit table to ascending order for date
df_xly = df_xly.sort_values('Date')

#make a new column for daily returns
df_xly['Daily Return'] = df_xly['Close'].pct_change()

#set a variable for trading days per year
trading_days_per_year = 252

#calculate annualied return
annualized_return_xly = ((1 + df_xly['Daily Return'].mean()) ** trading_days_per_year) - 1

#calculate standard deviation 
annualized_std_xly = df_xly['Daily Return'].std() * np.sqrt(trading_days_per_year)

#make a new table for daily returns 

daily_returns = {'ETF': ['SPY','XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY'],
        'Average Daily Return': [df_spy['Daily Return'].mean(), df_xlb['Daily Return'].mean(),
                                 df_xle['Daily Return'].mean(),df_xlf['Daily Return'].mean(),
                                 df_xli['Daily Return'].mean(), df_xlk['Daily Return'].mean(),
                                 df_xlp['Daily Return'].mean(),df_xlu['Daily Return'].mean(),
                                 df_xlv['Daily Return'].mean(),df_xly['Daily Return'].mean()]}

df_daily_return_all = pd.DataFrame(daily_returns)

#make a new table for annualized returns
annual_returns = {'ETF': ['SPY','XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY'],
        'Annualized Return': [annualized_return_spy, annualized_return_xlb,annualized_return_xle,
                              annualized_return_xlf, annualized_return_xli, annualized_return_xlk,
                              annualized_return_xlp, annualized_return_xlu, annualized_return_xlv,
                              annualized_return_xly]}

df_annualized_return = pd.DataFrame(annual_returns)

#make a new table for standard deviation
std_data = {'ETF': ['SPY','XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY'],
        'Standard Deviation': [annualized_std_spy,annualized_std_xlb,annualized_std_xle,annualized_std_xlf,
                                 annualized_std_xli, annualized_std_xlk, annualized_std_xlp,
                                 annualized_std_xlu, annualized_std_xlv, annualized_std_xly]}
          
df_std = pd.DataFrame(std_data)



#show table include Date an Daily Return
df_spy_simple = df_spy[['Date','Daily Return']]
df_xlb_simple = df_xlb[['Date','Daily Return']]
df_xle_simple = df_xle[['Date','Daily Return']]
df_xlf_simple = df_xlf[['Date','Daily Return']]
df_xli_simple = df_xli[['Date','Daily Return']]
df_xlk_simple = df_xlk[['Date','Daily Return']]
df_xlp_simple = df_xlp[['Date','Daily Return']]
df_xlu_simple = df_xlu[['Date','Daily Return']]
df_xlv_simple = df_xlv[['Date','Daily Return']]
df_xly_simple = df_xly[['Date','Daily Return']]


# List of DataFrames and their corresponding ETF names
df_list = [
    (df_spy_simple, 'SPY'),
    (df_xlb_simple, 'XLB'),
    (df_xle_simple, 'XLE'),
    (df_xlf_simple, 'XLF'),
    (df_xli_simple, 'XLI'),
    (df_xlk_simple, 'XLK'),
    (df_xlp_simple, 'XLP'),
    (df_xlu_simple, 'XLU'),
    (df_xlv_simple, 'XLV'),
    (df_xly_simple, 'XLY')
]

# Start with the first DataFrame (SPY)
combined_df = df_list[0][0].rename(columns={'Daily Return': f'{df_list[0][1]}_Return'})

# Merge the rest of the DataFrames
for df, etf_name in df_list[1:]:
    df_renamed = df.rename(columns={'Daily Return': f'{etf_name}'})
    combined_df = pd.merge(combined_df, df_renamed, on='Date', how='outer')

# Sort the combined DataFrame by Date
combined_df = combined_df.sort_values('Date', ascending=True).reset_index(drop=True)

#Delete any rows that include NaN
combined_df_cleaned = combined_df.dropna()
#calculate covariance matrix
cov_matrix_daily_return = combined_df_cleaned.cov()


# Convert the Date column to datetime 
combined_df_cleaned['Date'] = pd.to_datetime(combined_df_cleaned['Date'])

# Create a new column with month/year format
combined_df_cleaned['Date'] = combined_df_cleaned['Date'].dt.strftime('%m/%Y')

# If you want to replace the original Date column:
# combined_df_cleaned['Date'] = combined_df_cleaned['Date'].dt.strftime('%m/%Y')

# Group by Month/Year and calculate the mean of daily returns for each month
monthly_returns = combined_df_cleaned.groupby('Date').mean()

monthly_returns.sort_index(inplace=True)

#covariance for monthly returns
cov_matrix_monthly_return = monthly_returns.cov()

##Calculate a rolling 90-day correlation of each sector ETF with the S&P index

merged_df_xlb = pd.merge(df_spy, df_xlb, left_index=True, right_index=True, suffixes=('_spy', '_xlb'))

# Sort the index to ensure chronological order
merged_df_xlb.sort_index(inplace=True)

# Calculate the rolling 90-day correlation with XLB
rolling_corr_xlb = merged_df_xlb['Daily Return_spy'].rolling(window=90).corr(merged_df_xlb['Daily Return_xlb'])

#XLE

merged_df_xle = pd.merge(df_spy, df_xle, left_index=True, right_index=True, suffixes=('_spy', '_xle'))

# Sort the index to ensure chronological order
merged_df_xle.sort_index(inplace=True)

# Calculate the rolling 90-day correlation with XLE
rolling_corr_xle = merged_df_xle['Daily Return_spy'].rolling(window=90).corr(merged_df_xle['Daily Return_xle'])


#XLF

merged_df_xlf = pd.merge(df_spy, df_xlf, left_index=True, right_index=True, suffixes=('_spy', '_xlf'))

# Sort the index to ensure chronological order
merged_df_xlf.sort_index(inplace=True)

# Calculate the rolling 90-day correlation with XLF
rolling_corr_xlf = merged_df_xlf['Daily Return_spy'].rolling(window=90).corr(merged_df_xlf['Daily Return_xlf'])

#XLI
merged_df_xli = pd.merge(df_spy, df_xli, left_index=True, right_index=True, suffixes=('_spy', '_xli'))

# Sort the index to ensure chronological order
merged_df_xli.sort_index(inplace=True)

# Calculate the rolling 90-day correlation with XLI
rolling_corr_xli = merged_df_xli['Daily Return_spy'].rolling(window=90).corr(merged_df_xli['Daily Return_xli'])

#XLK
merged_df_xlk = pd.merge(df_spy, df_xlk, left_index=True, right_index=True, suffixes=('_spy', '_xlk'))

# Sort the index to ensure chronological order
merged_df_xlk.sort_index(inplace=True)

# Calculate the rolling 90-day correlation with XLK
rolling_corr_xlk = merged_df_xlk['Daily Return_spy'].rolling(window=90).corr(merged_df_xlk['Daily Return_xlk'])


#XLP
merged_df_xlp = pd.merge(df_spy, df_xlp, left_index=True, right_index=True, suffixes=('_spy', '_xlp'))

# Sort the index to ensure chronological order
merged_df_xlp.sort_index(inplace=True)

# Calculate the rolling 90-day correlation with XLP
rolling_corr_xlp = merged_df_xlp['Daily Return_spy'].rolling(window=90).corr(merged_df_xlp['Daily Return_xlp'])

#XLU
merged_df_xlu = pd.merge(df_spy, df_xlu, left_index=True, right_index=True, suffixes=('_spy', '_xlu'))

# Sort the index to ensure chronological order
merged_df_xlu.sort_index(inplace=True)

# Calculate the rolling 90-day correlation with XLU
rolling_corr_xlu = merged_df_xlu['Daily Return_spy'].rolling(window=90).corr(merged_df_xlu['Daily Return_xlu'])

#XLV
merged_df_xlv = pd.merge(df_spy, df_xlv, left_index=True, right_index=True, suffixes=('_spy', '_xlv'))

# Sort the index to ensure chronological order
merged_df_xlv.sort_index(inplace=True)

# Calculate the rolling 90-day correlation with XLV
rolling_corr_xlv = merged_df_xlv['Daily Return_spy'].rolling(window=90).corr(merged_df_xlv['Daily Return_xlv'])

#XLY
merged_df_xly = pd.merge(df_spy, df_xly, left_index=True, right_index=True, suffixes=('_spy', '_xly'))

# Sort the index to ensure chronological order
merged_df_xly.sort_index(inplace=True)

# Calculate the rolling 90-day correlation with XLY
rolling_corr_xly = merged_df_xly['Daily Return_spy'].rolling(window=90).corr(merged_df_xly['Daily Return_xly'])

##CAPM model

x = combined_df_cleaned['SPY_Return']
y1 = combined_df_cleaned['XLB']

import statsmodels.api as sm

#Culculate Beta for XLB
x_sm = sm.add_constant(x)

model = sm.OLS(y1,x_sm) 

result_xlb = model.fit()

#Culculate Beta for XLE
y2 = combined_df_cleaned['XLE']

model2 = sm.OLS(y2,x_sm) 

result_xle = model2.fit()
  
#Culculate Beta for XLF
y3 = combined_df_cleaned['XLF']

model3 = sm.OLS(y3,x_sm) 

result_xlf = model3.fit()

#Culculate Beta for XLI
y4 = combined_df_cleaned['XLI']

model4 = sm.OLS(y4,x_sm) 

result_xli = model4.fit()

#Culculate Beta for XLF
y5 = combined_df_cleaned['XLK']

model5 = sm.OLS(y5,x_sm) 

result_xlk = model5.fit()

#Culculate Beta for XLP
y6 = combined_df_cleaned['XLP']

model6 = sm.OLS(y6,x_sm) 

result_xlp = model6.fit()

#Culculate Beta for XLU
y7 = combined_df_cleaned['XLU']

model7 = sm.OLS(y7,x_sm) 

result_xlu = model7.fit()

#Culculate Beta for XLV
y8 = combined_df_cleaned['XLV']

model8 = sm.OLS(y8,x_sm) 

result_xlv = model8.fit()

#Culculate Beta for XLY
y9 = combined_df_cleaned['XLY']

model9 = sm.OLS(y9,x_sm) 

result_xly = model9.fit()


# rolling_90_return = pd.DataFrame(index=combined_df_cleaned.index)
# etfs = ['SPY_Return','XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY']

# for etf in etfs:
#    rolling_90_return[etf] = combined_df_cleaned.rolling(window=90)



##CAPM model Beta'

# x = rolling_corr_cleaned['SPY_Return']
# y1 = rolling_corr_cleaned['XLB']

##I changed something



