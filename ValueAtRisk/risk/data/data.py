import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean
import datetime as datet
from pandas_datareader import data as pandasdr
import yfinance as yf


# Storing data in CSV files in case of API failure.

end = datet.datetime.now()
start = end - datet.timedelta(days=2000)

symbols = ['AAPL', 'TSLA', 'MSFT', 'IBM', 'SAP', 'ORCL']


yf.pdr_override()
for i in symbols:
    pandasdr.get_data_yahoo(symbols, end=end, start=start)['Close'].to_csv(i+'.csv')
