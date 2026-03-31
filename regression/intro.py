import pandas
import yfinance
import math
import numpy as np

df = yfinance.download('GOOGL', auto_adjust=True)
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

df['HL_PCT'] = (df['High'] - df['Close']) / df['Close'] * 100
df['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100

df = df[['Close', 'HL_PCT', 'PCT_change', 'Volume']]

print(df)

forecast_col = 'Close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.1*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
print(df.head)