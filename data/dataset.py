import pandas as pd 
import numpy as np

ds = pd.read_csv('data/raw/eth-usd-max.csv')
ds['snapped_at'] = pd.to_datetime(ds['snapped_at'], yearfirst=True)
price = ds['price']
ds['price_pct_change'] = price.pct_change()

ma7 = ds['price'].rolling(window = 7).mean()
ma14 = ds['price'].rolling(window = 14).mean()

ds['price_vs_ma7'] = price - ma7
ds['price_vs_ma14'] = price - ma14  

next_price = price.shift(-1)
ds['momentum_7'] = price - price.shift(7)
ds['future_return'] = (next_price - price) / price

change = ds['price'].diff()
up = change.copy(); 
up[up < 0] = 0
down = -change.copy(); 
down[down < 0] = 0

avg_up = up.rolling(14).mean()
avg_down = down.rolling(14).mean()

rs = avg_up/avg_down
ds['rsi'] = 100 - (100/(1+rs))

ema12 = ds['price'].ewm(span=12).mean()
ema26 = ds['price'].ewm(span=26).mean()
ds['macd'] = ema12-ema26
ds['signal_line'] = ds['macd'].ewm(span=9).mean()
ds['histogram'] = ds['macd'] - ds['signal_line']


def get_signal(row):
    if row['future_return'] > 0.03:
        return 1  # Buy
    elif row['future_return'] < -0.03:
        return 0  # Sell    

ds['signal'] = ds.apply(get_signal, axis=1)
ds = ds.drop('future_return', axis=1)

ds = ds.dropna()

ds.to_csv('data/processed/eth-dataset-processed.csv')