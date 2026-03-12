import pandas as pd
import numpy as np
import os

# ─── Load raw data ────────────────────────────────────────────────────────────
ds = pd.read_csv('data/raw/eth-usd-max.csv')
ds['snapped_at'] = pd.to_datetime(ds['snapped_at'], yearfirst=True)
ds = ds.sort_values('snapped_at').reset_index(drop=True)
p = ds['price']

# ─── Base features ───────────────────────────────────────────────────────────
ds['price_pct_change'] = p.pct_change()
ds['price_vs_ma7']     = p - p.rolling(7).mean()
ds['price_vs_ma14']    = p - p.rolling(14).mean()

# ─── Multi-window momentum & returns ─────────────────────────────────────────
for w in [3, 5, 7, 14, 21]:
    ds[f'momentum_{w}'] = p - p.shift(w)
    ds[f'return_{w}d']  = p.pct_change(w)

# ─── Volatility ───────────────────────────────────────────────────────────────
ds['volatility_7']  = p.pct_change().rolling(7).std()
ds['volatility_14'] = p.pct_change().rolling(14).std()

# ─── RSI + derived ───────────────────────────────────────────────────────────
change   = p.diff()
avg_up   = change.clip(lower=0).rolling(14).mean()
avg_down = (-change).clip(lower=0).rolling(14).mean()
ds['rsi']            = 100 - (100 / (1 + avg_up / (avg_down + 1e-9)))
ds['rsi_ma3']        = ds['rsi'].rolling(3).mean()
ds['rsi_delta']      = ds['rsi'].diff()
ds['rsi_lag1']       = ds['rsi'].shift(1)
ds['rsi_oversold']   = (ds['rsi'] < 30).astype(int)
ds['rsi_overbought'] = (ds['rsi'] > 70).astype(int)

# ─── MACD + derived ──────────────────────────────────────────────────────────
ema12 = p.ewm(span=12, adjust=False).mean()
ema26 = p.ewm(span=26, adjust=False).mean()
ds['macd']        = ema12 - ema26
ds['signal_line'] = ds['macd'].ewm(span=9, adjust=False).mean()
ds['histogram']   = ds['macd'] - ds['signal_line']
ds['hist_slope']  = ds['histogram'].diff()
ds['hist_accel']  = ds['hist_slope'].diff()
ds['macd_cross']  = (ds['macd'] > ds['signal_line']).astype(int)
ds['macd_lag1']   = ds['macd'].shift(1)

# ─── Volume features ─────────────────────────────────────────────────────────
vol_ma7            = ds['total_volume'].rolling(7).mean()
ds['volume_ratio'] = ds['total_volume'] / (vol_ma7 + 1e-9)
ds['volume_spike'] = (ds['volume_ratio'] > 2.0).astype(int)

# ─── Price position in rolling range ─────────────────────────────────────────
p_min = p.rolling(14).min()
p_max = p.rolling(14).max()
ds['price_position'] = (p - p_min) / (p_max - p_min + 1e-9)

# ─── Lagged returns ───────────────────────────────────────────────────────────
ds['pct_lag1'] = ds['price_pct_change'].shift(1)
ds['pct_lag2'] = ds['price_pct_change'].shift(2)

# ─── Target signal ────────────────────────────────────────────────────────────
next_price        = p.shift(-1)
ds['future_return'] = (next_price - p) / p

def get_signal(row):
    if row['future_return'] > 0.03:
        return 1   # BUY
    elif row['future_return'] < -0.03:
        return 0   # SELL
    return np.nan  # HOLD escluso

ds['signal'] = ds.apply(get_signal, axis=1)
ds = ds.drop(columns=['future_return'])
ds = ds.dropna()

# ─── Save ─────────────────────────────────────────────────────────────────────
os.makedirs('data/processed', exist_ok=True)
ds.to_csv('data/processed/eth-dataset-processed.csv', index=False)
buy  = int((ds['signal'] == 1).sum())
sell = int((ds['signal'] == 0).sum())
print(f"Dataset salvato: {len(ds)} righe | BUY={buy} SELL={sell} | features={len(ds.columns)-3}")