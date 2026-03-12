from matplotlib import pyplot as plt
import pandas as pd

import sys
import os

# aggiungi la cartella root di Bethacon al path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

from data import dataset
from main import x_train, x_test, y_test, y_train, x


ds = dataset.ds
ds_features = ds.drop(['snapped_at'], axis=1)

def figure1(): 
    corr_matrix = ds_features.corr()
    num_features = len(corr_matrix.columns)
    plt.figure(figsize=(10,10))
    plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    num_features = len(corr_matrix.columns)
    plt.xticks(ticks=range(num_features), labels=corr_matrix.columns, rotation=90)
    plt.yticks(ticks=range(num_features), labels=corr_matrix.columns)
    plt.colorbar()
    plt.show()
    
def figure2():
    ds.sort_values('snapped_at', inplace=True)
    asse_x = ds['snapped_at']
    price = ds['price']
    ma7 = ds['ma7']
    ma14 = ds['ma14']
    plt.figure(figsize=(12,6))
    plt.title('Ethereum Price & Moving Averages')
    plt.plot(asse_x, price, color='blue', label='Price')
    plt.plot(asse_x, ma7, color='orange', label='MA7')
    plt.plot(asse_x, ma14, color='green', label='MA14')
    plt.legend()
    plt.show()
    
def plot_signals(df, models, subset_len):
    df_sorted = df.sort_values('snapped_at')
    df_subset = df_sorted.tail(subset_len)
    
    signal_marker = {0: 'v', 1: '^', 2: 'o'}
    signal_color = {0: 'red', 1: 'green', 2: 'gray'}
    
    # crea la figura e dimensioni prima di tutto
    plt.figure(figsize=(12,6))
    
    # disegno prezzo
    plt.plot(df_subset['snapped_at'], df_subset['price'], label='Price', color='blue')
    
    # disegno segnali predetti dai modelli
    for name, model in models.items():
        df_subset[f'pred_{name}'] = model.predict(df_subset[x.columns])
        signals = df_subset[f'pred_{name}']
        for sig_val in [0, 1, 2]:
            mask = signals == sig_val
            plt.scatter(
                df_subset['snapped_at'][mask],
                df_subset['price'][mask],
                marker=signal_marker[sig_val],
                color=signal_color[sig_val],
                label=f'{name} signal {sig_val}'
            )
    
    plt.legend()
    plt.title('Ethereum Price & Model Signals')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
            