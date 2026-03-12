import pandas as pd
import numpy as np
import joblib
import streamlit as st
import altair as alt

st.title("Bethacon Trading Simulator - Live Prediction (Future)")

@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/eth-dataset-processed.csv")
    df['snapped_at'] = pd.to_datetime(df['snapped_at'])
    df = df.sort_values('snapped_at')
    return df

ds = load_data()
x_columns = ['price_vs_ma7', 'price_vs_ma14', 'momentum_7', 'price_pct_change', 
             'total_volume', 'rsi', 'macd', 'signal_line', 'histogram']

model = joblib.load('model/bethacon_v1.pkl')

capital_init = st.sidebar.number_input("Initial Capital ($)", value=10000)
fee_pct = st.sidebar.number_input("Fee per trade (%)", value=0.1)
days_future = st.sidebar.slider("Future days to predict", 10, 200, 50)

# Ultimo giorno storico
last_day = ds.iloc[-1]
last_date = last_day['snapped_at']

# Copia ultima riga per base dei futuri
future_df = pd.DataFrame([last_day.copy() for _ in range(days_future)])
future_df['snapped_at'] = [last_date + pd.Timedelta(days=i+1) for i in range(days_future)]

def simulate_future_trading(future_df, model, capital, fee):
    cash = capital
    position = 0
    equity_curve = []
    signals = []

    current_price = st.sidebar.number_input("Actual Price ($)", value=float(ds['price'].iloc[-1]))
    # Ultime features
    last_features = future_df.iloc[0][x_columns].values

    for i in range(len(future_df)):
        # Predizione del segnale
        features = last_features.reshape(1, -1)
        signal = model.predict(features)[0]
        signals.append(signal)

        # Prezzo simulato: piccolo random walk a partire dall'ultimo prezzo
        if i == 0:
            price = current_price
        else:
            price = current_price * (1 + np.random.normal(0, 0.01))  # variazioni del 1% circa
        current_price = price  # aggiorniamo per il giorno successivo

        # Gestione posizione
        if signal == 1 and cash > 0:
            position = cash / price * (1 - fee/100)
            cash = 0
        elif signal == 0 and position > 0:
            cash = position * price * (1 - fee/100)
            position = 0

        total_equity = cash + position * price
        equity_curve.append(total_equity)

        # Aggiorna features per il prossimo giorno se vuoi fare un modello step-by-step più realistico
        last_features = last_features  # per ora le manteniamo costanti

    future_df['signal_pred'] = signals
    future_df['price'] = [current_price * (1 + np.random.normal(0, 0.01)) for _ in range(len(future_df))]  # opzionale
    return future_df, equity_curve

future_df, equity = simulate_future_trading(future_df, model, capital_init, fee_pct)
future_df['equity'] = equity

def plot_trading_signals(future_df):
    base = alt.Chart(future_df).encode(x='snapped_at:T')

    # Linea prezzo
    price_line = base.mark_line(color='blue').encode(
        y=alt.Y('price:Q', title='Prezzo ETH'),
        tooltip=['snapped_at:T', 'price:Q']
    )

    # Linea equity su asse Y secondario
    equity_line = base.mark_line(color='green').encode(
        y=alt.Y('equity:Q', title='Equity', axis=alt.Axis(orient='right')),
        tooltip=['snapped_at:T', 'equity:Q']
    )

    # Punti BUY
    buy_points = base.transform_filter(alt.datum.signal_pred == 1).mark_point(
        shape='triangle-up', size=80, color='green', opacity=0.8
    ).encode(
        y='price:Q',
        tooltip=['snapped_at:T', 'price:Q', 'equity:Q']
    )

    # Punti SELL
    sell_points = base.transform_filter(alt.datum.signal_pred == 0).mark_point(
        shape='triangle-down', size=80, color='red', opacity=0.8
    ).encode(
        y='price:Q',
        tooltip=['snapped_at:T', 'price:Q', 'equity:Q']
    )

    # Chart combinato
    chart = (price_line + equity_line + buy_points + sell_points).interactive()
    st.altair_chart(chart, use_container_width=True)

plot_trading_signals(future_df)
# Statistiche
def trading_stats(equity_curve, capital_init):
    returns = np.diff(equity_curve) / equity_curve[:-1]
    total_return = (equity_curve[-1] - capital_init) / capital_init * 100
    max_drawdown = np.min(equity_curve / np.maximum.accumulate(equity_curve) - 1) * 100
    sharpe_ratio = np.mean(returns) / (np.std(returns)+1e-9) * np.sqrt(252)
    return total_return, max_drawdown, sharpe_ratio

total_ret, max_dd, sharpe = trading_stats(equity, capital_init)

st.subheader("Future Forecast Statistics")
st.markdown(f"- **Initial Capital:** €{capital_init: 0.2f}")
st.markdown(f"- **Estimated Final Capital:** €{equity[-1]: 0.2f}")
st.markdown(f"- **Estimated Total Return:** {total_ret: 0.2f}%")
st.markdown(f"- **Estimated Max Drawdown:** {max_dd: 0.2f}%")
st.markdown(f"- **Estimated Sharpe Ratio:** {sharpe: 0.2f}")