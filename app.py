import pandas as pd
import numpy as np
import joblib
import streamlit as st
import altair as alt

st.set_page_config(page_title="Bethacon Trading", layout="wide")
st.title("Bethacon Trading Simulator - Future Scenarios")

@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/eth-dataset-processed.csv")
    df['snapped_at'] = pd.to_datetime(df['snapped_at'])
    df = df.sort_values('snapped_at').reset_index(drop=True)
    return df

@st.cache_resource
def load_model():
    return joblib.load('model/bethacon_v1.pkl')

ds = load_data()
model = load_model()

x_columns = ['price_vs_ma7', 'price_vs_ma14', 'momentum_7', 'price_pct_change', 
             'total_volume', 'rsi', 'macd', 'signal_line', 'histogram']

st.sidebar.header("Simulation Parameters")
last_known_price = float(ds['price'].iloc[-1])
current_eth_price = st.sidebar.number_input("Current ETH Price ($)", value=last_known_price)
capital_init = st.sidebar.number_input("Initial Capital ($)", value=10000.0)
fee_pct = st.sidebar.number_input("Trading Fee (%)", value=0.1)
days_future = st.sidebar.slider("Future Days to Simulate", 10, 200, 50)
volatility = st.sidebar.slider("Estimated Daily Volatility (%)", 1.0, 10.0, 3.0) / 100.0

historical_tail = ds.tail(50).copy()
future_dates = [historical_tail['snapped_at'].iloc[-1] + pd.Timedelta(days=i+1) for i in range(days_future)]
future_prices = [current_eth_price]

for _ in range(1, days_future):
    next_price = future_prices[-1] * (1 + np.random.normal(0, volatility))
    future_prices.append(next_price)

future_df = pd.DataFrame({
    'snapped_at': future_dates,
    'price': future_prices,
    'total_volume': historical_tail['total_volume'].mean()
})

combined_df = pd.concat([historical_tail, future_df], ignore_index=True)

combined_df['ma7'] = combined_df['price'].rolling(window=7).mean()
combined_df['ma14'] = combined_df['price'].rolling(window=14).mean()
combined_df['price_vs_ma7'] = combined_df['price'] - combined_df['ma7']
combined_df['price_vs_ma14'] = combined_df['price'] - combined_df['ma14']
combined_df['momentum_7'] = combined_df['price'] - combined_df['price'].shift(7)
combined_df['price_pct_change'] = combined_df['price'].pct_change()

delta = combined_df['price'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / (loss + 1e-9)
combined_df['rsi'] = 100 - (100 / (1 + rs))

ema12 = combined_df['price'].ewm(span=12, adjust=False).mean()
ema26 = combined_df['price'].ewm(span=26, adjust=False).mean()
combined_df['macd'] = ema12 - ema26
combined_df['signal_line'] = combined_df['macd'].ewm(span=9, adjust=False).mean()
combined_df['histogram'] = combined_df['macd'] - combined_df['signal_line']

future_sim_df = combined_df.tail(days_future).copy().reset_index(drop=True)
future_sim_df = future_sim_df.fillna(0)

X_future = future_sim_df[x_columns].values
future_sim_df['signal_pred'] = model.predict(X_future)

cash = capital_init
position = 0
equity_curve = []
trade_log = []

for i, row in future_sim_df.iterrows():
    signal = row['signal_pred']
    price = row['price']
    date = row['snapped_at']
    
    if signal == 1 and cash > 0:
        position = cash / price * (1 - fee_pct/100)
        cash = 0
        trade_log.append({"Date": date, "Action": "BUY", "Price": price, "Equity": position * price})
    elif signal == 0 and position > 0:
        cash = position * price * (1 - fee_pct/100)
        position = 0
        trade_log.append({"Date": date, "Action": "SELL", "Price": price, "Equity": cash})
        
    equity_curve.append(cash + (position * price))

future_sim_df['equity'] = equity_curve

st.subheader("Future Scenario Analysis")

color_scale = alt.Scale(
    domain=['ETH Price', 'BUY Signal', 'SELL Signal'],
    range=['#1f77b4', '#00ff00', '#ff0000']
)

base = alt.Chart(future_sim_df).encode(
    x=alt.X('snapped_at:T', title='Date')
).properties(width=900)

price_line = base.mark_line(strokeWidth=2).encode(
    y=alt.Y('price:Q', title='ETH Price ($)', scale=alt.Scale(zero=False)),
    color=alt.Color('type:N', scale=color_scale, title="Legend"),
    tooltip=['snapped_at:T', 'price:Q']
).transform_calculate(type='"ETH Price"')

buy_points = base.transform_filter(alt.datum.signal_pred == 1).mark_point(
    shape='triangle-up', size=200, filled=True, opacity=1
).encode(
    y='price:Q',
    color=alt.Color('type:N'),
    tooltip=['snapped_at:T', 'price:Q']
).transform_calculate(type='"BUY Signal"')

sell_points = base.transform_filter(alt.datum.signal_pred == 0).mark_point(
    shape='triangle-down', size=200, filled=True, opacity=1
).encode(
    y='price:Q',
    color=alt.Color('type:N'),
    tooltip=['snapped_at:T', 'price:Q']
).transform_calculate(type='"SELL Signal"')

upper_chart = alt.layer(price_line, buy_points, sell_points).properties(height=350)

# Lower chart: Equity area
equity_area = base.mark_area(
    line={'color':'#2ca02c'},
    color=alt.Gradient(
        gradient='linear',
        stops=[alt.GradientStop(color='#2ca02c', offset=0),
               alt.GradientStop(color='transparent', offset=1)],
        x1=1, x2=1, y1=1, y2=0
    ),
    opacity=0.3
).encode(
    y=alt.Y('equity:Q', title='Portfolio Equity ($)', scale=alt.Scale(zero=False)),
    tooltip=['snapped_at:T', 'equity:Q']
)

# Horizontal line for Initial Capital reference
breakeven_line = alt.Chart(pd.DataFrame({'y': [capital_init]})).mark_rule(
    strokeDash=[5, 5], color='gray', strokeWidth=2
).encode(y='y:Q')

lower_chart = alt.layer(equity_area, breakeven_line).properties(height=200)

final_chart = alt.vconcat(
    upper_chart, 
    lower_chart
).resolve_scale(
    color='shared'
).configure_axis(
    labelFontSize=11,
    titleFontSize=12
).configure_legend(
    orient='top',
    padding=10
)

st.altair_chart(final_chart, use_container_width=True)

st.subheader("Forecast Statistics")
total_return = (equity_curve[-1] - capital_init) / capital_init * 100
returns = np.diff(equity_curve) / (np.array(equity_curve[:-1]) + 1e-9)
max_drawdown = np.min(equity_curve / np.maximum.accumulate(equity_curve) - 1) * 100
std_returns = np.std(returns)
sharpe = np.mean(returns) / std_returns * np.sqrt(365) if std_returns > 0 else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Initial Capital", f"${capital_init:,.2f}")
c2.metric("Estimated Final Capital", f"${equity_curve[-1]:,.2f}", f"{total_return:.2f}%")
c3.metric("Max Drawdown", f"{max_drawdown:.2f}%")
c4.metric("Sharpe Ratio", f"{sharpe:.2f}")

if trade_log:
    with st.expander("View Simulated Trade Log"):
        log_df = pd.DataFrame(trade_log)
        st.dataframe(log_df.style.format({"Price": "${:,.2f}", "Equity": "${:,.2f}"}), use_container_width=True)
else:
    st.info("No trades executed by the model during the simulated period.")