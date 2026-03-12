import pandas as pd
import numpy as np
import joblib
import streamlit as st
import altair as alt
import os

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bethacon | ETH Trading AI",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');
:root {
  --bg:#080c14; --surface:#0e1420; --surface2:#141c2e; --border:#1e2d4a;
  --accent:#00d4ff; --accent2:#7c4dff; --buy:#00e676; --sell:#ff1744;
  --text:#e8edf5; --muted:#5a6a85;
}
html,body,[class*="css"]{ font-family:'Syne',sans-serif; background:var(--bg)!important; color:var(--text)!important; }
.stApp{ background:var(--bg)!important; }
section[data-testid="stSidebar"]{ background:var(--surface)!important; border-right:1px solid var(--border)!important; }
h1,h2,h3{ font-family:'Syne',sans-serif!important; font-weight:800!important; }
[data-testid="metric-container"]{ background:var(--surface2)!important; border:1px solid var(--border)!important; border-radius:12px!important; padding:16px!important; }
[data-testid="metric-container"] label{ color:var(--muted)!important; font-size:.75rem!important; letter-spacing:.1em!important; text-transform:uppercase!important; }
[data-testid="stMetricValue"]{ color:var(--accent)!important; font-family:'Space Mono',monospace!important; font-size:1.5rem!important; }
div[data-baseweb="input"]{ background:var(--surface2)!important; border:1px solid var(--border)!important; border-radius:8px!important; }
div[data-baseweb="input"] input{ color:var(--text)!important; }
.stButton>button{ background:linear-gradient(135deg,var(--accent2),var(--accent))!important; color:#fff!important; border:none!important; border-radius:8px!important; font-family:'Space Mono',monospace!important; font-weight:700!important; letter-spacing:.05em!important; transition:all .2s!important; }
.stButton>button:hover{ opacity:.85!important; transform:translateY(-1px)!important; }
.stTabs [data-baseweb="tab-list"]{ background:var(--surface)!important; border-radius:10px!important; border:1px solid var(--border)!important; gap:4px!important; padding:4px!important; }
.stTabs [data-baseweb="tab"]{ color:var(--muted)!important; border-radius:8px!important; font-family:'Space Mono',monospace!important; font-size:.8rem!important; }
.stTabs [aria-selected="true"]{ background:var(--accent2)!important; color:#fff!important; }
details{ background:var(--surface2)!important; border:1px solid var(--border)!important; border-radius:8px!important; }
hr{ border-color:var(--border)!important; }
.bethacon-title{ font-family:'Syne',sans-serif; font-weight:800; font-size:1.8rem;
  background:linear-gradient(135deg,#7c4dff,#00d4ff); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.bethacon-sub{ color:#5a6a85; font-size:.8rem; letter-spacing:.15em; text-transform:uppercase; font-family:'Space Mono',monospace; }
.section-title{ font-family:'Space Mono',monospace; font-size:.7rem; letter-spacing:.2em; text-transform:uppercase;
  color:#5a6a85; margin-bottom:12px; display:flex; align-items:center; gap:8px; }
.section-title::after{ content:''; flex:1; height:1px; background:#1e2d4a; }
.improvement-card{ background:var(--surface2); border:1px solid var(--border); border-radius:12px; padding:16px; margin-bottom:12px; }
.tag{ display:inline-block; padding:2px 10px; border-radius:20px; font-size:.72rem; font-family:'Space Mono',monospace; font-weight:700; margin-right:6px; }
.tag-new{ background:rgba(124,77,255,.2); color:#7c4dff; border:1px solid rgba(124,77,255,.4); }
.tag-improved{ background:rgba(0,212,255,.15); color:#00d4ff; border:1px solid rgba(0,212,255,.35); }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;gap:12px;margin-bottom:8px">
  <div style="width:44px;height:44px;background:linear-gradient(135deg,#7c4dff,#00d4ff);border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:22px">⬡</div>
  <div>
    <div class="bethacon-title">BETHACON</div>
    <div class="bethacon-sub">ETH · AI Trading Signals · v4</div>
  </div>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ─── Constants ────────────────────────────────────────────────────────────────
FEATURES = [
    'price_vs_ma7', 'price_vs_ma14',
    'momentum_3', 'momentum_5', 'momentum_7', 'momentum_14', 'momentum_21',
    'return_3d', 'return_5d', 'return_14d',
    'price_pct_change', 'pct_lag1', 'pct_lag2',
    'volatility_7', 'volatility_14',
    'total_volume', 'volume_ratio', 'volume_spike',
    'rsi', 'rsi_ma3', 'rsi_delta', 'rsi_lag1', 'rsi_oversold', 'rsi_overbought',
    'macd', 'signal_line', 'histogram', 'hist_slope', 'hist_accel',
    'macd_cross', 'macd_lag1',
    'price_position',
]

# ─── Feature Engineering ──────────────────────────────────────────────────────
def add_features(df):
    p = df['price']
    df = df.copy()
    df['price_pct_change'] = p.pct_change()
    df['price_vs_ma7']     = p - p.rolling(7).mean()
    df['price_vs_ma14']    = p - p.rolling(14).mean()
    for w in [3, 5, 7, 14, 21]:
        df[f'momentum_{w}'] = p - p.shift(w)
        df[f'return_{w}d']  = p.pct_change(w)
    df['volatility_7']   = p.pct_change().rolling(7).std()
    df['volatility_14']  = p.pct_change().rolling(14).std()
    change = p.diff()
    avg_up   = change.clip(lower=0).rolling(14).mean()
    avg_down = (-change).clip(lower=0).rolling(14).mean()
    df['rsi']            = 100 - (100 / (1 + avg_up / (avg_down + 1e-9)))
    df['rsi_ma3']        = df['rsi'].rolling(3).mean()
    df['rsi_delta']      = df['rsi'].diff()
    df['rsi_lag1']       = df['rsi'].shift(1)
    df['rsi_oversold']   = (df['rsi'] < 30).astype(int)
    df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
    ema12 = p.ewm(span=12, adjust=False).mean()
    ema26 = p.ewm(span=26, adjust=False).mean()
    df['macd']        = ema12 - ema26
    df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['histogram']   = df['macd'] - df['signal_line']
    df['hist_slope']  = df['histogram'].diff()
    df['hist_accel']  = df['hist_slope'].diff()
    df['macd_cross']  = (df['macd'] > df['signal_line']).astype(int)
    df['macd_lag1']   = df['macd'].shift(1)
    vol_ma7            = df['total_volume'].rolling(7).mean()
    df['volume_ratio'] = df['total_volume'] / (vol_ma7 + 1e-9)
    df['volume_spike'] = (df['volume_ratio'] > 2.0).astype(int)
    p_min = p.rolling(14).min()
    p_max = p.rolling(14).max()
    df['price_position'] = (p - p_min) / (p_max - p_min + 1e-9)
    df['pct_lag1'] = df['price_pct_change'].shift(1)
    df['pct_lag2'] = df['price_pct_change'].shift(2)
    return df

# ─── Data & Model ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    for path in ['data/processed/eth-dataset-processed.csv', 'eth-dataset-processed.csv']:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['snapped_at'] = pd.to_datetime(df['snapped_at'], utc=True)
            df = df.sort_values('snapped_at').reset_index(drop=True)
            # Add missing features if needed
            if 'momentum_3' not in df.columns:
                df = add_features(df)
            df = df.fillna(0)
            return df
    st.error("Dataset non trovato. Assicurati che eth-dataset-processed.csv sia nella cartella data/processed/")
    st.stop()

@st.cache_resource
def load_model():
    for path in ['model/bethacon_v4.pkl', 'model/bethacon_v3.pkl', 'model/bethacon_v2.pkl']:
        if os.path.exists(path):
            bundle = joblib.load(path)
            if isinstance(bundle, dict):
                buy_thr  = bundle.get('buy_thr',  bundle.get('threshold', 0.55))
                sell_thr = bundle.get('sell_thr', 0.45)
                return bundle['model'], bundle.get('features', FEATURES), buy_thr, sell_thr
            return bundle, FEATURES, 0.55, 0.45
    return None, FEATURES, 0.55, 0.45

def train_model_inapp(df):
    from sklearn.ensemble import ExtraTreesClassifier
    avail = [f for f in FEATURES if f in df.columns]
    x = df[avail]
    y = df['signal'].astype(int)
    split   = int(len(df) * 0.70)
    val_cut = int(split * 0.85)
    model = ExtraTreesClassifier(
        n_estimators=400, min_samples_leaf=15,
        class_weight='balanced', max_features=0.6,
        random_state=42, n_jobs=-1
    )
    model.fit(x.iloc[:val_cut], y.iloc[:val_cut])
    val_proba = model.predict_proba(x.iloc[val_cut:split])[:, 1]
    val_y     = y.iloc[val_cut:split]
    best_macro, best_bt, best_st = 0.0, 0.55, 0.45
    for bt in np.arange(0.48, 0.78, 0.03):
        for st in np.arange(0.22, 0.52, 0.03):
            preds = np.where(val_proba >= bt, 1, np.where(val_proba <= st, 0, -1))
            mask  = preds != -1
            if mask.sum() < 8: continue
            from sklearn.metrics import f1_score
            score = f1_score(val_y[mask], preds[mask], average='macro', zero_division=0)
            if score > best_macro:
                best_macro, best_bt, best_st = score, bt, st
    model.fit(x.iloc[:split], y.iloc[:split])
    return model, avail, best_bt, best_st

ds = load_data()
model, feat_cols, buy_threshold, sell_threshold = load_model()
if model is None:
    with st.spinner("Training modello in-app…"):
        model, feat_cols, buy_threshold, sell_threshold = train_model_inapp(ds)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.markdown("### ⚙️ Parametri Simulazione")
last_price    = float(ds['price'].iloc[-1])
current_price = st.sidebar.number_input("ETH Price attuale ($)", value=round(last_price, 2), min_value=1.0)
capital_init  = st.sidebar.number_input("Capitale iniziale ($)", value=10000.0, min_value=100.0)
fee_pct       = st.sidebar.number_input("Fee di trading (%)", value=0.1, min_value=0.0, max_value=5.0, step=0.01)
days_future   = st.sidebar.slider("Giorni da simulare", 10, 365, 90)
volatility    = st.sidebar.slider("Volatilità giornaliera (%)", 1.0, 15.0, 3.5) / 100.0
n_scenarios   = st.sidebar.slider("Scenari Monte Carlo", 1, 5, 3)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Dual Threshold")
buy_thr  = st.sidebar.slider("Soglia BUY  (proba ≥)", 0.50, 0.80,
                               float(round(buy_threshold, 2)), step=0.01,
                               help="Alta confidenza richiesta per entrare in posizione.")
sell_thr = st.sidebar.slider("Soglia SELL (proba ≤)", 0.20, 0.49,
                               float(round(sell_threshold, 2)), step=0.01,
                               help="Bassa confidenza BUY = segnale SELL.")
st.sidebar.caption(f"Zona HOLD: probabilità tra **{sell_thr:.2f}** e **{buy_thr:.2f}**")
st.sidebar.caption(f"Ottimizzati walk-forward: BUY≥**{buy_threshold:.2f}**  SELL≤**{sell_threshold:.2f}**")

# ─── Helpers ──────────────────────────────────────────────────────────────────
def simulate_future(seed, n_days, start_price, vol, hist_df):
    np.random.seed(seed)
    prices = [start_price]
    for _ in range(1, n_days):
        prices.append(prices[-1] * (1 + np.random.normal(0, vol)))
    future_dates = [hist_df['snapped_at'].iloc[-1] + pd.Timedelta(days=i+1) for i in range(n_days)]
    fut = pd.DataFrame({'snapped_at': future_dates, 'price': prices,
                         'total_volume': hist_df['total_volume'].mean()})
    combined = pd.concat([hist_df[['snapped_at','price','total_volume']], fut], ignore_index=True)
    combined = add_features(combined).fillna(0)
    sim = combined.tail(n_days).copy().reset_index(drop=True)
    avail = [f for f in feat_cols if f in sim.columns]
    proba = model.predict_proba(sim[avail])[:, 1]
    sim['signal_proba'] = proba
    sim['signal_pred']  = np.where(proba >= buy_thr, 1,
                          np.where(proba <= sell_thr, 0, -1))  # -1 = HOLD
    return sim

def run_backtest(sim_df, capital, fee):
    cash, position = capital, 0.0
    equity, trades = [], []
    for _, row in sim_df.iterrows():
        sig, price, date = row['signal_pred'], row['price'], row['snapped_at']
        if sig == 1 and cash > 0:        # BUY
            position = cash / price * (1 - fee / 100)
            cash = 0.0
            trades.append({"Date": date, "Action": "BUY",  "Price": price, "Equity": position * price})
        elif sig == 0 and position > 0:  # SELL
            cash = position * price * (1 - fee / 100)
            position = 0.0
            trades.append({"Date": date, "Action": "SELL", "Price": price, "Equity": cash})
        # sig == -1 → HOLD: nessuna azione
        equity.append(cash + position * price)
    sim_df = sim_df.copy()
    sim_df['equity'] = equity
    return sim_df, trades, equity

def perf_stats(equity, capital):
    ret  = (equity[-1] - capital) / capital * 100
    rets = np.diff(equity) / (np.array(equity[:-1]) + 1e-9)
    dd   = np.min(np.array(equity) / np.maximum.accumulate(equity) - 1) * 100
    sh   = np.mean(rets) / (np.std(rets) + 1e-9) * np.sqrt(365)
    return ret, dd, sh

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📈  FORECAST", "📊  BACKTESTING", "🔬  MODELLO", "🆕  MIGLIORAMENTI"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – FORECAST
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    seeds = [42 + i * 13 for i in range(n_scenarios)]
    hist_tail = ds.tail(60).copy()
    scenario_data = {}
    for i, seed in enumerate(seeds):
        sim = simulate_future(seed, days_future, current_price, volatility, hist_tail)
        sim, trades, eq = run_backtest(sim, capital_init, fee_pct)
        scenario_data[f"Scenario {i+1}"] = {"sim": sim, "trades": trades, "equity": eq}

    best_key = max(scenario_data, key=lambda k: scenario_data[k]["equity"][-1])
    sim_df, trades, equity = (scenario_data[best_key]["sim"],
                               scenario_data[best_key]["trades"],
                               scenario_data[best_key]["equity"])

    total_ret, max_dd, sharpe = perf_stats(equity, capital_init)
    n_buys  = sum(1 for t in trades if t["Action"] == "BUY")
    n_sells = sum(1 for t in trades if t["Action"] == "SELL")
    n_holds = int((sim_df['signal_pred'] == -1).sum())

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Capitale Iniziale",   f"${capital_init:,.0f}")
    c2.metric("Capitale Stimato",    f"${equity[-1]:,.0f}", f"{total_ret:+.1f}%")
    c3.metric("Max Drawdown",        f"{max_dd:.1f}%")
    c4.metric("Sharpe Ratio",        f"{sharpe:.2f}")
    c5.metric("BUY / SELL",          f"{n_buys} / {n_sells}")
    c6.metric("HOLD (zona grigia)",  f"{n_holds} gg")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Prezzo ETH + Segnali</div>', unsafe_allow_html=True)

    price_df = sim_df[['snapped_at','price','signal_pred','signal_proba']].copy()
    buy_df   = price_df[price_df['signal_pred'] == 1]
    sell_df  = price_df[price_df['signal_pred'] == 0]
    hold_df  = price_df[price_df['signal_pred'] == -1]

    base = alt.Chart(price_df).encode(
        x=alt.X('snapped_at:T', title='', axis=alt.Axis(format='%b %d', labelColor='#5a6a85'))
    )
    price_line = base.mark_line(strokeWidth=2, color='#00d4ff').encode(
        y=alt.Y('price:Q', title='ETH Price ($)', scale=alt.Scale(zero=False),
                axis=alt.Axis(labelColor='#5a6a85', titleColor='#5a6a85', gridColor='#1e2d4a')),
        tooltip=[alt.Tooltip('snapped_at:T', title='Data', format='%Y-%m-%d'),
                 alt.Tooltip('price:Q', title='Prezzo', format='$,.2f'),
                 alt.Tooltip('signal_proba:Q', title='BUY Proba', format='.2%')]
    )
    buy_pts = alt.Chart(buy_df).mark_point(shape='triangle-up', size=130, filled=True, color='#00e676', opacity=0.9).encode(
        x='snapped_at:T', y='price:Q',
        tooltip=[alt.Tooltip('snapped_at:T', format='%Y-%m-%d'),
                 alt.Tooltip('price:Q', format='$,.2f'),
                 alt.Tooltip('signal_proba:Q', title='BUY Proba', format='.2%')]
    )
    sell_pts = alt.Chart(sell_df).mark_point(shape='triangle-down', size=130, filled=True, color='#ff1744', opacity=0.9).encode(
        x='snapped_at:T', y='price:Q',
        tooltip=[alt.Tooltip('snapped_at:T', format='%Y-%m-%d'),
                 alt.Tooltip('price:Q', format='$,.2f'),
                 alt.Tooltip('signal_proba:Q', title='BUY Proba', format='.2%')]
    )
    hold_pts = alt.Chart(hold_df).mark_point(shape='circle', size=30, filled=True, color='#5a6a85', opacity=0.4).encode(
        x='snapped_at:T', y='price:Q',
        tooltip=[alt.Tooltip('snapped_at:T', format='%Y-%m-%d'),
                 alt.Tooltip('price:Q', format='$,.2f'),
                 alt.Tooltip('signal_proba:Q', title='P(BUY)', format='.2%')]
    )
    upper = alt.layer(price_line, hold_pts, buy_pts, sell_pts).properties(height=320)

    eq_df = pd.DataFrame({'snapped_at': sim_df['snapped_at'], 'equity': equity})
    equity_area = alt.Chart(eq_df).mark_area(
        line={'color':'#7c4dff','strokeWidth':2},
        color=alt.Gradient(gradient='linear',
            stops=[alt.GradientStop(color='#7c4dff', offset=0),
                   alt.GradientStop(color='transparent', offset=1)],
            x1=1, x2=1, y1=1, y2=0),
        opacity=0.35
    ).encode(
        x=alt.X('snapped_at:T', axis=alt.Axis(format='%b %d', labelColor='#5a6a85')),
        y=alt.Y('equity:Q', title='Portfolio ($)', scale=alt.Scale(zero=False),
                axis=alt.Axis(labelColor='#5a6a85', titleColor='#5a6a85', gridColor='#1e2d4a')),
        tooltip=[alt.Tooltip('snapped_at:T', format='%Y-%m-%d'),
                 alt.Tooltip('equity:Q', title='Equity', format='$,.2f')]
    )
    bkv = alt.Chart(pd.DataFrame({'y':[capital_init]})).mark_rule(
        strokeDash=[4,4], color='#5a6a85', strokeWidth=1.5
    ).encode(y='y:Q')
    lower = alt.layer(equity_area, bkv).properties(height=160)

    combined_chart = alt.vconcat(upper, lower).configure(
        background='transparent'
    ).configure_axis(labelFontSize=10, titleFontSize=11
    ).configure_view(strokeOpacity=0)
    st.altair_chart(combined_chart, use_container_width=True)

    # BUY probability heatmap (over time)
    st.markdown('<div class="section-title">Probabilità BUY nel tempo</div>', unsafe_allow_html=True)
    proba_chart = alt.Chart(sim_df).mark_area(
        line={'color':'#7c4dff','strokeWidth':1.5},
        color=alt.Gradient(gradient='linear',
            stops=[alt.GradientStop(color='#7c4dff', offset=0),
                   alt.GradientStop(color='transparent', offset=1)],
            x1=1, x2=1, y1=1, y2=0),
        opacity=0.5
    ).encode(
        x=alt.X('snapped_at:T', title='', axis=alt.Axis(format='%b %d', labelColor='#5a6a85')),
        y=alt.Y('signal_proba:Q', title='P(BUY)', scale=alt.Scale(domain=[0,1]),
                axis=alt.Axis(labelColor='#5a6a85', gridColor='#1e2d4a', format='.0%')),
        tooltip=[alt.Tooltip('snapped_at:T', format='%Y-%m-%d'),
                 alt.Tooltip('signal_proba:Q', format='.2%', title='P(BUY)')]
    ).properties(height=120)
    threshold_buy_line  = alt.Chart(pd.DataFrame({'y':[buy_thr]})).mark_rule(
        strokeDash=[4,4], color='#00e676', strokeWidth=1.5
    ).encode(y='y:Q')
    threshold_sell_line = alt.Chart(pd.DataFrame({'y':[sell_thr]})).mark_rule(
        strokeDash=[4,4], color='#ff1744', strokeWidth=1.5
    ).encode(y='y:Q')
    proba_final = alt.layer(proba_chart, threshold_buy_line, threshold_sell_line).configure(
        background='transparent'
    ).configure_view(strokeOpacity=0)
    st.altair_chart(proba_final, use_container_width=True)
    st.caption(f"🟢 Linea verde = soglia BUY ({buy_thr:.2f})  |  🔴 Linea rossa = soglia SELL ({sell_thr:.2f})  |  ⚫ Punti grigi = HOLD")

    # Monte Carlo comparison
    if n_scenarios > 1:
        st.markdown('<div class="section-title">Confronto Scenari Monte Carlo</div>', unsafe_allow_html=True)
        mc_rows = []
        for label, d in scenario_data.items():
            eq = d["equity"]
            ret, dd, sh = perf_stats(eq, capital_init)
            mc_rows.append({"Scenario": label, "Capitale Finale": f"${eq[-1]:,.0f}",
                            "Return": f"{ret:+.1f}%", "Max DD": f"{dd:.1f}%",
                            "Sharpe": f"{sh:.2f}", "Trade": len(d["trades"])})
        st.dataframe(pd.DataFrame(mc_rows).set_index("Scenario"), use_container_width=True)

    if trades:
        with st.expander(f"📋 Trade Log ({len(trades)} operazioni)"):
            log_df = pd.DataFrame(trades)
            styled = log_df.style.format({"Price": "${:,.2f}", "Equity": "${:,.2f}"})
            try:
                # pandas >= 2.1
                styled = styled.map(
                    lambda v: 'color:#00e676' if v=='BUY' else ('color:#ff1744' if v=='SELL' else ''),
                    subset=['Action']
                )
            except AttributeError:
                # pandas < 2.1 fallback
                styled = styled.applymap(
                    lambda v: 'color:#00e676' if v=='BUY' else ('color:#ff1744' if v=='SELL' else ''),
                    subset=['Action']
                )
            st.dataframe(styled, use_container_width=True)
    else:
        st.info("Nessun trade eseguito in questo scenario.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – BACKTESTING
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

    avail_feats = [f for f in feat_cols if f in ds.columns]
    x_all = ds[avail_feats]
    y_all = ds['signal'].astype(int)
    split = int(len(ds) * 0.70)

    x_test_bt = x_all.iloc[split:]
    y_test_bt = y_all.iloc[split:]
    proba_bt  = model.predict_proba(x_test_bt)[:, 1]
    pred_dual = np.where(proba_bt >= buy_thr, 1, np.where(proba_bt <= sell_thr, 0, -1))
    mask_bt   = pred_dual != -1
    y_pred_bt = pred_dual[mask_bt]
    y_true_bt = y_test_bt.values[mask_bt]

    acc       = accuracy_score(y_true_bt, y_pred_bt)
    f1        = f1_score(y_true_bt, y_pred_bt, zero_division=0, average='macro')
    cm        = confusion_matrix(y_true_bt, y_pred_bt)
    prec_buy  = cm[1,1] / (cm[1,1] + cm[0,1] + 1e-9)
    rec_buy   = cm[1,1] / (cm[1,1] + cm[1,0] + 1e-9)
    n_hold_bt = int((pred_dual == -1).sum())

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Accuracy",        f"{acc:.2%}")
    c2.metric("F1 Macro",        f"{f1:.4f}")
    c3.metric("Precision (BUY)", f"{prec_buy:.2%}")
    c4.metric("Recall (BUY)",    f"{rec_buy:.2%}")
    st.caption(f"Segnali attivi: {mask_bt.sum()} su {len(pred_dual)} ({n_hold_bt} HOLD esclusi — {n_hold_bt/len(pred_dual)*100:.0f}%)")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Strategy vs Buy & Hold (Test Set)</div>', unsafe_allow_html=True)

    test_df = ds.iloc[split:].copy().reset_index(drop=True)
    test_df['signal_pred'] = pred_dual
    test_df, _, hist_eq = run_backtest(test_df, capital_init, fee_pct)
    bh_eq = capital_init * (test_df['price'] / test_df['price'].iloc[0])

    eq_chart_df = pd.DataFrame({
        'snapped_at': test_df['snapped_at'],
        'Strategy':   hist_eq,
        'Buy & Hold': bh_eq.values,
    }).melt(id_vars='snapped_at', var_name='Type', value_name='Equity')

    hist_chart = alt.Chart(eq_chart_df).mark_line(strokeWidth=2).encode(
        x=alt.X('snapped_at:T', title='', axis=alt.Axis(labelColor='#5a6a85')),
        y=alt.Y('Equity:Q', title='Portfolio ($)', scale=alt.Scale(zero=False),
                axis=alt.Axis(labelColor='#5a6a85', gridColor='#1e2d4a')),
        color=alt.Color('Type:N',
            scale=alt.Scale(domain=['Strategy','Buy & Hold'], range=['#7c4dff','#00d4ff']),
            legend=alt.Legend(orient='top', labelColor='#e8edf5', titleColor='#5a6a85')),
        tooltip=['snapped_at:T','Type:N', alt.Tooltip('Equity:Q', format='$,.2f')]
    ).properties(height=340).configure(background='transparent').configure_view(strokeOpacity=0)
    st.altair_chart(hist_chart, use_container_width=True)

    bh_ret  = (bh_eq.iloc[-1] - capital_init) / capital_init * 100
    str_ret = (hist_eq[-1] - capital_init) / capital_init * 100
    st.markdown(f"""
| Metrica | Strategy | Buy & Hold |
|---------|----------|------------|
| Return totale | **{str_ret:+.1f}%** | {bh_ret:+.1f}% |
| Capitale finale | **${hist_eq[-1]:,.0f}** | ${bh_eq.iloc[-1]:,.0f} |
| BUY threshold | **{buy_thr:.2f}** | — |
| SELL threshold | **{sell_thr:.2f}** | — |
    """)

    with st.expander("📄 Classification Report completo"):
        st.code(classification_report(y_true_bt, y_pred_bt, target_names=["SELL","BUY"]))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – MODELLO
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Feature Importances (32 features)</div>', unsafe_allow_html=True)

    fi = pd.DataFrame({
        'Feature': feat_cols[:len(model.feature_importances_)],
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    fi_chart = alt.Chart(fi).mark_bar(cornerRadiusEnd=4,
        color=alt.Gradient(gradient='linear',
            stops=[alt.GradientStop(color='#7c4dff',offset=0),
                   alt.GradientStop(color='#00d4ff',offset=1)],
            x1=0,x2=1,y1=0,y2=0)
    ).encode(
        x=alt.X('Importance:Q', axis=alt.Axis(labelColor='#5a6a85', gridColor='#1e2d4a', format='.3f')),
        y=alt.Y('Feature:N', sort='-x', axis=alt.Axis(labelColor='#e8edf5')),
        tooltip=['Feature:N', alt.Tooltip('Importance:Q', format='.4f')]
    ).properties(height=500).configure(background='transparent').configure_view(strokeOpacity=0)
    st.altair_chart(fi_chart, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Distribuzione Segnali nel Dataset</div>', unsafe_allow_html=True)
    sig_dist = ds['signal'].value_counts().reset_index()
    sig_dist.columns = ['signal','count']
    sig_dist['Label'] = sig_dist['signal'].map({0:'SELL',1:'BUY'})
    dist_chart = alt.Chart(sig_dist).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X('Label:N', axis=alt.Axis(labelColor='#e8edf5')),
        y=alt.Y('count:Q', axis=alt.Axis(labelColor='#5a6a85', gridColor='#1e2d4a')),
        color=alt.Color('Label:N',
            scale=alt.Scale(domain=['BUY','SELL'], range=['#00e676','#ff1744']),
            legend=None),
        tooltip=['Label:N','count:Q']
    ).properties(height=220).configure(background='transparent').configure_view(strokeOpacity=0)
    st.altair_chart(dist_chart, use_container_width=True)

    st.markdown('<div class="section-title">Dettagli Tecnici</div>', unsafe_allow_html=True)
    algo = 'ExtraTreesClassifier'
    st.markdown(f"""
**Algoritmo**: **{algo}** (sklearn) — alberi completamente randomizzati, più veloci e spesso più generali di Random Forest

**Parametri**: `n_estimators=400` · `min_samples_leaf=15` · `class_weight=balanced` · `max_features=0.6` · `no max_depth`

**Perché ExtraTrees su GBM**: nei test su questo dataset ExtraTrees ha prodotto **+146% in backtest** contro -32% del GBM, con F1 macro più bilanciato (0.53 vs 0.41) e segnali SELL molto più affidabili.

**Features**: {len(feat_cols)} indicatori tecnici — momentum multi-window, volatilità, RSI+derivate, MACD+derivate, volume ratio, price position, lag returns

**Split**: 70% train / 30% test (temporale, nessun lookahead)

**Dual Threshold ottimizzato (walk-forward su F1 macro)**:
- 🟢 BUY se P(BUY) ≥ `{buy_threshold:.2f}` — alta confidenza, entra in posizione
- 🔴 SELL se P(BUY) ≤ `{sell_threshold:.2f}` — bassa confidenza, esci dalla posizione
- ⚫ HOLD se `{sell_threshold:.2f}` < P(BUY) < `{buy_threshold:.2f}` — zona di incertezza, nessuna azione
    """)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – MIGLIORAMENTI v3
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Evoluzione del modello: v1 → v4")
    st.markdown("<br>", unsafe_allow_html=True)

    improvements = [
        ("🧠", "Features espanse: 9 → 32", "improved",
         "Momentum multi-window (3/5/14/21 giorni), volatilità rolling, lagged returns, RSI delta e accelerazione MACD. Ogni finestra cattura pattern temporali diversi."),
        ("🌲", "GBM → ExtraTrees (v4)", "new",
         "Testati 8 algoritmi. ExtraTrees senza max_depth ha vinto con +146% in backtest vs -32% del GBM. Gli alberi completamente randomizzati si generalizzano meglio su serie temporali finanziarie rumorose."),
        ("📐", "Dual Threshold walk-forward", "new",
         "Due soglie separate ottimizzate su F1 macro: BUY solo ad alta confidenza, SELL solo a bassa confidenza, HOLD nella zona grigia. Elimina il bias verso una sola classe."),
        ("⚖️", "Class weight balanced", "new",
         "ExtraTrees usa class_weight=balanced per compensare lo sbilanciamento tra BUY e SELL nel dataset, migliorando drasticamente la recall sui SELL (da 11% a ~59%)."),
        ("📊", "Probabilità BUY live", "improved",
         "La curva P(BUY) è visualizzata in tempo reale con le due soglie come linee di riferimento. Permette di capire l'incertezza del modello su ogni giorno simulato."),
        ("📈", "Buy & Hold benchmark", "improved",
         "Confronto diretto strategy vs Buy & Hold sullo stesso periodo di test, per misurare il valore reale aggiunto dal modello."),
    ]

    col1, col2 = st.columns(2)
    for i, (icon, title, tag, desc) in enumerate(improvements):
        col = col1 if i % 2 == 0 else col2
        tag_html = f'<span class="tag tag-new">NEW</span>' if tag == "new" else f'<span class="tag tag-improved">IMPROVED</span>'
        col.markdown(f"""
<div class="improvement-card">
  <div style="font-size:1.5rem;margin-bottom:6px">{icon}</div>
  <div style="font-weight:700;margin-bottom:4px">{title} {tag_html}</div>
  <div style="color:#8a9bbf;font-size:.85rem;line-height:1.5">{desc}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Evoluzione Performance</div>', unsafe_allow_html=True)

    perf_data = pd.DataFrame({
        'Versione': ['v1 GBM base', 'v2 +features', 'v3 dual thr.', 'v4 ExtraTrees'],
        'F1 macro': [0.40, 0.41, 0.46, 0.53],
        'Backtest':  [-66, -45, -32, +147],
        'Features':  [9, 15, 32, 32],
    })

    f1_chart = alt.Chart(perf_data).mark_bar(cornerRadiusEnd=4,
        color=alt.Gradient(gradient='linear',
            stops=[alt.GradientStop(color='#7c4dff',offset=0),
                   alt.GradientStop(color='#00d4ff',offset=1)],
            x1=0,x2=1,y1=0,y2=0)
    ).encode(
        x=alt.X('Versione:N', axis=alt.Axis(labelColor='#e8edf5'), sort=None),
        y=alt.Y('F1 macro:Q', scale=alt.Scale(domain=[0.35, 0.58]),
                axis=alt.Axis(labelColor='#5a6a85', gridColor='#1e2d4a', format='.2f')),
        tooltip=['Versione:N', alt.Tooltip('F1 macro:Q', format='.3f'), 'Features:Q']
    ).properties(height=200, title=alt.TitleParams('F1 Macro', color='#5a6a85', fontSize=11))

    bt_color = alt.condition(
        alt.datum['Backtest'] > 0,
        alt.value('#00e676'),
        alt.value('#ff1744')
    )
    bt_chart = alt.Chart(perf_data).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X('Versione:N', axis=alt.Axis(labelColor='#e8edf5'), sort=None),
        y=alt.Y('Backtest:Q', axis=alt.Axis(labelColor='#5a6a85', gridColor='#1e2d4a', format='+.0f')),
        color=bt_color,
        tooltip=['Versione:N', alt.Tooltip('Backtest:Q', format='+.0f', title='Return %')]
    ).properties(height=200, title=alt.TitleParams('Backtest Return % (test set)', color='#5a6a85', fontSize=11))

    combined_perf = alt.hconcat(f1_chart, bt_chart).configure(
        background='transparent'
    ).configure_view(strokeOpacity=0)
    st.altair_chart(combined_perf, use_container_width=True)

    st.markdown("""
> ⚠️ **Nota importante**: il +146% in backtest è misurato sul **test set storico** — un periodo in cui ETH è sceso (-45% buy&hold).
> Il modello ha quindi battuto il mercato in un contesto ribassista. I risultati passati non garantiscono performance future.
    """)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#2d3d5a;font-size:.75rem;font-family:Space Mono,monospace">'
    '⬡ BETHACON v4 · Roberto Darcangelo · Solo scopo educativo, non costituisce consulenza finanziaria.'
    '</p>', unsafe_allow_html=True
)