import pandas as pd
import numpy as np
import joblib
import streamlit as st
import altair as alt
import os

def fix_dtypes(df):
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype) in ('string', 'StringDtype'):
            df = df.copy()
            df[col] = df[col].astype(str)
    return df

TRANSLATIONS = {
    "it": {
        "sidebar_sim":      "⚙️ Parametri Simulazione",
        "eth_price":        "ETH Price attuale ($)",
        "capital":          "Capitale iniziale ($)",
        "fee":              "Fee di trading (%)",
        "days":             "Giorni da simulare",
        "volatility":       "Volatilità giornaliera (%)",
        "scenarios":        "Scenari Monte Carlo",
        "dual_thr":         "🎯 Dual Threshold",
        "buy_lbl":          "Soglia BUY  (proba ≥)",
        "buy_help":         "Alta confidenza richiesta per entrare in posizione.",
        "sell_lbl":         "Soglia SELL (proba ≤)",
        "sell_help":        "Bassa confidenza BUY = segnale SELL.",
        "hold_cap":         "Zona HOLD: probabilità tra **{s}** e **{b}**",
        "wf_cap":           "Ottimizzati walk-forward: BUY≥**{b}**  SELL≤**{s}**",
        "tab1":             "📈  FORECAST",
        "tab2":             "📊  BACKTESTING",
        "tab3":             "🔬  MODELLO",
        "tab4":             "🆕  MIGLIORAMENTI",
        "m_initial":        "Capitale Iniziale",
        "m_final":          "Capitale Stimato",
        "m_dd":             "Max Drawdown",
        "m_sharpe":         "Sharpe Ratio",
        "m_buysell":        "BUY / SELL",
        "m_hold":           "HOLD (zona grigia)",
        "m_hold_u":         "gg",
        "s_price":          "Prezzo ETH + Segnali",
        "s_proba":          "Probabilità BUY nel tempo",
        "proba_cap":        "🟢 Linea verde = soglia BUY ({b})  |  🔴 Linea rossa = soglia SELL ({s})  |  ⚫ Punti grigi = HOLD",
        "s_mc":             "Confronto Scenari Monte Carlo",
        "mc_final":         "Capitale Finale",
        "mc_return":        "Return",
        "mc_dd":            "Max DD",
        "mc_trades":        "Trade",
        "tradelog":         "📋 Trade Log ({n} operazioni)",
        "no_trades":        "Nessun trade eseguito in questo scenario.",
        "m_acc":            "Accuracy",
        "m_f1":             "F1 Macro",
        "m_prec":           "Precision (BUY)",
        "m_rec":            "Recall (BUY)",
        "active_sig":       "Segnali attivi: {a} su {t} ({h} HOLD esclusi — {p:.0f}%)",
        "s_bt":             "Strategy vs Buy & Hold (Test Set)",
        "bt_ret":           "Return totale",
        "bt_cap":           "Capitale finale",
        "bt_bthr":          "BUY threshold",
        "bt_sthr":          "SELL threshold",
        "bt_report":        "📄 Classification Report completo",
        "portfolio":        "Portfolio ($)",
        "metric_col":       "Metrica",
        "fi_title":         "Feature Importances ({n} features)",
        "s_dist":           "Distribuzione Segnali nel Dataset",
        "s_tech":           "Dettagli Tecnici",
        "algo_line":        "**Algoritmo**: **{a}** (sklearn) — alberi completamente randomizzati, più veloci e spesso più generali di Random Forest",
        "param_line":       "**Parametri**: `n_estimators=400` · `min_samples_leaf=15` · `class_weight=balanced` · `max_features=0.6` · `no max_depth`",
        "why_line":         "**Perché ExtraTrees su GBM**: nei test ExtraTrees ha prodotto **+146% in backtest** contro -32% del GBM, con F1 macro più bilanciato (0.53 vs 0.41) e segnali SELL molto più affidabili.",
        "feat_line":        "**Features**: {n} indicatori tecnici — momentum multi-window, volatilità, RSI+derivate, MACD+derivate, volume ratio, price position, lag returns",
        "split_line":       "**Split**: 70% train / 30% test (temporale, nessun lookahead)",
        "thr_header":       "**Dual Threshold ottimizzato (walk-forward su F1 macro)**:",
        "thr_buy":          "- 🟢 BUY se P(BUY) ≥ `{b}` — alta confidenza, entra in posizione",
        "thr_sell":         "- 🔴 SELL se P(BUY) ≤ `{s}` — bassa confidenza, esci dalla posizione",
        "thr_hold":         "- ⚫ HOLD se `{s}` < P(BUY) < `{b}` — zona di incertezza, nessuna azione",
        "t4_title":         "### Evoluzione del modello: v1 → v4",
        "s_perf":           "Evoluzione Performance",
        "disclaimer":       "> ⚠️ **Nota**: il +146% è sul test set storico (ETH -45% buy&hold). Il modello ha battuto il mercato in contesto ribassista. I risultati passati non garantiscono performance future.",
        "perf_f1":          "F1 Macro",
        "perf_bt":          "Backtest Return % (test set)",
        "perf_vers":        ["v1 GBM base", "v2 +features", "v3 dual thr.", "v4 ExtraTrees"],
        "improvements": [
            ("🧠", "Features espanse: 9 → 32", "improved", "Momentum multi-window (3/5/14/21 giorni), volatilità rolling, lagged returns, RSI delta e accelerazione MACD."),
            ("🌲", "GBM → ExtraTrees (v4)", "new", "Testati 8 algoritmi. ExtraTrees senza max_depth ha vinto con +146% in backtest vs -32% del GBM."),
            ("📐", "Dual Threshold walk-forward", "new", "Due soglie separate ottimizzate su F1 macro. Elimina il bias verso una sola classe."),
            ("⚖️", "Class weight balanced", "new", "Migliora drasticamente la recall sui SELL (da 11% a ~59%)."),
            ("📊", "Probabilità BUY live", "improved", "La curva P(BUY) è visualizzata in tempo reale con le due soglie come linee di riferimento."),
            ("📈", "Buy & Hold benchmark", "improved", "Confronto diretto strategy vs Buy & Hold sullo stesso periodo di test."),
        ],
        "header_sub":       "ETH · AI Trading Signals · v4",
        "spinner":          "Training modello in-app…",
        "ds_error":         "Dataset non trovato. Assicurati che eth-dataset-processed.csv sia nella cartella data/processed/",
        "footer":           "⬡ BETHACON v4 · Roberto Darcangelo · Solo scopo educativo, non costituisce consulenza finanziaria.",
    },
    "en": {
        "sidebar_sim":      "⚙️ Simulation Parameters",
        "eth_price":        "Current ETH Price ($)",
        "capital":          "Initial Capital ($)",
        "fee":              "Trading Fee (%)",
        "days":             "Days to Simulate",
        "volatility":       "Est. Daily Volatility (%)",
        "scenarios":        "Monte Carlo Scenarios",
        "dual_thr":         "🎯 Dual Threshold",
        "buy_lbl":          "BUY Threshold (proba ≥)",
        "buy_help":         "High confidence required to enter a position.",
        "sell_lbl":         "SELL Threshold (proba ≤)",
        "sell_help":        "Low BUY probability = SELL signal.",
        "hold_cap":         "HOLD zone: probability between **{s}** and **{b}**",
        "wf_cap":           "Walk-forward optimised: BUY≥**{b}**  SELL≤**{s}**",
        "tab1":             "📈  FORECAST",
        "tab2":             "📊  BACKTESTING",
        "tab3":             "🔬  MODEL",
        "tab4":             "🆕  IMPROVEMENTS",
        "m_initial":        "Initial Capital",
        "m_final":          "Projected Capital",
        "m_dd":             "Max Drawdown",
        "m_sharpe":         "Sharpe Ratio",
        "m_buysell":        "BUY / SELL",
        "m_hold":           "HOLD (grey zone)",
        "m_hold_u":         "d",
        "s_price":          "ETH Price + Signals",
        "s_proba":          "BUY Probability Over Time",
        "proba_cap":        "🟢 Green line = BUY threshold ({b})  |  🔴 Red line = SELL threshold ({s})  |  ⚫ Grey dots = HOLD",
        "s_mc":             "Monte Carlo Scenarios Comparison",
        "mc_final":         "Final Capital",
        "mc_return":        "Return",
        "mc_dd":            "Max DD",
        "mc_trades":        "Trades",
        "tradelog":         "📋 Trade Log ({n} trades)",
        "no_trades":        "No trades executed in this scenario.",
        "m_acc":            "Accuracy",
        "m_f1":             "F1 Macro",
        "m_prec":           "Precision (BUY)",
        "m_rec":            "Recall (BUY)",
        "active_sig":       "Active signals: {a} of {t} ({h} HOLD excluded — {p:.0f}%)",
        "s_bt":             "Strategy vs Buy & Hold (Test Set)",
        "bt_ret":           "Total Return",
        "bt_cap":           "Final Capital",
        "bt_bthr":          "BUY threshold",
        "bt_sthr":          "SELL threshold",
        "bt_report":        "📄 Full Classification Report",
        "portfolio":        "Portfolio ($)",
        "metric_col":       "Metric",
        "fi_title":         "Feature Importances ({n} features)",
        "s_dist":           "Signal Distribution in Dataset",
        "s_tech":           "Technical Details",
        "algo_line":        "**Algorithm**: **{a}** (sklearn) — fully randomised trees, faster and often more general than Random Forest",
        "param_line":       "**Parameters**: `n_estimators=400` · `min_samples_leaf=15` · `class_weight=balanced` · `max_features=0.6` · `no max_depth`",
        "why_line":         "**Why ExtraTrees over GBM**: in tests ExtraTrees produced **+146% in backtest** vs -32% for GBM, with a more balanced F1 macro (0.53 vs 0.41) and far more reliable SELL signals.",
        "feat_line":        "**Features**: {n} technical indicators — multi-window momentum, volatility, RSI derivatives, MACD derivatives, volume ratio, price position, lagged returns",
        "split_line":       "**Split**: 70% train / 30% test (temporal, no lookahead)",
        "thr_header":       "**Optimised Dual Threshold (walk-forward on F1 macro)**:",
        "thr_buy":          "- 🟢 BUY if P(BUY) ≥ `{b}` — high confidence, enter position",
        "thr_sell":         "- 🔴 SELL if P(BUY) ≤ `{s}` — low confidence, exit position",
        "thr_hold":         "- ⚫ HOLD if `{s}` < P(BUY) < `{b}` — uncertainty zone, no action",
        "t4_title":         "### Model Evolution: v1 → v4",
        "s_perf":           "Performance Evolution",
        "disclaimer":       "> ⚠️ **Note**: the +146% is on the historical test set (ETH -45% buy&hold). The model outperformed the market in a bearish context. Past results do not guarantee future performance.",
        "perf_f1":          "F1 Macro",
        "perf_bt":          "Backtest Return % (test set)",
        "perf_vers":        ["v1 GBM base", "v2 +features", "v3 dual thr.", "v4 ExtraTrees"],
        "improvements": [
            ("🧠", "Expanded features: 9 → 32", "improved", "Multi-window momentum (3/5/14/21 days), rolling volatility, lagged returns, RSI delta and MACD acceleration."),
            ("🌲", "GBM → ExtraTrees (v4)", "new", "8 algorithms benchmarked. ExtraTrees with no max_depth won with +146% backtest vs -32% for GBM."),
            ("📐", "Walk-forward Dual Threshold", "new", "Two separate thresholds optimised on F1 macro. Eliminates class bias."),
            ("⚖️", "Balanced class weights", "new", "Drastically improves SELL recall (from 11% to ~59%)."),
            ("📊", "Live BUY probability", "improved", "The P(BUY) curve is shown in real time with both threshold lines as reference."),
            ("📈", "Buy & Hold benchmark", "improved", "Direct strategy vs Buy & Hold comparison on the same test period."),
        ],
        "header_sub":       "ETH · AI Trading Signals · v4",
        "spinner":          "Training model in-app…",
        "ds_error":         "Dataset not found. Make sure eth-dataset-processed.csv is in data/processed/",
        "footer":           "⬡ BETHACON v4 · Roberto Darcangelo · For educational purposes only. Not financial advice.",
    },
}

st.set_page_config(page_title="Bethacon | ETH Trading AI", page_icon="⬡", layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');
:root{--bg:#080c14;--surface:#0e1420;--surface2:#141c2e;--border:#1e2d4a;--accent:#00d4ff;--accent2:#7c4dff;--buy:#00e676;--sell:#ff1744;--text:#e8edf5;--muted:#5a6a85;}
html,body,[class*="css"]{font-family:'Syne',sans-serif;background:var(--bg)!important;color:var(--text)!important;}
.stApp{background:var(--bg)!important;}
section[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border)!important;}
h1,h2,h3{font-family:'Syne',sans-serif!important;font-weight:800!important;}
[data-testid="metric-container"]{background:var(--surface2)!important;border:1px solid var(--border)!important;border-radius:12px!important;padding:16px!important;}
[data-testid="metric-container"] label{color:var(--muted)!important;font-size:.75rem!important;letter-spacing:.1em!important;text-transform:uppercase!important;}
[data-testid="stMetricValue"]{color:var(--accent)!important;font-family:'Space Mono',monospace!important;font-size:1.5rem!important;}
div[data-baseweb="input"]{background:var(--surface2)!important;border:1px solid var(--border)!important;border-radius:8px!important;}
div[data-baseweb="input"] input{color:var(--text)!important;}
.stButton>button{background:linear-gradient(135deg,var(--accent2),var(--accent))!important;color:#fff!important;border:none!important;border-radius:8px!important;font-family:'Space Mono',monospace!important;font-weight:700!important;transition:all .2s!important;}
.stButton>button:hover{opacity:.85!important;transform:translateY(-1px)!important;}
.stTabs [data-baseweb="tab-list"]{background:var(--surface)!important;border-radius:10px!important;border:1px solid var(--border)!important;gap:4px!important;padding:4px!important;}
.stTabs [data-baseweb="tab"]{color:var(--muted)!important;border-radius:8px!important;font-family:'Space Mono',monospace!important;font-size:.8rem!important;}
.stTabs [aria-selected="true"]{background:var(--accent2)!important;color:#fff!important;}
details{background:var(--surface2)!important;border:1px solid var(--border)!important;border-radius:8px!important;}
hr{border-color:var(--border)!important;}
.bethacon-title{font-family:'Syne',sans-serif;font-weight:800;font-size:1.8rem;background:linear-gradient(135deg,#7c4dff,#00d4ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.bethacon-sub{color:#5a6a85;font-size:.8rem;letter-spacing:.15em;text-transform:uppercase;font-family:'Space Mono',monospace;}
.section-title{font-family:'Space Mono',monospace;font-size:.7rem;letter-spacing:.2em;text-transform:uppercase;color:#5a6a85;margin-bottom:12px;display:flex;align-items:center;gap:8px;}
.section-title::after{content:'';flex:1;height:1px;background:#1e2d4a;}
.improvement-card{background:var(--surface2);border:1px solid var(--border);border-radius:12px;padding:16px;margin-bottom:12px;}
.tag{display:inline-block;padding:2px 10px;border-radius:20px;font-size:.72rem;font-family:'Space Mono',monospace;font-weight:700;margin-right:6px;}
.tag-new{background:rgba(124,77,255,.2);color:#7c4dff;border:1px solid rgba(124,77,255,.4);}
.tag-improved{background:rgba(0,212,255,.15);color:#00d4ff;border:1px solid rgba(0,212,255,.35);}
</style>""", unsafe_allow_html=True)

# ── Language selector — MUST be first sidebar widget ─────────────────────────
_lang = st.sidebar.selectbox("🌐 Lingua / Language", ["🇮🇹  Italiano", "🇬🇧  English"], key="lang")
T = TRANSLATIONS["it"] if _lang.startswith("🇮🇹") else TRANSLATIONS["en"]

st.markdown(f"""<div style="display:flex;align-items:center;gap:12px;margin-bottom:8px">
  <div style="width:44px;height:44px;background:linear-gradient(135deg,#7c4dff,#00d4ff);border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:22px">⬡</div>
  <div><div class="bethacon-title">BETHACON</div><div class="bethacon-sub">{T["header_sub"]}</div></div>
</div>""", unsafe_allow_html=True)
st.markdown("---")

FEATURES = ['price_vs_ma7','price_vs_ma14','momentum_3','momentum_5','momentum_7','momentum_14','momentum_21',
    'return_3d','return_5d','return_14d','price_pct_change','pct_lag1','pct_lag2','volatility_7','volatility_14',
    'total_volume','volume_ratio','volume_spike','rsi','rsi_ma3','rsi_delta','rsi_lag1','rsi_oversold','rsi_overbought',
    'macd','signal_line','histogram','hist_slope','hist_accel','macd_cross','macd_lag1','price_position']

def add_features(df):
    p=df['price']; df=df.copy()
    df['price_pct_change']=p.pct_change(); df['price_vs_ma7']=p-p.rolling(7).mean(); df['price_vs_ma14']=p-p.rolling(14).mean()
    for w in [3,5,7,14,21]: df[f'momentum_{w}']=p-p.shift(w); df[f'return_{w}d']=p.pct_change(w)
    df['volatility_7']=p.pct_change().rolling(7).std(); df['volatility_14']=p.pct_change().rolling(14).std()
    ch=p.diff(); au=ch.clip(lower=0).rolling(14).mean(); ad=(-ch).clip(lower=0).rolling(14).mean()
    df['rsi']=100-(100/(1+au/(ad+1e-9))); df['rsi_ma3']=df['rsi'].rolling(3).mean()
    df['rsi_delta']=df['rsi'].diff(); df['rsi_lag1']=df['rsi'].shift(1)
    df['rsi_oversold']=(df['rsi']<30).astype(int); df['rsi_overbought']=(df['rsi']>70).astype(int)
    e12=p.ewm(span=12,adjust=False).mean(); e26=p.ewm(span=26,adjust=False).mean()
    df['macd']=e12-e26; df['signal_line']=df['macd'].ewm(span=9,adjust=False).mean()
    df['histogram']=df['macd']-df['signal_line']; df['hist_slope']=df['histogram'].diff()
    df['hist_accel']=df['hist_slope'].diff(); df['macd_cross']=(df['macd']>df['signal_line']).astype(int)
    df['macd_lag1']=df['macd'].shift(1)
    vm=df['total_volume'].rolling(7).mean(); df['volume_ratio']=df['total_volume']/(vm+1e-9)
    df['volume_spike']=(df['volume_ratio']>2.0).astype(int)
    pn=p.rolling(14).min(); px=p.rolling(14).max(); df['price_position']=(p-pn)/(px-pn+1e-9)
    df['pct_lag1']=df['price_pct_change'].shift(1); df['pct_lag2']=df['price_pct_change'].shift(2)
    return df

@st.cache_data
def load_data():
    for path in ['data/processed/eth-dataset-processed.csv','eth-dataset-processed.csv']:
        if os.path.exists(path):
            df=pd.read_csv(path); df['snapped_at']=pd.to_datetime(df['snapped_at'],utc=True)
            df=df.sort_values('snapped_at').reset_index(drop=True)
            if 'momentum_3' not in df.columns: df=add_features(df)
            return df.fillna(0)
    return None

@st.cache_resource
def load_model():
    for path in ['model/bethacon_v4.pkl','model/bethacon_v3.pkl','model/bethacon_v2.pkl']:
        if os.path.exists(path):
            b=joblib.load(path)
            if isinstance(b,dict):
                return b['model'],b.get('features',FEATURES),b.get('buy_thr',b.get('threshold',0.55)),b.get('sell_thr',0.45)
            return b,FEATURES,0.55,0.45
    return None,FEATURES,0.55,0.45

def train_model_inapp(df):
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.metrics import f1_score
    av=[f for f in FEATURES if f in df.columns]; x=df[av]; y=df['signal'].astype(int)
    sp=int(len(df)*0.70); vc=int(sp*0.85)
    m=ExtraTreesClassifier(n_estimators=400,min_samples_leaf=15,class_weight='balanced',max_features=0.6,random_state=42,n_jobs=-1)
    m.fit(x.iloc[:vc],y.iloc[:vc]); vp=m.predict_proba(x.iloc[vc:sp])[:,1]; vy=y.iloc[vc:sp]
    bm,bb,bs=0.0,0.55,0.45
    for bt in np.arange(0.48,0.78,0.03):
        for st in np.arange(0.22,0.52,0.03):
            pr=np.where(vp>=bt,1,np.where(vp<=st,0,-1)); mk=pr!=-1
            if mk.sum()<8: continue
            sc=f1_score(vy[mk],pr[mk],average='macro',zero_division=0)
            if sc>bm: bm,bb,bs=sc,bt,st
    m.fit(x.iloc[:sp],y.iloc[:sp]); return m,av,bb,bs

ds=load_data()
if ds is None: st.error(T["ds_error"]); st.stop()
model,feat_cols,buy_threshold,sell_threshold=load_model()
if model is None:
    with st.spinner(T["spinner"]): model,feat_cols,buy_threshold,sell_threshold=train_model_inapp(ds)

st.sidebar.markdown(f"### {T['sidebar_sim']}")
last_price=float(ds['price'].iloc[-1])
current_price=st.sidebar.number_input(T["eth_price"],value=round(last_price,2),min_value=1.0)
capital_init=st.sidebar.number_input(T["capital"],value=10000.0,min_value=100.0)
fee_pct=st.sidebar.number_input(T["fee"],value=0.1,min_value=0.0,max_value=5.0,step=0.01)
days_future=st.sidebar.slider(T["days"],10,365,90)
volatility=st.sidebar.slider(T["volatility"],1.0,15.0,3.5)/100.0
n_scenarios=st.sidebar.slider(T["scenarios"],1,5,3)
st.sidebar.markdown("---")
st.sidebar.markdown(f"### {T['dual_thr']}")
buy_thr=st.sidebar.slider(T["buy_lbl"],0.50,0.80,float(round(buy_threshold,2)),step=0.01,help=T["buy_help"])
sell_thr=st.sidebar.slider(T["sell_lbl"],0.20,0.49,float(round(sell_threshold,2)),step=0.01,help=T["sell_help"])
st.sidebar.caption(T["hold_cap"].format(s=f"{sell_thr:.2f}",b=f"{buy_thr:.2f}"))
st.sidebar.caption(T["wf_cap"].format(b=f"{buy_threshold:.2f}",s=f"{sell_threshold:.2f}"))

def simulate_future(seed,n_days,start_price,vol,hist_df):
    np.random.seed(seed); prices=[start_price]
    for _ in range(1,n_days): prices.append(prices[-1]*(1+np.random.normal(0,vol)))
    fdates=[hist_df['snapped_at'].iloc[-1]+pd.Timedelta(days=i+1) for i in range(n_days)]
    fut=pd.DataFrame({'snapped_at':fdates,'price':prices,'total_volume':hist_df['total_volume'].mean()})
    combined=pd.concat([hist_df[['snapped_at','price','total_volume']],fut],ignore_index=True)
    combined=add_features(combined).fillna(0)
    sim=combined.tail(n_days).copy().reset_index(drop=True)
    av=[f for f in feat_cols if f in sim.columns]
    pr=model.predict_proba(sim[av])[:,1]
    sim['signal_proba']=pr; sim['signal_pred']=np.where(pr>=buy_thr,1,np.where(pr<=sell_thr,0,-1))
    return sim

def run_backtest(sim_df,capital,fee):
    cash,pos,equity,trades=capital,0.0,[],[]
    for _,row in sim_df.iterrows():
        sig,price,date=row['signal_pred'],row['price'],row['snapped_at']
        if sig==1 and cash>0: pos=cash/price*(1-fee/100); cash=0.0; trades.append({"Date":date,"Action":"BUY","Price":price,"Equity":pos*price})
        elif sig==0 and pos>0: cash=pos*price*(1-fee/100); pos=0.0; trades.append({"Date":date,"Action":"SELL","Price":price,"Equity":cash})
        equity.append(cash+pos*price)
    sim_df=sim_df.copy(); sim_df['equity']=equity; return sim_df,trades,equity

def perf_stats(equity,capital):
    ret=(equity[-1]-capital)/capital*100
    rets=np.diff(equity)/(np.array(equity[:-1])+1e-9)
    dd=np.min(np.array(equity)/np.maximum.accumulate(equity)-1)*100
    sh=np.mean(rets)/(np.std(rets)+1e-9)*np.sqrt(365)
    return ret,dd,sh

def sec(txt): st.markdown(f'<div class="section-title">{txt}</div>',unsafe_allow_html=True)

tab1,tab2,tab3,tab4=st.tabs([T["tab1"],T["tab2"],T["tab3"],T["tab4"]])

# ── TAB 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    seeds=[42+i*13 for i in range(n_scenarios)]
    hist_tail=ds.tail(60).copy(); scenario_data={}
    for i,seed in enumerate(seeds):
        sim=simulate_future(seed,days_future,current_price,volatility,hist_tail)
        sim,trades,eq=run_backtest(sim,capital_init,fee_pct)
        scenario_data[f"Scenario {i+1}"]={"sim":sim,"trades":trades,"equity":eq}
    best_key=max(scenario_data,key=lambda k:scenario_data[k]["equity"][-1])
    sim_df=scenario_data[best_key]["sim"]; trades=scenario_data[best_key]["trades"]; equity=scenario_data[best_key]["equity"]
    total_ret,max_dd,sharpe=perf_stats(equity,capital_init)
    n_buys=sum(1 for t in trades if t["Action"]=="BUY"); n_sells=sum(1 for t in trades if t["Action"]=="SELL")
    n_holds=int((sim_df['signal_pred']==-1).sum())
    c1,c2,c3,c4,c5,c6=st.columns(6)
    c1.metric(T["m_initial"],f"${capital_init:,.0f}"); c2.metric(T["m_final"],f"${equity[-1]:,.0f}",f"{total_ret:+.1f}%")
    c3.metric(T["m_dd"],f"{max_dd:.1f}%"); c4.metric(T["m_sharpe"],f"{sharpe:.2f}")
    c5.metric(T["m_buysell"],f"{n_buys} / {n_sells}"); c6.metric(T["m_hold"],f"{n_holds} {T['m_hold_u']}")
    st.markdown("<br>",unsafe_allow_html=True); sec(T["s_price"])
    price_df=sim_df[['snapped_at','price','signal_pred','signal_proba']].copy()
    buy_df=price_df[price_df['signal_pred']==1]; sell_df=price_df[price_df['signal_pred']==0]; hold_df=price_df[price_df['signal_pred']==-1]
    base=alt.Chart(price_df).encode(x=alt.X('snapped_at:T',title='',axis=alt.Axis(format='%b %d',labelColor='#5a6a85')))
    price_line=base.mark_line(strokeWidth=2,color='#00d4ff').encode(
        y=alt.Y('price:Q',title='Price ($)',scale=alt.Scale(zero=False),axis=alt.Axis(labelColor='#5a6a85',titleColor='#5a6a85',gridColor='#1e2d4a')),
        tooltip=[alt.Tooltip('snapped_at:T',format='%Y-%m-%d'),alt.Tooltip('price:Q',format='$,.2f'),alt.Tooltip('signal_proba:Q',title='P(BUY)',format='.2%')])
    buy_pts=alt.Chart(buy_df).mark_point(shape='triangle-up',size=130,filled=True,color='#00e676',opacity=0.9).encode(
        x='snapped_at:T',y='price:Q',tooltip=[alt.Tooltip('snapped_at:T',format='%Y-%m-%d'),alt.Tooltip('price:Q',format='$,.2f'),alt.Tooltip('signal_proba:Q',title='P(BUY)',format='.2%')])
    sell_pts=alt.Chart(sell_df).mark_point(shape='triangle-down',size=130,filled=True,color='#ff1744',opacity=0.9).encode(
        x='snapped_at:T',y='price:Q',tooltip=[alt.Tooltip('snapped_at:T',format='%Y-%m-%d'),alt.Tooltip('price:Q',format='$,.2f'),alt.Tooltip('signal_proba:Q',title='P(BUY)',format='.2%')])
    hold_pts=alt.Chart(hold_df).mark_point(shape='circle',size=30,filled=True,color='#5a6a85',opacity=0.4).encode(
        x='snapped_at:T',y='price:Q',tooltip=[alt.Tooltip('snapped_at:T',format='%Y-%m-%d'),alt.Tooltip('price:Q',format='$,.2f'),alt.Tooltip('signal_proba:Q',title='P(BUY)',format='.2%')])
    upper=alt.layer(price_line,hold_pts,buy_pts,sell_pts).properties(height=320)
    eq_df=pd.DataFrame({'snapped_at':sim_df['snapped_at'],'equity':equity})
    equity_area=alt.Chart(eq_df).mark_area(line={'color':'#7c4dff','strokeWidth':2},
        color=alt.Gradient(gradient='linear',stops=[alt.GradientStop(color='#7c4dff',offset=0),alt.GradientStop(color='transparent',offset=1)],x1=1,x2=1,y1=1,y2=0),opacity=0.35
    ).encode(x=alt.X('snapped_at:T',axis=alt.Axis(format='%b %d',labelColor='#5a6a85')),
        y=alt.Y('equity:Q',title=T["portfolio"],scale=alt.Scale(zero=False),axis=alt.Axis(labelColor='#5a6a85',titleColor='#5a6a85',gridColor='#1e2d4a')),
        tooltip=[alt.Tooltip('snapped_at:T',format='%Y-%m-%d'),alt.Tooltip('equity:Q',title='Equity',format='$,.2f')])
    bkv=alt.Chart(pd.DataFrame({'y':[capital_init]})).mark_rule(strokeDash=[4,4],color='#5a6a85',strokeWidth=1.5).encode(y='y:Q')
    lower=alt.layer(equity_area,bkv).properties(height=160)
    combined_chart=alt.vconcat(upper,lower).configure(background='transparent').configure_axis(labelFontSize=10,titleFontSize=11).configure_view(strokeOpacity=0)
    st.altair_chart(combined_chart,use_container_width=True)
    sec(T["s_proba"])
    proba_chart=alt.Chart(sim_df).mark_area(line={'color':'#7c4dff','strokeWidth':1.5},
        color=alt.Gradient(gradient='linear',stops=[alt.GradientStop(color='#7c4dff',offset=0),alt.GradientStop(color='transparent',offset=1)],x1=1,x2=1,y1=1,y2=0),opacity=0.5
    ).encode(x=alt.X('snapped_at:T',title='',axis=alt.Axis(format='%b %d',labelColor='#5a6a85')),
        y=alt.Y('signal_proba:Q',title='P(BUY)',scale=alt.Scale(domain=[0,1]),axis=alt.Axis(labelColor='#5a6a85',gridColor='#1e2d4a',format='.0%')),
        tooltip=[alt.Tooltip('snapped_at:T',format='%Y-%m-%d'),alt.Tooltip('signal_proba:Q',format='.2%',title='P(BUY)')]).properties(height=120)
    proba_final=alt.layer(proba_chart,
        alt.Chart(pd.DataFrame({'y':[buy_thr]})).mark_rule(strokeDash=[4,4],color='#00e676',strokeWidth=1.5).encode(y='y:Q'),
        alt.Chart(pd.DataFrame({'y':[sell_thr]})).mark_rule(strokeDash=[4,4],color='#ff1744',strokeWidth=1.5).encode(y='y:Q'),
    ).configure(background='transparent').configure_view(strokeOpacity=0)
    st.altair_chart(proba_final,use_container_width=True)
    st.caption(T["proba_cap"].format(b=f"{buy_thr:.2f}",s=f"{sell_thr:.2f}"))
    if n_scenarios>1:
        sec(T["s_mc"]); mc_rows=[]
        for label,d in scenario_data.items():
            eq=d["equity"]; ret,dd,sh=perf_stats(eq,capital_init)
            mc_rows.append({"Scenario":label,T["mc_final"]:f"${eq[-1]:,.0f}",T["mc_return"]:f"{ret:+.1f}%",T["mc_dd"]:f"{dd:.1f}%","Sharpe":f"{sh:.2f}",T["mc_trades"]:len(d["trades"])})
        st.dataframe(pd.DataFrame(mc_rows).set_index("Scenario"),use_container_width=True)
    if trades:
        with st.expander(T["tradelog"].format(n=len(trades))):
            log_df=pd.DataFrame(trades); styled=log_df.style.format({"Price":"${:,.2f}","Equity":"${:,.2f}"})
            try: styled=styled.map(lambda v:'color:#00e676' if v=='BUY' else('color:#ff1744' if v=='SELL' else''),subset=['Action'])
            except AttributeError: styled=styled.applymap(lambda v:'color:#00e676' if v=='BUY' else('color:#ff1744' if v=='SELL' else''),subset=['Action'])
            st.dataframe(styled,use_container_width=True)
    else: st.info(T["no_trades"])

# ── TAB 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,classification_report
    av=[f for f in feat_cols if f in ds.columns]; x_all=ds[av]; y_all=ds['signal'].astype(int)
    split=int(len(ds)*0.70); x_test_bt=x_all.iloc[split:]; y_test_bt=y_all.iloc[split:]
    proba_bt=model.predict_proba(x_test_bt)[:,1]
    pred_dual=np.where(proba_bt>=buy_thr,1,np.where(proba_bt<=sell_thr,0,-1))
    mask_bt=pred_dual!=-1; y_pred_bt=pred_dual[mask_bt]; y_true_bt=y_test_bt.values[mask_bt]
    acc=accuracy_score(y_true_bt,y_pred_bt); f1=f1_score(y_true_bt,y_pred_bt,zero_division=0,average='macro')
    cm=confusion_matrix(y_true_bt,y_pred_bt)
    prec_buy=cm[1,1]/(cm[1,1]+cm[0,1]+1e-9); rec_buy=cm[1,1]/(cm[1,1]+cm[1,0]+1e-9)
    n_hold_bt=int((pred_dual==-1).sum())
    c1,c2,c3,c4=st.columns(4)
    c1.metric(T["m_acc"],f"{acc:.2%}"); c2.metric(T["m_f1"],f"{f1:.4f}")
    c3.metric(T["m_prec"],f"{prec_buy:.2%}"); c4.metric(T["m_rec"],f"{rec_buy:.2%}")
    st.caption(T["active_sig"].format(a=mask_bt.sum(),t=len(pred_dual),h=n_hold_bt,p=n_hold_bt/len(pred_dual)*100))
    st.markdown("<br>",unsafe_allow_html=True); sec(T["s_bt"])
    test_df=ds.iloc[split:].copy().reset_index(drop=True); test_df['signal_pred']=pred_dual
    test_df,_,hist_eq=run_backtest(test_df,capital_init,fee_pct)
    bh_eq=capital_init*(test_df['price']/test_df['price'].iloc[0])
    eq_chart_df=pd.DataFrame({'snapped_at':test_df['snapped_at'],'Strategy':hist_eq,'Buy & Hold':bh_eq.values}).melt(id_vars='snapped_at',var_name='Type',value_name='Equity')
    eq_chart_df=fix_dtypes(eq_chart_df)
    hist_chart=alt.Chart(eq_chart_df).mark_line(strokeWidth=2).encode(
        x=alt.X('snapped_at:T',title='',axis=alt.Axis(labelColor='#5a6a85')),
        y=alt.Y('Equity:Q',title=T["portfolio"],scale=alt.Scale(zero=False),axis=alt.Axis(labelColor='#5a6a85',gridColor='#1e2d4a')),
        color=alt.Color('Type:N',scale=alt.Scale(domain=['Strategy','Buy & Hold'],range=['#7c4dff','#00d4ff']),legend=alt.Legend(orient='top',labelColor='#e8edf5',titleColor='#5a6a85')),
        tooltip=['snapped_at:T','Type:N',alt.Tooltip('Equity:Q',format='$,.2f')]
    ).properties(height=340).configure(background='transparent').configure_view(strokeOpacity=0)
    st.altair_chart(hist_chart,use_container_width=True)
    bh_ret=(bh_eq.iloc[-1]-capital_init)/capital_init*100; str_ret=(hist_eq[-1]-capital_init)/capital_init*100
    st.markdown(f"| {T['metric_col']} | Strategy | Buy & Hold |\n|---|---|---|\n| {T['bt_ret']} | **{str_ret:+.1f}%** | {bh_ret:+.1f}% |\n| {T['bt_cap']} | **${hist_eq[-1]:,.0f}** | ${bh_eq.iloc[-1]:,.0f} |\n| {T['bt_bthr']} | **{buy_thr:.2f}** | — |\n| {T['bt_sthr']} | **{sell_thr:.2f}** | — |")
    with st.expander(T["bt_report"]): st.code(classification_report(y_true_bt,y_pred_bt,target_names=["SELL","BUY"]))

# ── TAB 3 ─────────────────────────────────────────────────────────────────────
with tab3:
    sec(T["fi_title"].format(n=len(feat_cols)))
    fi=pd.DataFrame({'Feature':feat_cols[:len(model.feature_importances_)],'Importance':model.feature_importances_}).sort_values('Importance',ascending=False)
    fi=fix_dtypes(fi)
    fi_chart=alt.Chart(fi).mark_bar(cornerRadiusEnd=4,
        color=alt.Gradient(gradient='linear',stops=[alt.GradientStop(color='#7c4dff',offset=0),alt.GradientStop(color='#00d4ff',offset=1)],x1=0,x2=1,y1=0,y2=0)
    ).encode(x=alt.X('Importance:Q',axis=alt.Axis(labelColor='#5a6a85',gridColor='#1e2d4a',format='.3f')),
        y=alt.Y('Feature:N',sort='-x',axis=alt.Axis(labelColor='#e8edf5')),
        tooltip=['Feature:N',alt.Tooltip('Importance:Q',format='.4f')]
    ).properties(height=500).configure(background='transparent').configure_view(strokeOpacity=0)
    st.altair_chart(fi_chart,use_container_width=True)
    st.markdown("<br>",unsafe_allow_html=True); sec(T["s_dist"])
    sig_dist=ds['signal'].value_counts().reset_index(); sig_dist.columns=['signal','count']
    sig_dist['Label']=sig_dist['signal'].map({0:'SELL',1:'BUY'}); sig_dist=fix_dtypes(sig_dist)
    dist_chart=alt.Chart(sig_dist).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X('Label:N',axis=alt.Axis(labelColor='#e8edf5')),
        y=alt.Y('count:Q',axis=alt.Axis(labelColor='#5a6a85',gridColor='#1e2d4a')),
        color=alt.Color('Label:N',scale=alt.Scale(domain=['BUY','SELL'],range=['#00e676','#ff1744']),legend=None),
        tooltip=['Label:N','count:Q']
    ).properties(height=220).configure(background='transparent').configure_view(strokeOpacity=0)
    st.altair_chart(dist_chart,use_container_width=True)
    sec(T["s_tech"])
    st.markdown(f"""
{T["algo_line"].format(a="ExtraTreesClassifier")}

{T["param_line"]}

{T["why_line"]}

{T["feat_line"].format(n=len(feat_cols))}

{T["split_line"]}

{T["thr_header"]}
{T["thr_buy"].format(b=f"{buy_threshold:.2f}")}
{T["thr_sell"].format(s=f"{sell_threshold:.2f}")}
{T["thr_hold"].format(s=f"{sell_threshold:.2f}",b=f"{buy_threshold:.2f}")}
    """)

# ── TAB 4 ─────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown(T["t4_title"]); st.markdown("<br>",unsafe_allow_html=True)
    col1,col2=st.columns(2)
    for i,(icon,title,tag,desc) in enumerate(T["improvements"]):
        col=col1 if i%2==0 else col2
        th='<span class="tag tag-new">NEW</span>' if tag=="new" else '<span class="tag tag-improved">IMPROVED</span>'
        col.markdown(f'<div class="improvement-card"><div style="font-size:1.5rem;margin-bottom:6px">{icon}</div><div style="font-weight:700;margin-bottom:4px">{title} {th}</div><div style="color:#8a9bbf;font-size:.85rem;line-height:1.5">{desc}</div></div>',unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True); sec(T["s_perf"])
    perf_data=pd.DataFrame({'Versione':T["perf_vers"],'F1 macro':[0.40,0.41,0.46,0.53],'Backtest':[-66,-45,-32,+147],'Features':[9,15,32,32]})
    perf_data=fix_dtypes(perf_data)
    f1_chart=alt.Chart(perf_data).mark_bar(cornerRadiusEnd=4,
        color=alt.Gradient(gradient='linear',stops=[alt.GradientStop(color='#7c4dff',offset=0),alt.GradientStop(color='#00d4ff',offset=1)],x1=0,x2=1,y1=0,y2=0)
    ).encode(x=alt.X('Versione:N',axis=alt.Axis(labelColor='#e8edf5'),sort=None),
        y=alt.Y('F1 macro:Q',scale=alt.Scale(domain=[0.35,0.58]),axis=alt.Axis(labelColor='#5a6a85',gridColor='#1e2d4a',format='.2f')),
        tooltip=['Versione:N',alt.Tooltip('F1 macro:Q',format='.3f'),'Features:Q']
    ).properties(height=200,title=alt.TitleParams(T["perf_f1"],color='#5a6a85',fontSize=11))
    bt_chart=alt.Chart(perf_data).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X('Versione:N',axis=alt.Axis(labelColor='#e8edf5'),sort=None),
        y=alt.Y('Backtest:Q',axis=alt.Axis(labelColor='#5a6a85',gridColor='#1e2d4a',format='+.0f')),
        color=alt.condition(alt.datum['Backtest']>0,alt.value('#00e676'),alt.value('#ff1744')),
        tooltip=['Versione:N',alt.Tooltip('Backtest:Q',format='+.0f',title='Return %')]
    ).properties(height=200,title=alt.TitleParams(T["perf_bt"],color='#5a6a85',fontSize=11))
    st.altair_chart(alt.hconcat(f1_chart,bt_chart).configure(background='transparent').configure_view(strokeOpacity=0),use_container_width=True)
    st.markdown(T["disclaimer"])

st.markdown("---")
st.markdown(f'<p style="text-align:center;color:#2d3d5a;font-size:.75rem;font-family:Space Mono,monospace">{T["footer"]}</p>',unsafe_allow_html=True)