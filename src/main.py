import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import joblib
import os

# ─── Features ────────────────────────────────────────────────────────────────
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

# ─── Load data ────────────────────────────────────────────────────────────────
ds = pd.read_csv('data/processed/eth-dataset-processed.csv')
ds = ds.sort_values('snapped_at').reset_index(drop=True)
ds['signal'] = ds['signal'].astype(int)

x = ds[FEATURES]
y = ds['signal']

train_size = int(len(ds) * 0.70)
val_size   = int(train_size * 0.85)

x_train_full = x.iloc[:train_size]
y_train_full = y.iloc[:train_size]
x_test       = x.iloc[train_size:]
y_test       = y.iloc[train_size:]

# ─── Walk-Forward Dual Threshold Tuning ──────────────────────────────────────
et_temp = ExtraTreesClassifier(
    n_estimators=400, min_samples_leaf=15,
    class_weight='balanced', max_features=0.6,
    random_state=42, n_jobs=-1
)
et_temp.fit(x.iloc[:val_size], y.iloc[:val_size])
val_proba = et_temp.predict_proba(x.iloc[val_size:train_size])[:, 1]
val_y     = y.iloc[val_size:train_size]

best_macro, best_buy_thr, best_sell_thr = 0.0, 0.55, 0.45
for buy_thr in np.arange(0.48, 0.78, 0.03):
    for sell_thr in np.arange(0.22, 0.52, 0.03):
        preds = np.where(val_proba >= buy_thr, 1,
                np.where(val_proba <= sell_thr, 0, -1))
        mask = preds != -1
        if mask.sum() < 10:
            continue
        score = f1_score(val_y[mask], preds[mask], average='macro', zero_division=0)
        if score > best_macro:
            best_macro, best_buy_thr, best_sell_thr = score, buy_thr, sell_thr

print(f"Dual threshold → BUY >= {best_buy_thr:.2f}  |  SELL <= {best_sell_thr:.2f}  "
      f"(val F1 macro = {best_macro:.4f})")

# ─── Final Model ─────────────────────────────────────────────────────────────
et = ExtraTreesClassifier(
    n_estimators=400, min_samples_leaf=15,
    class_weight='balanced', max_features=0.6,
    random_state=42, n_jobs=-1
)
et.fit(x_train_full, y_train_full)

# ─── Evaluation ──────────────────────────────────────────────────────────────
test_proba  = et.predict_proba(x_test)[:, 1]
y_pred_dual = np.where(test_proba >= best_buy_thr, 1,
              np.where(test_proba <= best_sell_thr, 0, -1))
mask         = y_pred_dual != -1
y_pred_masked = y_pred_dual[mask]
y_test_masked = y_test.values[mask]

n_hold = (y_pred_dual == -1).sum()
n_buy  = (y_pred_dual == 1).sum()
n_sell = (y_pred_dual == 0).sum()

print("\n" + "=" * 57)
print("  BETHACON v4 – ExtraTrees · MODEL EVALUATION")
print("=" * 57)
print(f"  BUY  threshold : >= {best_buy_thr:.2f}   ({n_buy} segnali)")
print(f"  SELL threshold : <= {best_sell_thr:.2f}   ({n_sell} segnali)")
print(f"  HOLD zone      :    ({n_hold} segnali, {n_hold/len(y_pred_dual)*100:.1f}%)")
print(f"\n  Accuracy (BUY+SELL) : {accuracy_score(y_test_masked, y_pred_masked):.4f}")
print(f"  F1 macro (BUY+SELL) : {f1_score(y_test_masked, y_pred_masked, average='macro'):.4f}")
print("\nConfusion Matrix (esclude HOLD):")
print(confusion_matrix(y_test_masked, y_pred_masked))
print("\nClassification Report:")
print(classification_report(y_test_masked, y_pred_masked, target_names=["SELL", "BUY"]))

print("Feature Importances (top 10):")
fi = pd.Series(et.feature_importances_, index=FEATURES).sort_values(ascending=False)
for feat, imp in fi.head(10).items():
    bar = "█" * int(imp * 200)
    print(f"  {feat:<22} {imp:.4f}  {bar}")

# ─── Save ─────────────────────────────────────────────────────────────────────
os.makedirs('model', exist_ok=True)
joblib.dump({
    'model':     et,
    'features':  FEATURES,
    'buy_thr':   best_buy_thr,
    'sell_thr':  best_sell_thr,
    'threshold': best_buy_thr,   # legacy compat
    'algo':      'ExtraTreesClassifier',
}, 'model/bethacon_v4.pkl')
print("\nModello salvato: model/bethacon_v4.pkl")

# ─── Features ────────────────────────────────────────────────────────────────
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

# ─── Load data ────────────────────────────────────────────────────────────────
ds = pd.read_csv('data/processed/eth-dataset-processed.csv')
ds = ds.sort_values('snapped_at').reset_index(drop=True)
ds['signal'] = ds['signal'].astype(int)

x = ds[FEATURES]
y = ds['signal']

train_size = int(len(ds) * 0.70)
val_size   = int(train_size * 0.85)   # 85% of train for training, 15% for threshold tuning

x_train_full = x.iloc[:train_size]
y_train_full = y.iloc[:train_size]
x_test       = x.iloc[train_size:]
y_test       = y.iloc[train_size:]

# ─── Final Model ─────────────────────────────────────────────────────────────
et = ExtraTreesClassifier(
    n_estimators=400, min_samples_leaf=15,
    class_weight='balanced', max_features=0.6,
    random_state=42, n_jobs=-1
)
et.fit(x_train_full, y_train_full)

# ─── Evaluation ──────────────────────────────────────────────────────────────
test_proba  = et.predict_proba(x_test)[:, 1]
y_pred_dual = np.where(test_proba >= best_buy_thr, 1,
              np.where(test_proba <= best_sell_thr, 0, -1))
mask         = y_pred_dual != -1
y_pred_masked = y_pred_dual[mask]
y_test_masked = y_test.values[mask]

n_hold = (y_pred_dual == -1).sum()
n_buy  = (y_pred_dual == 1).sum()
n_sell = (y_pred_dual == 0).sum()

print("\n" + "=" * 57)
print("  BETHACON v4 – ExtraTrees · MODEL EVALUATION")
print("=" * 57)
print(f"  BUY  threshold : >= {best_buy_thr:.2f}   ({n_buy} segnali)")
print(f"  SELL threshold : <= {best_sell_thr:.2f}   ({n_sell} segnali)")
print(f"  HOLD zone      :    ({n_hold} segnali, {n_hold/len(y_pred_dual)*100:.1f}%)")
print(f"\n  Accuracy (BUY+SELL) : {accuracy_score(y_test_masked, y_pred_masked):.4f}")
print(f"  F1 macro (BUY+SELL) : {f1_score(y_test_masked, y_pred_masked, average='macro'):.4f}")
print("\nConfusion Matrix (esclude HOLD):")
print(confusion_matrix(y_test_masked, y_pred_masked))
print("\nClassification Report:")
print(classification_report(y_test_masked, y_pred_masked, target_names=["SELL", "BUY"]))

print("Feature Importances (top 10):")
fi = pd.Series(et.feature_importances_, index=FEATURES).sort_values(ascending=False)
for feat, imp in fi.head(10).items():
    bar = "█" * int(imp * 200)
    print(f"  {feat:<22} {imp:.4f}  {bar}")

# ─── Save ─────────────────────────────────────────────────────────────────────
os.makedirs('model', exist_ok=True)
joblib.dump({
    'model':     et,
    'features':  FEATURES,
    'buy_thr':   best_buy_thr,
    'sell_thr':  best_sell_thr,
    'threshold': best_buy_thr,
    'algo':      'ExtraTreesClassifier',
}, 'model/bethacon_v4.pkl')
print("\nModello salvato: model/bethacon_v4.pkl")