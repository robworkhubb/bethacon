from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import joblib

ds = pd.read_csv('data/processed/eth-dataset-processed.csv')
ds = ds.sort_values('snapped_at')
x = ds[['price_vs_ma7', 'price_vs_ma14', 'momentum_7', 'price_pct_change', 'total_volume', 'rsi', 'macd', 'signal_line', 'histogram']]
y = ds['signal']

train_size = int(len(ds) * 0.7)
x_train = x.iloc[:train_size]
y_train = y.iloc[:train_size]

x_test = x.iloc[train_size:]
y_test = y.iloc[train_size:]

# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=750, learning_rate=0.05,
                                max_depth=3, random_state=42)

# Fit sul training set
gb.fit(x_train, y_train)

y_pred = gb.predict(x_test)
print(y_pred)
# Salva modello per predizioni future
joblib.dump(gb, 'model/bethacon v1.pkl')
