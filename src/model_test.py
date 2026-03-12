from main import x_train, x_test, y_test, y_train, ds
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score; 
from sklearn.metrics import confusion_matrix;
from plot import plot_signals

log = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=500)
rf = RandomForestClassifier(n_estimators=750, max_depth=24, min_samples_split=2, random_state=42)
gb = GradientBoostingClassifier(n_estimators=750, learning_rate=0.05, max_depth=3, random_state=42)

log.fit(x_train, y_train)
rf.fit(x_train, y_train)
gb.fit(x_train, y_train)

y_pred_log = log.predict(x_test)
y_pred_rf = rf.predict(x_test)
y_pred_gb = gb.predict(x_test)

# Accuracy
log.score(x_test, y_test)
rf.score(x_test, y_test)
gb.score(x_test, y_test)

# F1-score
f1_score(y_test, y_pred_log, average='macro')
f1_score(y_test, y_pred_rf, average='macro')
f1_score(y_test, y_pred_gb, average='macro')

# Confusion Matrix
confusion_matrix(y_test, y_pred_log)
confusion_matrix(y_test, y_pred_rf)
confusion_matrix(y_test, y_pred_gb)

models = {
    'Logistic Regression': (log, y_pred_log),
    'Random Forest': (rf, y_pred_rf),
    'Gradient Boosting': (gb, y_pred_gb)
}

for name, (model, y_pred) in models.items():
    print(f"=== {name} ===")
    print(f"Accuracy: {model.score(x_test, y_test):.4f}")
    print(f"F1-score: {f1_score(y_test, y_pred, average='macro'):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\n")
    
models_dict = {
    'Gradient Boosting': gb
}
plot_signals(ds, models_dict, 500)