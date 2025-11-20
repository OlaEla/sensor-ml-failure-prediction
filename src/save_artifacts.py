# Код для сохранения артефактов
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score

# Загрузка подготовленных данных 
X_train = pd.read_csv('../data/processed/X_train.csv')
X_test = pd.read_csv('../data/processed/X_test.csv')
y_train = pd.read_csv('../data/processed/y_train.csv').values.ravel()
y_test = pd.read_csv('../data/processed/y_test.csv').values.ravel()

# Стандартизация признаков
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Обучение модели (Логистическая регрессия)
model = LogisticRegression(
    C=0.8,
    class_weight='balanced',  # Учитываем дисбаланс классов
    max_iter=1000,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# Оценка на тестовых данных
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Расчет метрик
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Сохранение модели и скейлера для последующего использования
with open('../artifacts/failure_prediction_model.pkl', 'wb') as file:
    pickle.dump(model, file)

with open('../artifacts/feature_scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

# Сохранение признаков
model_features = X_train.columns.tolist()
with open('../artifacts/model_features.pkl', 'wb') as file:
    pickle.dump(model_features, file)

print("✅ Все артефакты успешно сохранены.")
