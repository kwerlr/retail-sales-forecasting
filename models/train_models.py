"""
Train & evaluate all models. Saves results to models/results.csv.
Run: python models/train_models.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path

from utils.features import build_features, FEATURE_COLS
from utils.metrics import evaluate

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
print("📦 Loading data...")
df = pd.read_csv("data/train.csv", parse_dates=["date"])
stores = pd.read_csv("data/stores.csv")
df = df.merge(stores[["store_nbr", "type"]], on="store_nbr", how="left")

# Work on a single store+item to keep training fast for ARIMA/LSTM
STORE, ITEM = 1, "ITEM_001"
ser = df[(df.store_nbr == STORE) & (df.item_nbr == ITEM)].copy().sort_values("date").reset_index(drop=True)

# Time split
CUTOFF_VAL  = "2020-07-01"
CUTOFF_TEST = "2020-10-01"
train = ser[ser.date < CUTOFF_VAL]
val   = ser[(ser.date >= CUTOFF_VAL) & (ser.date < CUTOFF_TEST)]
test  = ser[ser.date >= CUTOFF_TEST]

all_results = []

# ────────────────────────────────────────────────────────────────────────────
# 1. SARIMA
# ────────────────────────────────────────────────────────────────────────────
print("\n🔹 Training SARIMA...")
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    sarima_model = SARIMAX(
        train["sales"],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 0, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    sarima_pred = sarima_model.forecast(steps=len(test))
    sarima_pred = np.maximum(sarima_pred, 0)
    r = evaluate(test["sales"].values, sarima_pred, "SARIMA")
    all_results.append(r)
    sarima_model.save(str(MODELS_DIR / "sarima_model.pkl"))
    print("  ✅ SARIMA done")
except Exception as e:
    print(f"  ⚠️ SARIMA failed: {e}")
    all_results.append({"model": "SARIMA", "MAE": None, "RMSE": None, "MAPE": None})

# ────────────────────────────────────────────────────────────────────────────
# 2. Prophet
# ────────────────────────────────────────────────────────────────────────────
print("\n🔹 Training Prophet...")
try:
    from prophet import Prophet
    prophet_df = train[["date", "sales"]].rename(columns={"date": "ds", "sales": "y"})
    prophet_model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
    )
    prophet_model.fit(prophet_df)
    future = prophet_model.make_future_dataframe(periods=len(test))
    forecast = prophet_model.predict(future)
    prophet_pred = forecast.tail(len(test))["yhat"].values
    prophet_pred = np.maximum(prophet_pred, 0)
    r = evaluate(test["sales"].values, prophet_pred, "Prophet")
    all_results.append(r)
    joblib.dump(prophet_model, MODELS_DIR / "prophet_model.pkl")
    print("  ✅ Prophet done")
except Exception as e:
    print(f"  ⚠️ Prophet failed: {e}")
    all_results.append({"model": "Prophet", "MAE": None, "RMSE": None, "MAPE": None})

# ────────────────────────────────────────────────────────────────────────────
# 3. XGBoost (multi-series, all stores/items)
# ────────────────────────────────────────────────────────────────────────────
print("\n🔹 Training XGBoost (full dataset)...")
try:
    import xgboost as xgb
    from sklearn.preprocessing import LabelEncoder

    feat_df = build_features(df.copy())
    feat_df = feat_df.dropna(subset=FEATURE_COLS)

    # Encode categoricals
    le_store = LabelEncoder().fit(feat_df["store_nbr"])
    le_item  = LabelEncoder().fit(feat_df["item_nbr"])
    feat_df["store_enc"] = le_store.transform(feat_df["store_nbr"])
    feat_df["item_enc"]  = le_item.transform(feat_df["item_nbr"])

    XGB_FEATURES = FEATURE_COLS + ["store_enc", "item_enc"]

    tr = feat_df[feat_df.date < CUTOFF_VAL]
    te = feat_df[feat_df.date >= CUTOFF_TEST]

    xgb_model = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.05,
        max_depth=6, subsample=0.8,
        colsample_bytree=0.8, n_jobs=-1,
        random_state=42, verbosity=0,
    )
    xgb_model.fit(
        tr[XGB_FEATURES], tr["sales"],
        eval_set=[(te[XGB_FEATURES], te["sales"])],
        verbose=False,
    )
    xgb_pred = np.maximum(xgb_model.predict(te[XGB_FEATURES]), 0)
    r = evaluate(te["sales"].values, xgb_pred, "XGBoost")
    all_results.append(r)

    joblib.dump(xgb_model, MODELS_DIR / "xgb_model.pkl")
    joblib.dump(le_store,  MODELS_DIR / "le_store.pkl")
    joblib.dump(le_item,   MODELS_DIR / "le_item.pkl")
    joblib.dump(XGB_FEATURES, MODELS_DIR / "xgb_features.pkl")
    print("  ✅ XGBoost done")
except Exception as e:
    print(f"  ⚠️ XGBoost failed: {e}")
    all_results.append({"model": "XGBoost", "MAE": None, "RMSE": None, "MAPE": None})

# ────────────────────────────────────────────────────────────────────────────
# 4. LSTM (PyTorch)
# ────────────────────────────────────────────────────────────────────────────
print("\n🔹 Training LSTM (PyTorch)...")
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import MinMaxScaler

    SEQ_LEN = 30
    EPOCHS   = 30
    BATCH    = 32
    DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler = MinMaxScaler()
    sales_vals = ser["sales"].values.reshape(-1, 1)
    scaled = scaler.fit_transform(sales_vals).flatten()

    n_train = len(train)
    n_test  = len(test)

    def make_sequences(data, seq_len):
        X, y = [], []
        for i in range(seq_len, len(data)):
            X.append(data[i - seq_len:i])
            y.append(data[i])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    X_tr, y_tr = make_sequences(scaled[:n_train], SEQ_LEN)
    X_tr_t = torch.tensor(X_tr).unsqueeze(-1).to(DEVICE)   # (N, SEQ_LEN, 1)
    y_tr_t = torch.tensor(y_tr).unsqueeze(-1).to(DEVICE)   # (N, 1)

    dataset = TensorDataset(X_tr_t, y_tr_t)
    loader  = DataLoader(dataset, batch_size=BATCH, shuffle=True)

    # ── Model ──────────────────────────────────────────────────────────────
    class LSTMModel(nn.Module):
        def __init__(self, input_size=1, hidden=64, hidden2=32, dropout=0.2):
            super().__init__()
            self.lstm1 = nn.LSTM(input_size, hidden, batch_first=True)
            self.drop1 = nn.Dropout(dropout)
            self.lstm2 = nn.LSTM(hidden, hidden2, batch_first=True)
            self.drop2 = nn.Dropout(dropout)
            self.fc    = nn.Linear(hidden2, 1)

        def forward(self, x):
            out, _ = self.lstm1(x)
            out = self.drop1(out)
            out, _ = self.lstm2(out)
            out = self.drop2(out[:, -1, :])   # last timestep
            return self.fc(out)

    model     = LSTMModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{EPOCHS}  loss={epoch_loss/len(loader):.5f}")

    # ── Recursive forecast ─────────────────────────────────────────────────
    model.eval()
    context = scaled[n_train - SEQ_LEN: n_train].tolist()
    lstm_preds_scaled = []
    with torch.no_grad():
        for _ in range(n_test):
            inp = torch.tensor(
                np.array(context[-SEQ_LEN:], dtype=np.float32)
            ).unsqueeze(0).unsqueeze(-1).to(DEVICE)          # (1, SEQ_LEN, 1)
            p = model(inp).item()
            lstm_preds_scaled.append(p)
            context.append(p)

    lstm_pred = scaler.inverse_transform(
        np.array(lstm_preds_scaled, dtype=np.float32).reshape(-1, 1)
    ).flatten()
    lstm_pred = np.maximum(lstm_pred, 0)

    r = evaluate(test["sales"].values, lstm_pred, "LSTM")
    all_results.append(r)
    torch.save(model.state_dict(), str(MODELS_DIR / "lstm_model.pt"))
    joblib.dump(scaler, MODELS_DIR / "lstm_scaler.pkl")
    print("  ✅ LSTM (PyTorch) done")
except Exception as e:
    print(f"  ⚠️ LSTM failed: {e}")
    all_results.append({"model": "LSTM", "MAE": None, "RMSE": None, "MAPE": None})

# ── Save results ─────────────────────────────────────────────────────────────
results_df = pd.DataFrame(all_results)
results_df.to_csv(MODELS_DIR / "results.csv", index=False)
print("\n📊 Model Comparison:")
print(results_df.to_string(index=False))
print("\n✅ All models saved to models/")
