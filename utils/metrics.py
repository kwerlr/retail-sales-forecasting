import numpy as np


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mape(y_true, y_pred, eps=1e-8):
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100


def evaluate(y_true, y_pred, label="Model"):
    m = mae(y_true, y_pred)
    r = rmse(y_true, y_pred)
    mp = mape(y_true, y_pred)
    print(f"[{label}] MAE={m:.2f}  RMSE={r:.2f}  MAPE={mp:.2f}%")
    return {"model": label, "MAE": round(m, 3), "RMSE": round(r, 3), "MAPE": round(mp, 3)}
