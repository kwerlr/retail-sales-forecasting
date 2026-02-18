# 🛒 Retail Sales Forecasting System
**Production-grade multi-series retail demand forecasting with Streamlit dashboard.**

Models: SARIMA · Prophet · XGBoost · LSTM  
Features: EDA · Feature Engineering · Inventory Simulation · Model Comparison

---

## 🚀 Quick Start (VS Code)

### 1. Extract & Open
Unzip the project folder, then open it in VS Code:
```
File → Open Folder → select retail_forecast/
```

### 2. Create Virtual Environment
Open the **VS Code terminal** (`Ctrl+`` ` `` `) and run:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
> ⚠️ TensorFlow may take a few minutes. Prophet requires pystan — if it fails, install separately:
> `pip install prophet`

### 4. Generate Data
```bash
python data/generate_data.py
```
This creates `data/train.csv` (synthetic retail sales) and `data/stores.csv`.

### 5. Run EDA (Optional)
```bash
python utils/eda.py
```
Saves 8 analysis plots to `eda_plots/`.

### 6. Train All Models
```bash
python models/train_models.py
```
Trains SARIMA, Prophet, XGBoost, LSTM and saves to `models/`.  
Prints MAE / RMSE / MAPE for each model.

### 7. Launch the Dashboard 🎉
```bash
streamlit run app.py
```
Opens at **http://localhost:8501**

---

## 📁 Project Structure
```
retail_forecast/
├── app.py                  ← Streamlit dashboard (5 pages)
├── requirements.txt
├── data/
│   ├── generate_data.py    ← Synthetic data generator
│   ├── train.csv           ← Generated sales data
│   └── stores.csv          ← Store metadata
├── models/
│   ├── train_models.py     ← Trains all 4 models
│   ├── results.csv         ← Model comparison metrics
│   ├── xgb_model.pkl
│   ├── sarima_model.pkl
│   ├── prophet_model.pkl
│   └── lstm_model.keras
└── utils/
    ├── features.py         ← Feature engineering pipeline
    ├── metrics.py          ← MAE, RMSE, MAPE
    ├── inventory.py        ← Safety stock, reorder point simulation
    └── eda.py              ← EDA plots generator
```

---

## 🧠 Feature Engineering
| Category | Features |
|---|---|
| Time | day_of_week, month, quarter, is_weekend, day_of_year |
| Fourier | sin/cos of yearly & weekly cycles |
| Lag | sales at lag 1, 7, 14, 28 days |
| Rolling | 7/14/30-day rolling mean & std |
| External | is_holiday, is_christmas, onpromotion |

---

## 📊 Models
| Model | Type | Best for |
|---|---|---|
| SARIMA | Statistical | Interpretable single-series baseline |
| Prophet | Decomposition | Holidays, missing data, business seasonality |
| XGBoost | Gradient Boosting | Multi-series, feature-rich, production |
| LSTM | Deep Learning | Complex long-range temporal patterns |

---

## 📦 Dashboard Pages
- **Overview** — KPIs, trend, category breakdown, DOW patterns
- **Forecast** — 7–90 day forecast with confidence intervals + promo analysis
- **Inventory** — Safety stock, reorder point, stockout simulation
- **Model Comparison** — Leaderboard with MAE/RMSE/MAPE charts
- **EDA** — Autocorrelation, distribution, seasonal decomposition

---

## 🔗 Using Real Kaggle Data
Replace the generated files with real data from:
https://www.kaggle.com/competitions/store-sales-time-series-forecasting

Rename/adapt columns to match:
- `date`, `store_nbr`, `item_nbr`, `family`, `sales`, `onpromotion`

---

## 💼 Portfolio Tips
- Add this to GitHub with a good README and screenshot
- Record a Loom walkthrough of the dashboard
- Mention: "multi-series forecasting with XGBoost and LSTM, inventory optimization, and a live Streamlit dashboard"
