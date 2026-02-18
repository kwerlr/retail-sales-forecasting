# 🛒 Retail Sales Forecasting System

An end-to-end **retail demand forecasting system** that mirrors real-world data science workflows used by companies like Walmart, Amazon, and Flipkart. Built with Python, this project covers the full pipeline — from raw data and feature engineering to model training, evaluation, and a live interactive dashboard.

---

## 🖥️ Dashboard Preview
<img width="1901" height="957" alt="image" src="https://github.com/user-attachments/assets/c24068d3-65b5-4ab4-a040-1e2134008f7a" />
<img width="1883" height="819" alt="image" src="https://github.com/user-attachments/assets/ff1e60eb-24bd-412b-9cfa-746aef899771" />
<img width="1439" height="513" alt="image" src="https://github.com/user-attachments/assets/07f93372-0af4-4613-ad03-e3ffc547f8ec" />
<img width="1412" height="861" alt="image" src="https://github.com/user-attachments/assets/4572370d-64ff-4414-860f-bcb5d21daac8" />
<img width="1919" height="902" alt="image" src="https://github.com/user-attachments/assets/fbe65d5e-62f0-4d69-9e8b-10e9a1d61229" />
<img width="1426" height="841" alt="image" src="https://github.com/user-attachments/assets/81eddf76-79ff-4962-b15b-ec97fd9deaf6" />
<img width="1432" height="865" alt="image" src="https://github.com/user-attachments/assets/67e3eeec-18dd-40dc-b180-e3e582a81246" />
<img width="1391" height="831" alt="image" src="https://github.com/user-attachments/assets/d2373876-f298-4eab-8ed3-796fe0f1518e" />
<img width="1871" height="880" alt="image" src="https://github.com/user-attachments/assets/967dd437-a636-44ab-b7da-74245b19943f" />
<img width="1409" height="846" alt="image" src="https://github.com/user-attachments/assets/0804986d-67d1-46a1-997a-a5a606e68f7c" />

---

## 🎯 Problem Statement
Retail companies lose millions every year from two problems:
- **Overstock** — too much inventory sitting on shelves, wasting capital
- **Understock** — running out of product, losing sales and customers

This system forecasts daily sales for every store-product combination and turns those forecasts into concrete inventory decisions — reorder points, safety stock, and stockout risk.

---

## 🏗️ Project Architecture
```
retail_forecast/
├── app.py                  ← Streamlit dashboard (5 pages)
├── requirements.txt
├── data/
│   ├── generate_data.py    ← Synthetic data generator
│   └── prepare_kaggle_data.py  ← Real Kaggle data converter
├── models/
│   └── train_models.py     ← Trains all 4 models
└── utils/
    ├── features.py         ← Feature engineering pipeline
    ├── metrics.py          ← MAE, RMSE, MAPE
    ├── inventory.py        ← Safety stock & reorder simulation
    └── eda.py              ← EDA plot generator
```

---

## 🤖 Models Implemented

| Model | Type | Key Strength |
|---|---|---|
| SARIMA | Statistical | Interpretable baseline, captures weekly seasonality |
| Prophet | Decomposition | Handles holidays, trend changepoints |
| XGBoost | Gradient Boosting | Multi-series, feature-rich, best retail performance |
| LSTM | Deep Learning (PyTorch) | Learns complex long-range temporal patterns |

All models are evaluated on a **time-based train/test split** — never random — to prevent data leakage.

---

## ⚙️ Feature Engineering

| Category | Features |
|---|---|
| Time | day_of_week, month, quarter, is_weekend, day_of_year |
| Fourier | sin/cos of yearly & weekly cycles (smooth seasonality) |
| Lag | sales at lag 1, 7, 14, 28 days |
| Rolling | 7/14/30-day rolling mean & standard deviation |
| External | is_holiday, is_christmas, onpromotion, store_type |

---

## 📊 Evaluation Metrics

- **MAE** — Average units off per day (business-friendly)
- **RMSE** — Penalizes large errors more heavily
- **MAPE** — Percentage error, scale-independent (best for retail)

---

## 📦 Inventory Simulation
Beyond forecasting, the system computes:
- **Safety Stock** — buffer inventory based on demand uncertainty
- **Reorder Point** — when to place a new order given lead time
- **Stockout Probability** — risk of running out before next order
- **Day-by-day trajectory** — simulates inventory levels over forecast horizon

---

## 🖥️ Dashboard Pages
1. **Overview** — Company-wide KPIs, trends, category breakdown
2. **Forecast** — 7–90 day forecast with confidence intervals per store/product
3. **Inventory** — Reorder alerts, safety stock, stockout simulation
4. **Model Comparison** — Leaderboard with MAE/RMSE/MAPE bar charts
5. **EDA** — Autocorrelation, seasonal patterns, promotion impact

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/kwerlr/retail-sales-forecasting.git
cd retail-sales-forecasting
```

### 2. Create virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate data
```bash
python data/generate_data.py
```

### 5. Train models
```bash
python models/train_models.py
```

### 6. Launch dashboard
```bash
streamlit run app.py
```
Open **http://localhost:8501** in your browser.

---

## 📁 Dataset
This project uses a **synthetic dataset** designed to mirror the real-world [Kaggle Store Sales — Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) dataset.

| Property | Value |
|---|---|
| Date range | 2017 – 2020 |
| Stores | 5 (Types A, B, C) |
| Products | 10 items across 5 families |
| Total rows | ~73,000 daily records |
| Patterns | Weekly seasonality, yearly seasonality, promotions, holidays |

To use the real Kaggle dataset, download it and run:
```bash
python data/prepare_kaggle_data.py
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11+ |
| Data | Pandas, NumPy |
| Visualization | Plotly, Matplotlib, Seaborn |
| Statistical Models | Statsmodels (SARIMA), Prophet |
| ML | XGBoost, Scikit-learn |
| Deep Learning | PyTorch (LSTM) |
| Dashboard | Streamlit |

---

## 💼 Skills Demonstrated
- Multi-series time series forecasting
- Feature engineering for tabular ML (lag features, Fourier terms, rolling stats)
- Time-based train/test splitting to prevent data leakage
- Model comparison with proper evaluation metrics
- Inventory optimization and business decision making from ML outputs
- End-to-end ML pipeline from raw data to interactive dashboard
- PyTorch LSTM implementation with recursive multi-step forecasting

---

## 👩‍💻 Author
**Vagisha**  
[GitHub](https://github.com/kwerlr)
