"""
Generate synthetic retail sales data simulating Kaggle Store Sales dataset.
Run this once to create train.csv and stores.csv
"""
import numpy as np
import pandas as pd
from datetime import date

np.random.seed(42)

# ── Config ──────────────────────────────────────────────────────────────────
START = date(2017, 1, 1)
END   = date(2020, 12, 31)
STORES = list(range(1, 6))          # 5 stores
ITEMS  = [f"ITEM_{i:03d}" for i in range(1, 11)]  # 10 items
FAMILIES = {                        # item → category
    "ITEM_001": "GROCERY", "ITEM_002": "GROCERY",
    "ITEM_003": "BEVERAGES", "ITEM_004": "BEVERAGES",
    "ITEM_005": "CLEANING", "ITEM_006": "CLEANING",
    "ITEM_007": "PRODUCE", "ITEM_008": "PRODUCE",
    "ITEM_009": "DAIRY", "ITEM_010": "DAIRY",
}
STORE_TYPES = {1: "A", 2: "A", 3: "B", 4: "B", 5: "C"}

# ── Stores metadata ──────────────────────────────────────────────────────────
stores_df = pd.DataFrame({
    "store_nbr": STORES,
    "type": [STORE_TYPES[s] for s in STORES],
    "cluster": [1, 1, 2, 2, 3],
})
stores_df.to_csv("data/stores.csv", index=False)

# ── Sales time series ────────────────────────────────────────────────────────
dates = pd.date_range(START, END, freq="D")
rows = []
for store in STORES:
    store_factor = {"A": 1.3, "B": 1.0, "C": 0.7}[STORE_TYPES[store]]
    for item in ITEMS:
        family = FAMILIES[item]
        base   = np.random.uniform(20, 200)
        for dt in dates:
            # Trend
            trend = (dt - pd.Timestamp(START)).days * np.random.uniform(0.005, 0.02)
            # Weekly seasonality (higher on weekends)
            weekly = 1.3 if dt.dayofweek >= 5 else 1.0
            # Yearly seasonality
            yearly = 1 + 0.3 * np.sin(2 * np.pi * dt.dayofyear / 365)
            # Holiday spikes (Christmas, New Year)
            holiday = 1.0
            if (dt.month == 12 and dt.day in range(20, 32)):
                holiday = 1.6
            if (dt.month == 1 and dt.day <= 3):
                holiday = 1.4
            # Promotion (random ~20% of days)
            promo = np.random.choice([0, 1], p=[0.8, 0.2])
            promo_boost = 1.25 if promo else 1.0
            # Noise
            noise = np.random.normal(1.0, 0.1)
            sales = max(0, (base + trend) * store_factor * weekly * yearly * holiday * promo_boost * noise)
            rows.append({
                "date": dt.date(),
                "store_nbr": store,
                "item_nbr": item,
                "family": family,
                "sales": round(sales, 2),
                "onpromotion": promo,
            })

df = pd.DataFrame(rows)
df.to_csv("data/train.csv", index=False)
print(f"✅ Generated {len(df):,} rows → data/train.csv")
print(f"✅ Generated stores metadata → data/stores.csv")
