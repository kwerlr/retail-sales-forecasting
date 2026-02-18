"""
Exploratory Data Analysis — run to generate EDA plots in eda_plots/
python utils/eda.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
from statsmodels.tsa.seasonal import seasonal_decompose

OUT = Path("eda_plots")
OUT.mkdir(exist_ok=True)
sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 120, "font.size": 11})

print("📦 Loading data...")
df = pd.read_csv("data/train.csv", parse_dates=["date"])
stores = pd.read_csv("data/stores.csv")
df = df.merge(stores, on="store_nbr", how="left")

# ── 1. Overall sales trend ──────────────────────────────────────────────────
print("📈 Plot 1: Overall sales trend")
daily = df.groupby("date")["sales"].sum().reset_index()
daily["roll7"] = daily["sales"].rolling(7).mean()
daily["roll30"] = daily["sales"].rolling(30).mean()

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(daily["date"], daily["sales"], alpha=0.3, color="steelblue", label="Daily")
ax.plot(daily["date"], daily["roll7"],  color="orange",    lw=1.5, label="7-day MA")
ax.plot(daily["date"], daily["roll30"], color="crimson",   lw=2,   label="30-day MA")
ax.set_title("Overall Sales Trend (All Stores & Items)", fontweight="bold")
ax.set_xlabel("Date"); ax.set_ylabel("Total Sales")
ax.legend(); ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.xticks(rotation=45); plt.tight_layout()
plt.savefig(OUT / "01_sales_trend.png"); plt.close()

# ── 2. Sales by store ───────────────────────────────────────────────────────
print("📈 Plot 2: Sales by store")
store_monthly = df.groupby(["date", "store_nbr"])["sales"].sum().reset_index()
store_monthly = store_monthly.set_index("date").groupby("store_nbr")["sales"].resample("M").sum().reset_index()

fig, ax = plt.subplots(figsize=(14, 5))
for s in df["store_nbr"].unique():
    d = store_monthly[store_monthly.store_nbr == s]
    ax.plot(d["date"], d["sales"], label=f"Store {s}", lw=1.5)
ax.set_title("Monthly Sales by Store", fontweight="bold")
ax.set_xlabel("Date"); ax.set_ylabel("Monthly Sales")
ax.legend(); plt.tight_layout()
plt.savefig(OUT / "02_sales_by_store.png"); plt.close()

# ── 3. Seasonal decomposition (Store 1, Item 1) ─────────────────────────────
print("📈 Plot 3: Seasonal decomposition")
ser = df[(df.store_nbr == 1) & (df.item_nbr == "ITEM_001")].set_index("date")["sales"]
ser = ser.asfreq("D").fillna(method="ffill")
result = seasonal_decompose(ser, model="additive", period=7)

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
for ax, comp, label in zip(axes,
    [ser, result.trend, result.seasonal, result.resid],
    ["Observed", "Trend", "Seasonal", "Residual"]):
    ax.plot(comp, lw=1); ax.set_ylabel(label)
axes[0].set_title("Seasonal Decomposition — Store 1, ITEM_001", fontweight="bold")
plt.tight_layout(); plt.savefig(OUT / "03_decomposition.png"); plt.close()

# ── 4. Day-of-week patterns ─────────────────────────────────────────────────
print("📈 Plot 4: Day-of-week patterns")
df["dow"] = df["date"].dt.day_name()
dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
dow_avg = df.groupby("dow")["sales"].mean().reindex(dow_order)

fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.bar(dow_avg.index, dow_avg.values,
              color=["#4C72B0"]*5 + ["#DD8452"]*2)
ax.set_title("Average Sales by Day of Week", fontweight="bold")
ax.set_ylabel("Avg Sales")
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout(); plt.savefig(OUT / "04_dow_pattern.png"); plt.close()

# ── 5. Promotion impact ─────────────────────────────────────────────────────
print("📈 Plot 5: Promotion impact")
promo = df.groupby("onpromotion")["sales"].mean()
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(["No Promo", "On Promo"], promo.values, color=["#4C72B0", "#2ca02c"])
ax.set_title("Average Sales: Promo vs No Promo", fontweight="bold")
ax.set_ylabel("Avg Sales")
for i, v in enumerate(promo.values):
    ax.text(i, v + 1, f"{v:.1f}", ha="center", va="bottom")
pct = (promo.values[1] / promo.values[0] - 1) * 100
ax.set_xlabel(f"Promotions boost sales by ~{pct:.1f}%")
plt.tight_layout(); plt.savefig(OUT / "05_promo_impact.png"); plt.close()

# ── 6. Family / category breakdown ─────────────────────────────────────────
print("📈 Plot 6: Category breakdown")
fam = df.groupby("family")["sales"].sum().sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(fam.index, fam.values, color=sns.color_palette("Set2", len(fam)))
ax.set_title("Total Sales by Product Family", fontweight="bold")
ax.set_xlabel("Total Sales")
plt.tight_layout(); plt.savefig(OUT / "06_family_breakdown.png"); plt.close()

# ── 7. Correlation heatmap (lag features) ───────────────────────────────────
print("📈 Plot 7: Lag correlation heatmap")
lag_df = ser.copy().to_frame(name="sales")
for lag in [1, 7, 14, 28, 30]:
    lag_df[f"lag_{lag}"] = lag_df["sales"].shift(lag)
lag_df = lag_df.dropna()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(lag_df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
            ax=ax, square=True)
ax.set_title("Correlation: Sales vs Lag Features", fontweight="bold")
plt.tight_layout(); plt.savefig(OUT / "07_lag_correlation.png"); plt.close()

# ── 8. Monthly seasonality heatmap ─────────────────────────────────────────
print("📈 Plot 8: Monthly heatmap")
df["month"]  = df["date"].dt.month
df["year"]   = df["date"].dt.year
pivot = df.pivot_table(values="sales", index="year", columns="month", aggfunc="mean")
fig, ax = plt.subplots(figsize=(12, 4))
sns.heatmap(pivot, cmap="YlOrRd", annot=True, fmt=".0f", ax=ax)
ax.set_title("Average Sales by Year × Month", fontweight="bold")
months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
ax.set_xticklabels(months)
plt.tight_layout(); plt.savefig(OUT / "08_monthly_heatmap.png"); plt.close()

print(f"\n✅ All EDA plots saved to {OUT}/")
