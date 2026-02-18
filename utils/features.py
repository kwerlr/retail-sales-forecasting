"""
Feature engineering: time features, lag features, rolling stats, holidays.
"""
import pandas as pd
import numpy as np


def make_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_week"]   = df["date"].dt.dayofweek
    df["day_of_month"]  = df["date"].dt.day
    df["week_of_year"]  = df["date"].dt.isocalendar().week.astype(int)
    df["month"]         = df["date"].dt.month
    df["quarter"]       = df["date"].dt.quarter
    df["year"]          = df["date"].dt.year
    df["is_weekend"]    = (df["day_of_week"] >= 5).astype(int)
    df["day_of_year"]   = df["date"].dt.dayofyear
    # Fourier terms for yearly seasonality
    df["sin_year"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["cos_year"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    df["sin_week"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["cos_week"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


def make_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    # Simple rule-based holidays
    df["is_christmas"]  = ((df["date"].dt.month == 12) & (df["date"].dt.day.between(20, 31))).astype(int)
    df["is_new_year"]   = ((df["date"].dt.month == 1)  & (df["date"].dt.day <= 3)).astype(int)
    df["is_holiday"]    = (df["is_christmas"] | df["is_new_year"]).astype(int)
    return df


def make_lag_features(df: pd.DataFrame, group_cols: list, target: str, lags: list) -> pd.DataFrame:
    df = df.copy().sort_values(["date"] + group_cols)
    for lag in lags:
        col = f"{target}_lag_{lag}"
        df[col] = df.groupby(group_cols)[target].shift(lag)
    return df


def make_rolling_features(df: pd.DataFrame, group_cols: list, target: str, windows: list) -> pd.DataFrame:
    df = df.copy().sort_values(["date"] + group_cols)
    for w in windows:
        df[f"{target}_roll_mean_{w}"] = (
            df.groupby(group_cols)[target]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        )
        df[f"{target}_roll_std_{w}"] = (
            df.groupby(group_cols)[target]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).std())
        )
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature pipeline."""
    df = make_time_features(df)
    df = make_holiday_features(df)
    df = make_lag_features(df, ["store_nbr", "item_nbr"], "sales", lags=[1, 7, 14, 28])
    df = make_rolling_features(df, ["store_nbr", "item_nbr"], "sales", windows=[7, 14, 30])
    return df


FEATURE_COLS = [
    "day_of_week", "day_of_month", "week_of_year", "month", "quarter", "year",
    "is_weekend", "sin_year", "cos_year", "sin_week", "cos_week",
    "is_holiday", "is_christmas", "is_new_year", "onpromotion",
    "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_28",
    "sales_roll_mean_7", "sales_roll_std_7",
    "sales_roll_mean_14", "sales_roll_std_14",
    "sales_roll_mean_30", "sales_roll_std_30",
]
