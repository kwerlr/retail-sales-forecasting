"""
Inventory simulation: safety stock, reorder point, stockout probability.
"""
import numpy as np
import pandas as pd


def compute_inventory_metrics(
    forecast: np.ndarray,
    forecast_std: np.ndarray = None,
    lead_time_days: int = 7,
    service_level: float = 0.95,
    current_stock: float = None,
) -> dict:
    """
    forecast       : array of daily demand forecasts (next N days)
    forecast_std   : uncertainty (std dev per day). If None, uses 20% of forecast.
    lead_time_days : supplier lead time
    service_level  : e.g. 0.95 → 95% fill rate
    current_stock  : current on-hand inventory
    """
    if forecast_std is None:
        forecast_std = forecast * 0.20

    z = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}.get(service_level, 1.645)

    avg_daily_demand = forecast.mean()
    std_daily_demand = forecast_std.mean()

    demand_during_lt   = avg_daily_demand * lead_time_days
    std_during_lt      = std_daily_demand * np.sqrt(lead_time_days)
    safety_stock       = z * std_during_lt
    reorder_point      = demand_during_lt + safety_stock
    total_forecast     = forecast.sum()

    if current_stock is not None:
        days_of_stock = current_stock / max(avg_daily_demand, 1e-8)
        stockout_prob = max(0.0, min(1.0, 1 - (current_stock / (reorder_point + 1e-8))))
        needs_reorder = current_stock <= reorder_point
    else:
        days_of_stock = None
        stockout_prob = None
        needs_reorder = None

    return {
        "avg_daily_demand":  round(avg_daily_demand, 2),
        "total_forecast":    round(total_forecast, 2),
        "safety_stock":      round(safety_stock, 2),
        "reorder_point":     round(reorder_point, 2),
        "days_of_stock":     round(days_of_stock, 1) if days_of_stock is not None else None,
        "stockout_prob":     round(stockout_prob * 100, 1) if stockout_prob is not None else None,
        "needs_reorder":     needs_reorder,
        "service_level":     service_level,
        "lead_time_days":    lead_time_days,
    }


def simulate_inventory_trajectory(
    forecast: np.ndarray,
    starting_stock: float,
    reorder_point: float,
    order_qty: float,
    lead_time_days: int = 7,
) -> pd.DataFrame:
    """Simulate day-by-day inventory levels with reordering."""
    stock = starting_stock
    in_transit = 0
    order_pending_day = None
    rows = []

    for day, demand in enumerate(forecast):
        # Receive order if lead time elapsed
        if order_pending_day is not None and day >= order_pending_day + lead_time_days:
            stock += order_qty
            in_transit = 0
            order_pending_day = None

        actual_sales = min(stock, demand)
        lost_sales   = max(0, demand - stock)
        stock -= actual_sales
        stockout = stock <= 0

        reorder_triggered = False
        if stock <= reorder_point and order_pending_day is None:
            order_pending_day = day
            in_transit = order_qty
            reorder_triggered = True

        rows.append({
            "day": day + 1,
            "forecast_demand": round(demand, 2),
            "actual_sales":    round(actual_sales, 2),
            "lost_sales":      round(lost_sales, 2),
            "stock_level":     round(stock, 2),
            "in_transit":      round(in_transit, 2),
            "stockout":        stockout,
            "reorder":         reorder_triggered,
        })

    return pd.DataFrame(rows)
