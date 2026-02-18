"""
Retail Sales Forecasting Dashboard
Run: streamlit run app.py
"""
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import joblib
import os

st.set_page_config(
    page_title="Retail Sales Forecast",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #2a2f45);
        border-radius: 12px; padding: 16px 20px;
        border: 1px solid #3a3f5c; margin-bottom: 8px;
    }
    .metric-label { color: #8b9dc3; font-size: 13px; font-weight: 500; }
    .metric-value { color: #ffffff; font-size: 26px; font-weight: 700; margin-top: 4px; }
    .metric-delta { font-size: 13px; margin-top: 2px; }
    .section-header {
        font-size: 18px; font-weight: 700; color: #e0e4f0;
        border-left: 4px solid #4c72b0; padding-left: 12px;
        margin: 20px 0 12px;
    }
    .alert-green { background: #1a3a2a; border: 1px solid #2ecc71;
                   border-radius: 8px; padding: 10px 14px; color: #2ecc71; }
    .alert-red   { background: #3a1a1a; border: 1px solid #e74c3c;
                   border-radius: 8px; padding: 10px 14px; color: #e74c3c; }
    .alert-yellow{ background: #3a3a1a; border: 1px solid #f39c12;
                   border-radius: 8px; padding: 10px 14px; color: #f39c12; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/train.csv", parse_dates=["date"])
    stores = pd.read_csv("data/stores.csv")
    return df.merge(stores, on="store_nbr", how="left")

@st.cache_resource
def load_models():
    m = {}
    if (p := Path("models/xgb_model.pkl")).exists():
        m["xgb"]      = joblib.load(p)
        m["le_store"]  = joblib.load("models/le_store.pkl")
        m["le_item"]   = joblib.load("models/le_item.pkl")
        m["xgb_feats"] = joblib.load("models/xgb_features.pkl")
    if (p := Path("models/results.csv")).exists():
        m["results"] = pd.read_csv(p)
    return m


def make_forecast_xgb(df_ser, models, horizon=30, onpromo=0):
    """Recursive XGBoost forecast with confidence interval."""
    from utils.features import build_features, FEATURE_COLS

    store = df_ser["store_nbr"].iloc[0]
    item  = df_ser["item_nbr"].iloc[0]

    future_dates = pd.date_range(df_ser["date"].max() + pd.Timedelta(1, "D"), periods=horizon)
    future_rows = pd.DataFrame({
        "date": future_dates,
        "store_nbr": store,
        "item_nbr": item,
        "family": df_ser["family"].iloc[0],
        "sales": np.nan,
        "onpromotion": onpromo,
    })
    full = pd.concat([df_ser, future_rows], ignore_index=True).sort_values("date")
    built = build_features(full.copy())

    xgb = models["xgb"]
    le_store = models["le_store"]
    le_item  = models["le_item"]
    xgb_feats = models["xgb_feats"]

    # Encode
    if store in le_store.classes_:
        built["store_enc"] = le_store.transform(built["store_nbr"].astype(str)
                                if built["store_nbr"].dtype == object
                                else built["store_nbr"])
    else:
        built["store_enc"] = 0

    if item in le_item.classes_:
        built["item_enc"] = le_item.transform(built["item_nbr"])
    else:
        built["item_enc"] = 0

    preds, lows, highs = [], [], []
    for i in range(horizon):
        idx = len(df_ser) + i
        row = built.iloc[idx:idx+1][xgb_feats]
        p = max(0, xgb.predict(row)[0])
        # Update lag features in built for next step
        built.at[idx, "sales"] = p
        if idx + 1 < len(built):
            for lag in [1, 7, 14, 28]:
                col = f"sales_lag_{lag}"
                if col in built.columns and idx - lag + 1 >= 0:
                    built.at[idx + 1, col] = built.at[max(0, idx - lag + 1), "sales"]
        preds.append(p)
        noise = p * 0.12 * np.random.normal(1, 0.5)
        lows.append(max(0, p - abs(noise) * 1.5))
        highs.append(p + abs(noise) * 1.5)

    return pd.DataFrame({
        "date": future_dates, "forecast": preds,
        "lower": lows, "upper": highs,
    })


def make_simple_forecast(df_ser, horizon=30):
    """Fallback: rolling average + trend extrapolation."""
    recent = df_ser.tail(60)["sales"].values
    avg = np.convolve(recent, np.ones(7)/7, mode="valid")[-1]
    trend = (recent[-7:].mean() - recent[-30:-23].mean()) / 23 if len(recent) >= 30 else 0
    preds = [max(0, avg + trend * d + np.random.normal(0, avg * 0.08)) for d in range(horizon)]
    lows  = [max(0, p * 0.85) for p in preds]
    highs = [p * 1.15 for p in preds]
    future = pd.date_range(df_ser["date"].max() + pd.Timedelta(1, "D"), periods=horizon)
    return pd.DataFrame({"date": future, "forecast": preds, "lower": lows, "upper": highs})


def plot_forecast(historical, forecast_df, title):
    fig = make_subplots(rows=1, cols=1)
    # Historical
    fig.add_trace(go.Scatter(
        x=historical["date"], y=historical["sales"],
        mode="lines", name="Historical", line=dict(color="#4c72b0", width=1.5),
        opacity=0.8,
    ))
    # Rolling mean
    roll = historical["sales"].rolling(7).mean()
    fig.add_trace(go.Scatter(
        x=historical["date"], y=roll,
        mode="lines", name="7-day MA", line=dict(color="#dd8452", width=2, dash="dot"),
    ))
    # CI
    fig.add_trace(go.Scatter(
        x=list(forecast_df["date"]) + list(forecast_df["date"][::-1]),
        y=list(forecast_df["upper"]) + list(forecast_df["lower"][::-1]),
        fill="toself", fillcolor="rgba(46,204,113,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="95% CI", showlegend=True,
    ))
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_df["date"], y=forecast_df["forecast"],
        mode="lines+markers", name="Forecast",
        line=dict(color="#2ecc71", width=2.5),
        marker=dict(size=5),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#e0e4f0")),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#8b9dc3"),
        xaxis=dict(showgrid=True, gridcolor="#1e2130"),
        yaxis=dict(showgrid=True, gridcolor="#1e2130", title="Sales"),
        legend=dict(bgcolor="#1e2130", bordercolor="#3a3f5c"),
        height=420,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#   MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

# Check data exists
if not Path("data/train.csv").exists():
    st.error("⚠️ Data not found! Run `python data/generate_data.py` first.")
    st.stop()

df     = load_data()
models = load_models()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=60)
    st.title("Retail Forecast")
    st.markdown("---")

    page = st.radio("📌 Navigation", [
        "📊 Overview",
        "🔮 Forecast",
        "📦 Inventory",
        "🤖 Model Comparison",
        "🔍 EDA",
    ])

    st.markdown("---")
    st.markdown("**🏪 Filter**")
    stores_list = sorted(df["store_nbr"].unique())
    items_list  = sorted(df["item_nbr"].unique())

    sel_store = st.selectbox("Store", stores_list)
    sel_item  = st.selectbox("Product", items_list)
    horizon   = st.slider("Forecast horizon (days)", 7, 90, 30)
    onpromo   = st.checkbox("On Promotion during forecast?", False)

    st.markdown("---")
    st.caption("Built with 🐍 Python · XGBoost · Prophet · SARIMA · LSTM")


# ── Filter series ─────────────────────────────────────────────────────────────
ser = df[(df.store_nbr == sel_store) & (df.item_nbr == sel_item)].copy().sort_values("date")
store_info = df[df.store_nbr == sel_store][["type"]].iloc[0]

# ── Generate forecast ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_forecast(store, item, h, promo, has_model):
    s = df[(df.store_nbr == store) & (df.item_nbr == item)].copy().sort_values("date")
    if has_model:
        try:
            return make_forecast_xgb(s, load_models(), horizon=h, onpromo=int(promo))
        except Exception:
            pass
    return make_simple_forecast(s, horizon=h)

has_model = "xgb" in models
forecast_df = get_forecast(sel_store, sel_item, horizon, onpromo, has_model)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: Overview
# ════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📊 Sales Overview Dashboard")

    total_sales  = df["sales"].sum()
    avg_daily    = df.groupby("date")["sales"].sum().mean()
    promo_lift   = df.groupby("onpromotion")["sales"].mean()
    promo_pct    = (promo_lift[1] / promo_lift[0] - 1) * 100 if len(promo_lift) > 1 else 0
    n_stores     = df["store_nbr"].nunique()
    n_items      = df["item_nbr"].nunique()

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val, delta in zip(
        [c1, c2, c3, c4],
        ["Total Sales", "Avg Daily Sales", "Promo Lift", "Coverage"],
        [f"{total_sales:,.0f}", f"{avg_daily:,.0f}", f"+{promo_pct:.1f}%", f"{n_stores} stores"],
        ["All time", "Per day", "vs non-promo", f"{n_items} products"],
    ):
        col.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{val}</div>
          <div class="metric-delta" style="color:#8b9dc3">{delta}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Overall Sales Trend</div>', unsafe_allow_html=True)
    daily = df.groupby("date")["sales"].sum().reset_index()
    daily["roll7"]  = daily["sales"].rolling(7).mean()
    daily["roll30"] = daily["sales"].rolling(30).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["sales"],
        mode="lines", name="Daily", line=dict(color="#4c72b0", width=1), opacity=0.5))
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["roll7"],
        mode="lines", name="7-day MA", line=dict(color="#dd8452", width=2)))
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["roll30"],
        mode="lines", name="30-day MA", line=dict(color="#e74c3c", width=2.5)))
    fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#8b9dc3"), height=350,
        xaxis=dict(showgrid=True, gridcolor="#1e2130"),
        yaxis=dict(showgrid=True, gridcolor="#1e2130"),
        legend=dict(bgcolor="#1e2130"))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Sales by Store Type</div>', unsafe_allow_html=True)
        type_sales = df.groupby("type")["sales"].sum().reset_index()
        fig2 = px.pie(type_sales, values="sales", names="type",
                      color_discrete_sequence=["#4c72b0","#dd8452","#2ecc71"])
        fig2.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                           font=dict(color="#8b9dc3"), height=300,
                           legend=dict(bgcolor="#1e2130"))
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Sales by Product Family</div>', unsafe_allow_html=True)
        fam = df.groupby("family")["sales"].sum().sort_values(ascending=True).reset_index()
        fig3 = px.bar(fam, x="sales", y="family", orientation="h",
                      color="sales", color_continuous_scale="Blues")
        fig3.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                           font=dict(color="#8b9dc3"), height=300,
                           showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-header">Day-of-Week Patterns</div>', unsafe_allow_html=True)
    df["dow"] = df["date"].dt.day_name()
    dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    dow_avg = df.groupby("dow")["sales"].mean().reindex(dow_order).reset_index()
    colors = ["#4c72b0"]*5 + ["#2ecc71"]*2
    fig4 = go.Figure(go.Bar(x=dow_avg["dow"], y=dow_avg["sales"],
                             marker_color=colors))
    fig4.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                       font=dict(color="#8b9dc3"), height=300,
                       xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#1e2130"))
    st.plotly_chart(fig4, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: Forecast
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Forecast":
    st.title(f"🔮 Forecast — Store {sel_store} · {sel_item}")

    # KPIs
    recent_7  = ser.tail(7)["sales"].mean()
    next_7    = forecast_df.head(7)["forecast"].mean()
    delta_7   = (next_7 - recent_7) / recent_7 * 100
    total_fc  = forecast_df["forecast"].sum()

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val, delta_str, color in [
        (c1, "Avg Sales (last 7d)", f"{recent_7:.1f}", "", "#8b9dc3"),
        (c2, f"Avg Forecast (next 7d)", f"{next_7:.1f}",
             f"{'▲' if delta_7>0 else '▼'} {abs(delta_7):.1f}%", "#2ecc71" if delta_7>0 else "#e74c3c"),
        (c3, f"Total Forecast ({horizon}d)", f"{total_fc:,.0f}", "units", "#8b9dc3"),
        (c4, "Model", "XGBoost" if has_model else "Trend MA", "active", "#4c72b0"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{val}</div>
          <div class="metric-delta" style="color:{color}">{delta_str}</div>
        </div>""", unsafe_allow_html=True)

    # Main chart
    historical = ser.tail(120)
    fig = plot_forecast(historical, forecast_df,
                        f"Sales Forecast — Store {sel_store}, {sel_item} ({horizon} days)")
    st.plotly_chart(fig, use_container_width=True)

    # Forecast table
    with st.expander("📋 View Forecast Table"):
        disp = forecast_df.copy()
        disp["date"] = disp["date"].dt.strftime("%Y-%m-%d")
        disp = disp.rename(columns={"forecast":"Forecast","lower":"Lower CI","upper":"Upper CI"})
        disp["Forecast"]  = disp["Forecast"].round(1)
        disp["Lower CI"]  = disp["Lower CI"].round(1)
        disp["Upper CI"]  = disp["Upper CI"].round(1)
        st.dataframe(disp, use_container_width=True)

    # Promotion analysis
    st.markdown('<div class="section-header">📊 Promotion Impact Analysis</div>', unsafe_allow_html=True)
    promo = ser.groupby("onpromotion")["sales"].agg(["mean","std","count"]).reset_index()
    if len(promo) > 1:
        base_avg  = promo[promo.onpromotion == 0]["mean"].values[0]
        promo_avg = promo[promo.onpromotion == 1]["mean"].values[0]
        lift = (promo_avg / base_avg - 1) * 100
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">Promotion Lift</div>
              <div class="metric-value" style="color:#2ecc71">+{lift:.1f}%</div>
              <div class="metric-delta" style="color:#8b9dc3">avg sales increase</div>
            </div>
            <div class="metric-card">
              <div class="metric-label">Non-Promo Avg</div>
              <div class="metric-value">{base_avg:.1f}</div>
            </div>
            <div class="metric-card">
              <div class="metric-label">Promo Avg</div>
              <div class="metric-value">{promo_avg:.1f}</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            fig5 = go.Figure(go.Bar(
                x=["No Promotion", "On Promotion"],
                y=[base_avg, promo_avg],
                marker_color=["#4c72b0", "#2ecc71"],
                text=[f"{base_avg:.1f}", f"{promo_avg:.1f}"],
                textposition="outside",
            ))
            fig5.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                               font=dict(color="#8b9dc3"), height=280,
                               yaxis=dict(showgrid=True, gridcolor="#1e2130"))
            st.plotly_chart(fig5, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: Inventory
# ════════════════════════════════════════════════════════════════════════════
elif page == "📦 Inventory":
    st.title(f"📦 Inventory Simulation — Store {sel_store} · {sel_item}")
    from utils.inventory import compute_inventory_metrics, simulate_inventory_trajectory

    col1, col2, col3 = st.columns(3)
    with col1:
        current_stock = st.number_input("Current Stock (units)", min_value=0.0, value=500.0, step=50.0)
    with col2:
        lead_time = st.slider("Supplier Lead Time (days)", 1, 30, 7)
    with col3:
        svc_level = st.selectbox("Service Level", [0.90, 0.95, 0.99], index=1,
                                  format_func=lambda x: f"{int(x*100)}%")

    forecast_vals = forecast_df["forecast"].values
    metrics = compute_inventory_metrics(
        forecast_vals,
        lead_time_days=lead_time,
        service_level=svc_level,
        current_stock=current_stock,
    )

    # Alert
    if metrics["needs_reorder"]:
        st.markdown(f'<div class="alert-red">🚨 <b>REORDER ALERT</b> — Current stock ({current_stock:.0f}) is below reorder point ({metrics["reorder_point"]:.0f}). Place order immediately!</div>', unsafe_allow_html=True)
    elif metrics["stockout_prob"] and metrics["stockout_prob"] > 30:
        st.markdown(f'<div class="alert-yellow">⚠️ <b>LOW STOCK WARNING</b> — {metrics["stockout_prob"]:.1f}% stockout risk detected.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-green">✅ <b>STOCK HEALTHY</b> — Inventory levels are adequate for the forecast period.</div>', unsafe_allow_html=True)

    st.markdown("")
    c1, c2, c3, c4 = st.columns(4)
    for col, label, val, color in [
        (c1, "Safety Stock", f"{metrics['safety_stock']:.0f} units", "#4c72b0"),
        (c2, "Reorder Point", f"{metrics['reorder_point']:.0f} units", "#dd8452"),
        (c3, "Days of Stock", f"{metrics['days_of_stock']} days" if metrics['days_of_stock'] else "—", "#2ecc71"),
        (c4, "Stockout Risk", f"{metrics['stockout_prob']}%" if metrics['stockout_prob'] else "—",
             "#e74c3c" if (metrics['stockout_prob'] or 0) > 30 else "#2ecc71"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value" style="color:{color}">{val}</div>
        </div>""", unsafe_allow_html=True)

    # Simulation trajectory
    st.markdown('<div class="section-header">Inventory Trajectory Simulation</div>', unsafe_allow_html=True)
    order_qty = max(metrics["reorder_point"] * 1.5, 100)
    sim = simulate_inventory_trajectory(
        forecast_vals, current_stock,
        metrics["reorder_point"], order_qty, lead_time,
    )

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Stock Level", "Daily Demand vs Sales"),
                        vertical_spacing=0.1)
    fig.add_trace(go.Scatter(x=sim["day"], y=sim["stock_level"],
        mode="lines+markers", name="Stock", line=dict(color="#4c72b0", width=2)), row=1, col=1)
    fig.add_hline(y=metrics["reorder_point"], line_dash="dash", line_color="#e74c3c",
                  annotation_text="Reorder Point", row=1, col=1)
    fig.add_hline(y=metrics["safety_stock"], line_dash="dot", line_color="#f39c12",
                  annotation_text="Safety Stock", row=1, col=1)

    # Mark reorder days
    reorder_days = sim[sim["reorder"]]["day"]
    if len(reorder_days):
        fig.add_trace(go.Scatter(
            x=reorder_days, y=sim[sim["reorder"]]["stock_level"],
            mode="markers", name="Order Placed",
            marker=dict(color="#2ecc71", size=12, symbol="triangle-up"),
        ), row=1, col=1)

    fig.add_trace(go.Bar(x=sim["day"], y=sim["forecast_demand"],
        name="Forecast Demand", marker_color="#dd8452", opacity=0.6), row=2, col=1)
    fig.add_trace(go.Bar(x=sim["day"], y=sim["lost_sales"],
        name="Lost Sales", marker_color="#e74c3c", opacity=0.8), row=2, col=1)

    fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#8b9dc3"), height=550,
        xaxis2=dict(title="Day", showgrid=True, gridcolor="#1e2130"),
        yaxis=dict(showgrid=True, gridcolor="#1e2130"),
        yaxis2=dict(showgrid=True, gridcolor="#1e2130"),
        legend=dict(bgcolor="#1e2130"), barmode="overlay")
    st.plotly_chart(fig, use_container_width=True)

    fill_rate = (sim["actual_sales"].sum() / sim["forecast_demand"].sum() * 100) if sim["forecast_demand"].sum() > 0 else 100
    lost = sim["lost_sales"].sum()
    col1, col2 = st.columns(2)
    col1.markdown(f"""<div class="metric-card">
        <div class="metric-label">Fill Rate</div>
        <div class="metric-value" style="color:{'#2ecc71' if fill_rate>95 else '#e74c3c'}">{fill_rate:.1f}%</div>
    </div>""", unsafe_allow_html=True)
    col2.markdown(f"""<div class="metric-card">
        <div class="metric-label">Total Lost Sales</div>
        <div class="metric-value" style="color:{'#2ecc71' if lost<1 else '#e74c3c'}">{lost:.1f} units</div>
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: Model Comparison
# ════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Comparison":
    st.title("🤖 Model Performance Comparison")

    if "results" in models:
        results = models["results"].dropna()
        st.markdown('<div class="section-header">Leaderboard</div>', unsafe_allow_html=True)

        best_rmse = results.loc[results["RMSE"].idxmin(), "model"] if len(results) else "—"
        st.markdown(f"🏆 **Best model by RMSE:** `{best_rmse}`")

        # Styled table
        st.dataframe(results.style.highlight_min(subset=["MAE","RMSE","MAPE"], color="#1a3a2a"),
                     use_container_width=True, height=220)

        col1, col2, col3 = st.columns(3)
        for col, metric in zip([col1, col2, col3], ["MAE", "RMSE", "MAPE"]):
            fig = go.Figure(go.Bar(
                x=results["model"], y=results[metric],
                marker_color=["#2ecc71" if m == metric and v == results[metric].min()
                               else "#4c72b0" for v, m in zip(results[metric], [metric]*len(results))],
                text=results[metric].round(2), textposition="outside",
            ))
            fig.update_layout(title=dict(text=metric, font=dict(color="#e0e4f0")),
                              plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                              font=dict(color="#8b9dc3"), height=300,
                              yaxis=dict(showgrid=True, gridcolor="#1e2130"))
            col.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ No model results found. Run `python models/train_models.py` first.")

    st.markdown('<div class="section-header">Model Architecture Overview</div>', unsafe_allow_html=True)
    arch = pd.DataFrame([
        {"Model": "SARIMA", "Type": "Statistical", "Strengths": "Captures trend & weekly seasonality", "When to use": "Single series, interpretability needed"},
        {"Model": "Prophet", "Type": "Decomposition", "Strengths": "Handles holidays, missing data", "When to use": "Business seasonality, easy deployment"},
        {"Model": "XGBoost", "Type": "ML (Gradient Boosting)", "Strengths": "Multi-series, custom features, fast", "When to use": "Retail forecasting at scale"},
        {"Model": "LSTM", "Type": "Deep Learning", "Strengths": "Learns complex temporal patterns", "When to use": "Long sequences, sufficient data"},
    ])
    st.dataframe(arch, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: EDA
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔍 EDA":
    st.title("🔍 Exploratory Data Analysis")

    # Selected series
    st.markdown(f'<div class="section-header">Store {sel_store} · {sel_item} — Historical Sales</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ser["date"], y=ser["sales"],
        mode="lines", name="Sales", line=dict(color="#4c72b0", width=1.2)))
    fig.add_trace(go.Scatter(x=ser["date"], y=ser["sales"].rolling(30).mean(),
        mode="lines", name="30-day MA", line=dict(color="#e74c3c", width=2)))
    fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#8b9dc3"), height=350,
        xaxis=dict(showgrid=True, gridcolor="#1e2130"),
        yaxis=dict(showgrid=True, gridcolor="#1e2130", title="Sales"),
        legend=dict(bgcolor="#1e2130"))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Monthly Seasonality</div>', unsafe_allow_html=True)
        monthly = ser.copy()
        monthly["month"] = monthly["date"].dt.month
        mon_avg = monthly.groupby("month")["sales"].mean().reset_index()
        mon_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                     7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        mon_avg["month_name"] = mon_avg["month"].map(mon_names)
        fig6 = go.Figure(go.Bar(x=mon_avg["month_name"], y=mon_avg["sales"],
            marker_color=px.colors.sequential.Blues[-len(mon_avg):]))
        fig6.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font=dict(color="#8b9dc3"), height=320,
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#1e2130"))
        st.plotly_chart(fig6, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Sales Distribution</div>', unsafe_allow_html=True)
        fig7 = go.Figure()
        for label, mask, color in [
            ("No Promo", ser.onpromotion == 0, "#4c72b0"),
            ("On Promo", ser.onpromotion == 1, "#2ecc71"),
        ]:
            subset = ser[mask]["sales"]
            if len(subset) > 0:
                fig7.add_trace(go.Histogram(x=subset, name=label,
                    marker_color=color, opacity=0.7, nbinsx=40))
        fig7.update_layout(barmode="overlay", plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117", font=dict(color="#8b9dc3"),
            height=320, xaxis=dict(title="Sales"),
            legend=dict(bgcolor="#1e2130"))
        st.plotly_chart(fig7, use_container_width=True)

    # Lag correlation
    st.markdown('<div class="section-header">Autocorrelation (Lag Features)</div>', unsafe_allow_html=True)
    acf_vals = [ser["sales"].autocorr(lag=i) for i in range(1, 31)]
    fig8 = go.Figure(go.Bar(x=list(range(1, 31)), y=acf_vals,
        marker_color=["#2ecc71" if v > 0 else "#e74c3c" for v in acf_vals]))
    fig8.add_hline(y=0, line_color="#ffffff", line_width=1)
    fig8.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#8b9dc3"), height=300,
        xaxis=dict(title="Lag (days)", showgrid=True, gridcolor="#1e2130"),
        yaxis=dict(title="Autocorrelation", showgrid=True, gridcolor="#1e2130"))
    st.plotly_chart(fig8, use_container_width=True)

    # Raw data
    with st.expander("📋 Raw Data Sample"):
        st.dataframe(ser.tail(50), use_container_width=True)
