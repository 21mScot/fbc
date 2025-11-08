# app.py — Future Bitcoin Calculator (Streamlit)
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# ========================================
# CONFIG & CONSTANTS
# ========================================
GENESIS = datetime(2009, 1, 3, 18, 15, 5)
BLOCK_TIME = 588
SATOSHIS = 100_000_000
HALVING = 210_000
FEE_API = "https://mempool.space/api/v1/blocks"
PRICE_API = "https://blockchain.info/ticker"

EARLIEST_DATA = datetime(2017, 8, 17).date()

st.set_page_config(page_title="Future Bitcoin Calculator", layout="wide")
st.title("Future Bitcoin Calculator")
st.markdown("Forecast **mining rewards**, **fees**, **price**, and **network growth**.")

# ========================================
# HELPERS
# ========================================
@st.cache_data
def get_historical_prices() -> pd.DataFrame:
    try:
        df = pd.read_csv("data/BTCUSD_daily.csv", skiprows=1)
        df["date"] = pd.to_datetime(df["Date"]).dt.date
        df = df[["date", "Close"]].rename(columns={"Close": "price"})
        df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Failed to load historical CSV: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=1800)
def get_avg_fee_btc(blocks: int = 2016) -> float:
    try:
        data = requests.get(FEE_API, timeout=10).json()[:blocks]
        total_fees = sum(b["extras"]["totalFees"] for b in data)
        return total_fees / len(data) / SATOSHIS
    except Exception as e:
        st.warning(f"Fee API failed ({e}) → using fallback 0.061 BTC")
        return 0.061


@st.cache_data(ttl=3600)
def get_current_price() -> int:
    try:
        data = requests.get(PRICE_API, timeout=10).json()
        return int(data["USD"]["last"])
    except Exception:
        return 104_000


def block_from_date(dt: datetime) -> int:
    return int((dt - GENESIS).total_seconds() / BLOCK_TIME)


def date_from_block(height: int) -> datetime:
    return GENESIS + timedelta(seconds=height * BLOCK_TIME)


def subsidy_at(height: int) -> float:
    return 50 / (2 ** (height // HALVING))


def power_law_price(days_since_genesis: float) -> float:
    return 1.5e-8 * (days_since_genesis ** 3.3)


def get_price_forecast_with_backcast(
    start_date: datetime.date,
    end_date: datetime.date,
    backcast: bool,
) -> pd.DataFrame:
    today = datetime.today().date()
    genesis = GENESIS.date()

    if backcast:
        series_start = max(EARLIEST_DATA, today - timedelta(days=8 * 365))
    else:
        series_start = start_date

    dates = []
    model_prices = []
    is_hist = []

    cur = series_start
    while cur <= end_date:
        days = (cur - genesis).days
        if days < 365:
            cur += timedelta(days=30)
            continue

        price = power_law_price(days)
        dates.append(cur)
        model_prices.append(price)
        is_hist.append(cur <= today)
        cur += timedelta(days=30)

    return pd.DataFrame({
        "Date": pd.to_datetime(dates),  # ← Convert to datetime for Plotly
        "Model Price": model_prices,
        "Historical": is_hist,
    })


# ========================================
# SIDEBAR
# ========================================
with st.sidebar:
    st.header("Forecast Settings")
    st.toggle("Dark Mode", value=False, key="dark_mode")

    backcast = st.checkbox(
        "Show Backcast (max 8 years, from 2017-08-17)",
        value=True,
    )

    today = datetime.today().date()
    four_years = today + timedelta(days=4 * 365)

    start_date = st.date_input("Start Date", value=today, min_value=today)
    end_date = st.date_input("End Date", value=four_years, min_value=today)

    hashrate_growth = st.slider("Hashrate Growth (%/yr)", 10, 150, 50) / 100
    fee_growth = st.slider("Fee Growth (%/yr)", -20, 100, 20) / 100

    price_model = st.selectbox("Price Model", ["Power-Law", "Stock-to-Flow", "Custom"], index=0)
    if price_model == "Custom":
        custom_price = st.number_input("Custom Price (USD)", 50_000, 5_000_000, 250_000)

# ========================================
# CALCULATIONS
# ========================================
start_dt = datetime.combine(start_date, datetime.min.time())
end_dt = datetime.combine(end_date, datetime.min.time())

h1 = block_from_date(start_dt)
h2 = block_from_date(end_dt)
blocks = h2 - h1 + 1

era1, era2 = h1 // HALVING, h2 // HALVING
if era1 == era2:
    subsidy_btc = blocks * subsidy_at(h1)
else:
    subsidy_btc = sum(subsidy_at(h) for h in range(h1, h2 + 1))

current_fee = get_avg_fee_btc()
years = (end_dt - start_dt).days / 365.25
avg_fee_forecast = current_fee * ((1 + fee_growth) ** (years / 2))
fees_btc = blocks * avg_fee_forecast

total_btc = subsidy_btc + fees_btc
current_price = get_current_price()
usd_value = total_btc * current_price

model_df = get_price_forecast_with_backcast(start_date, end_date, backcast)

hist_df = get_historical_prices()
if not hist_df.empty:
    hist_df["date"] = pd.to_datetime(hist_df["date"])  # ← Ensure datetime
    plot_df = model_df.merge(hist_df, left_on="Date", right_on="date", how="left")
    plot_df["Actual Price"] = plot_df["price"]
else:
    plot_df = model_df.copy()
    plot_df["Actual Price"] = np.nan

if price_model == "Custom":
    first_model = plot_df["Model Price"].iloc[0]
    plot_df["Model Price"] = plot_df["Model Price"] * (custom_price / first_model)

# ========================================
# DISPLAY METRICS
# ========================================
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Total BTC Mined", f"{total_btc:,.1f}")
with c2:
    st.metric("Subsidy", f"{subsidy_btc:,.1f} BTC")
with c3:
    st.metric("Fees", f"{fees_btc:,.1f} BTC")

c4, c5 = st.columns(2)
with c4:
    st.metric("Avg Fee per Block", f"{avg_fee_forecast:.4f} BTC")
with c5:
    st.metric("Value at Current Price", f"${usd_value:,.0f}")

# ========================================
# PRICE CHART (Backcast + Forecast + Model)
# ========================================
fig = go.Figure()

# ---- 1. ACTUAL HISTORICAL (ORANGE, SOLID) ----
hist = plot_df[plot_df["Historical"] & plot_df["Actual Price"].notna()]
if not hist.empty:
    fig.add_trace(
        go.Scatter(
            x=hist["Date"],
            y=hist["Actual Price"],
            mode="lines",
            name="Actual Price (Backcast)",
            line=dict(color="#ff7f0e", width=2.5),  # Orange, solid
        )
    )

# ---- 2. POWER-LAW MODEL (GREEN, DOTTED) ----
fig.add_trace(
    go.Scatter(
        x=plot_df["Date"],
        y=plot_df["Model Price"],
        mode="lines",
        name="Power-Law Model",
        line=dict(color="#2ca02c", width=2, dash="dot"),  # Green, dotted
    )
)

# ---- 3. FORECAST (BLUE, DASHED) ----
# Only show forecast part (from today onward)
forecast = plot_df[~plot_df["Historical"]]
if not forecast.empty:
    fig.add_trace(
        go.Scatter(
            x=forecast["Date"],
            y=forecast["Model Price"],
            mode="lines",
            name="Forecast (Future)",
            line=dict(color="#1f77b4", width=2.5, dash="dash"),  # Blue, dashed
        )
    )

# ---- 4. TODAY VERTICAL LINE ----
today_dt = pd.Timestamp.today().normalize()
if today_dt in plot_df["Date"].values:
    fig.add_vline(
        x=today_dt,
        line=dict(color="gray", dash="dash"),
        annotation_text="Today",
        annotation_position="top left",
    )

# ---- 5. LAYOUT ----
fig.update_layout(
    title="Bitcoin Price: Backcast (Actual) • Model • Forecast",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    hovermode="x unified",
    template="plotly_dark" if st.session_state.get("dark_mode", False) else "plotly_white",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    showlegend=True,
)

st.plotly_chart(fig, width="stretch")

# ========================================
# MONTHLY TABLE
# ========================================
st.subheader("Monthly Breakdown")
monthly = []
cur = start_dt
price_idx = 0
while cur <= end_dt:
    h = block_from_date(cur)
    price = plot_df.iloc[min(price_idx, len(plot_df)-1)]["Model Price"]
    monthly.append({
        "Month": cur.strftime("%Y-%m"),
        "Block": h,
        "Subsidy": subsidy_at(h),
        "Est. Fee": current_fee * (1 + fee_growth) ** ((cur - start_dt).days / 365.25),
        "Price": price,
    })
    cur += timedelta(days=30)
    price_idx += 1

df_monthly = pd.DataFrame(monthly)
st.dataframe(
    df_monthly.style.format({
        "Subsidy": "{:.6f}",
        "Est. Fee": "{:.6f}",
        "Price": "${:,.0f}",
    }),
    width="stretch",
)

csv = df_monthly.to_csv(index=False).encode()
st.download_button("Download CSV", csv, "future_bitcoin_forecast.csv", "text/csv")

# ========================================
# FOOTER
# ========================================
st.markdown("---")
price_str = f"${current_price:,} USD"
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    st.caption(f"**Data:** mempool.space • blockchain.info • **Price:** {price_str}")
with col2:
    st.caption("Block time: **588 s** • Halving: **210 000** blocks")
with col3:
    st.caption(f"Updated: **{datetime.now():%Y-%m-%d %H:%M}**")