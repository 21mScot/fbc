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

# how much of the chart should be history vs forecast
HISTORY_RATIO = 0.85   # show ~85% history and 15% forecast

st.set_page_config(page_title="Future Bitcoin Calculator", layout="wide")
st.title("Future Bitcoin Calculator")
st.markdown("Forecast **mining rewards**, **fees**, **price**, and **network growth**.")


# ========================================
# HELPERS
# ========================================
@st.cache_data
def get_historical_prices() -> pd.DataFrame:
    """Load historical BTC daily price CSV if available."""
    try:
        df = pd.read_csv("data/BTCUSD_daily.csv", skiprows=1)
        df["date"] = pd.to_datetime(df["Date"]).dt.date
        df = df[["date", "Close"]].rename(columns={"Close": "price"})
        df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
        return df
    except Exception:
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
    except Exception as e:
        st.warning(f"Price fetch failed: {e} — using fallback 99321")
        return 99321


def subsidy_at(h: int) -> float:
    return 50 / (2 ** (h // HALVING))


def block_from_date(date: datetime) -> int:
    return int((date - GENESIS).total_seconds() / BLOCK_TIME)


def date_from_block(height: int) -> datetime:
    return GENESIS + timedelta(seconds=height * BLOCK_TIME)


def power_law_price(days_since_genesis: int) -> float:
    # crude descriptive model — we’ll re-scale it later
    return 0.00002 * (days_since_genesis ** 5.8)


def build_price_series(start_date: datetime, end_date: datetime, backcast: bool) -> pd.DataFrame:
    """Monthly-ish model series from start → end."""
    genesis = GENESIS
    today = datetime.today().date()

    if backcast:
        series_start = min(start_date.date(), EARLIEST_DATA)
        series_start = datetime.combine(series_start, datetime.min.time())
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
        is_hist.append(cur.date() <= today)
        cur += timedelta(days=30)

    return pd.DataFrame({
        "Date": pd.to_datetime(dates),
        "Model Price": model_prices,
        "Historical": is_hist,
    })


def monte_carlo_gbm(start_price: float, start_dt: datetime, end_dt: datetime,
                    mu: float, sigma: float, n_sims: int = 500) -> pd.DataFrame:
    days = (end_dt - start_dt).days
    if days <= 0:
        days = 1

    dates = [start_dt + timedelta(days=i) for i in range(days + 1)]
    dt = 1 / 365

    sims = np.zeros((len(dates), n_sims), dtype=float)
    sims[0, :] = start_price

    for t in range(1, len(dates)):
        z = np.random.normal(size=n_sims)
        sims[t, :] = sims[t - 1, :] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)

    p5 = np.percentile(sims, 5, axis=1)
    p25 = np.percentile(sims, 25, axis=1)
    p50 = np.percentile(sims, 50, axis=1)
    p75 = np.percentile(sims, 75, axis=1)
    p95 = np.percentile(sims, 95, axis=1)

    return pd.DataFrame({
        "Date": dates,
        "P05": p5,
        "P25": p25,
        "P50": p50,
        "P75": p75,
        "P95": p95,
    })


def estimate_mu_sigma_from_history(hist_df: pd.DataFrame) -> tuple[float, float]:
    if hist_df is None or hist_df.empty:
        annual_mu = 0.35
        annual_sigma = 0.75
        daily_mu = (1 + annual_mu) ** (1 / 365) - 1
        daily_sigma = annual_sigma / np.sqrt(252)
        return daily_mu, daily_sigma

    prices = hist_df["price"].astype(float).values
    returns = np.diff(np.log(prices))
    daily_mu = returns.mean()
    daily_sigma = returns.std()
    return daily_mu, daily_sigma


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

    price_model = st.selectbox(
        "Price Model",
        [
            "Power-Law",
            "Stock-to-Flow (placeholder)",
            "Monte Carlo (GBM)",
            "Custom",
        ],
        index=0,
    )
    if price_model == "Custom":
        custom_price = st.number_input("Custom Price (USD)", 50_000, 5_000_000, 250_000)

# ========================================
# THEME (DARK / LIGHT) VIA CSS
# ========================================
if st.session_state.get("dark_mode", False):
    st.markdown(
        """
        <style>
        .stApp, body {
            background-color: #0e1117 !important;
            color: #f7f7f7 !important;
        }
        [data-testid="stSidebar"],
        section[data-testid="stSidebar"],
        div[data-testid="stSidebar"] > div:first-child {
            background-color: #141820 !important;
            color: #f7f7f7 !important;
        }
        [data-testid="stSidebar"] * {
            color: #f7f7f7 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
        .stApp, body {
            background-color: #ffffff !important;
            color: #0e1117 !important;
        }
        [data-testid="stSidebar"],
        section[data-testid="stSidebar"],
        div[data-testid="stSidebar"] > div:first-child {
            background-color: #f7f7f7 !important;
            color: #0e1117 !important;
        }
        [data-testid="stSidebar"] * {
            color: #0e1117 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ========================================
# CALCULATIONS (mining side)
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

# ========================================
# BUILD BASE PRICE SERIES
# ========================================
model_df = build_price_series(start_dt, end_dt, backcast=backcast)
hist_df = get_historical_prices()

if backcast and not hist_df.empty:
    hist_df["date"] = pd.to_datetime(hist_df["date"])
    plot_df = model_df.merge(hist_df, left_on="Date", right_on="date", how="left")
    plot_df["Actual Price"] = plot_df["price"]
else:
    plot_df = model_df.copy()
    plot_df["Actual Price"] = np.nan

# re-scale model to last actual price to keep y-axis meaningful
if backcast and not hist_df.empty and "price" in hist_df.columns:
    try:
        last_actual_date = hist_df["date"].max()
        last_actual_price = float(
            hist_df.loc[hist_df["date"] == last_actual_date, "price"].iloc[0]
        )
        join_mask = plot_df["Date"] >= pd.to_datetime(last_actual_date)
        if join_mask.any():
            model_at_join = float(plot_df.loc[join_mask, "Model Price"].iloc[0])
            if model_at_join > 0:
                scale_factor = last_actual_price / model_at_join
                plot_df["Model Price"] = plot_df["Model Price"] * scale_factor
    except Exception as e:
        st.warning(f"Model scaling skipped ({e})")

# custom model
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
# PRICE CHART (with HISTORY_RATIO trim)
# ========================================
fig = go.Figure()
dark = st.session_state.get("dark_mode", False)
template = "plotly_dark" if dark else "plotly_white"

today_ts = pd.Timestamp.today().normalize()  # real timestamp
mc_df = None

# make a trimmed copy just for the chart
plot_df_trim = plot_df.copy()
forecast_cutoff = None

history_mask = plot_df_trim["Actual Price"].notna()
if backcast and history_mask.any():
    hist_start = plot_df_trim.loc[history_mask, "Date"].min()
    hist_end = plot_df_trim.loc[history_mask, "Date"].max()
    hist_days = max((hist_end - hist_start).days, 1)
    forecast_share = 1 - HISTORY_RATIO  # e.g. 0.15
    max_forecast_days = int(hist_days * forecast_share)
    forecast_cutoff = hist_end + pd.Timedelta(days=max_forecast_days)
    plot_df_trim = plot_df_trim[plot_df_trim["Date"] <= forecast_cutoff]

if price_model == "Monte Carlo (GBM)":
    # actuals solid
    hist_part = plot_df_trim[plot_df_trim["Actual Price"].notna()]
    if not hist_part.empty:
        fig.add_trace(
            go.Scatter(
                x=hist_part["Date"],
                y=hist_part["Actual Price"],
                name="Actual Price",
                mode="lines",
                line=dict(color="#f7931a", width=2),
            )
        )

    daily_mu, daily_sigma = estimate_mu_sigma_from_history(hist_df)
    mc_df = monte_carlo_gbm(
        start_price=current_price,
        start_dt=start_dt,
        end_dt=end_dt,
        mu=daily_mu,
        sigma=daily_sigma,
        n_sims=500,
    )

    # trim MC as well
    if forecast_cutoff is not None:
        mc_df = mc_df[pd.to_datetime(mc_df["Date"]) <= forecast_cutoff]

    fig.add_trace(go.Scatter(x=mc_df["Date"], y=mc_df["P95"], line=dict(width=0), showlegend=False))
    fig.add_trace(
        go.Scatter(
            x=mc_df["Date"],
            y=mc_df["P05"],
            fill="tonexty",
            name="90% band",
            opacity=0.15,
            line=dict(width=0),
        )
    )
    fig.add_trace(go.Scatter(x=mc_df["Date"], y=mc_df["P75"], line=dict(width=0), showlegend=False))
    fig.add_trace(
        go.Scatter(
            x=mc_df["Date"],
            y=mc_df["P25"],
            fill="tonexty",
            name="50% band",
            opacity=0.25,
            line=dict(width=0),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=mc_df["Date"],
            y=mc_df["P50"],
            name="Forecast (MC median)",
            mode="lines",
            line=dict(color="#f7931a", width=2, dash="dash"),
        )
    )

else:
    # non-MC: actuals solid
    hist_part = plot_df_trim[plot_df_trim["Actual Price"].notna()]
    if not hist_part.empty:
        fig.add_trace(
            go.Scatter(
                x=hist_part["Date"],
                y=hist_part["Actual Price"],
                name="Actual Price",
                mode="lines",
                line=dict(color="#f7931a", width=2),
            )
        )

    # forecast dashed
    future_mask = plot_df_trim["Date"].dt.normalize() > today_ts
    future_part = plot_df_trim[future_mask]
    if not future_part.empty:
        fig.add_trace(
            go.Scatter(
                x=future_part["Date"],
                y=future_part["Model Price"],
                name="Forecast",
                mode="lines",
                line=dict(color="#f7931a", width=2, dash="dash"),
            )
        )

# ✅ TODAY LINE — use shape + annotation (no add_vline → no summing error)
fig.add_shape(
    type="line",
    x0=today_ts,
    x1=today_ts,
    y0=0,
    y1=1,
    xref="x",
    yref="paper",
    line=dict(color="gray", dash="dot"),
)
fig.add_annotation(
    x=today_ts,
    y=1,
    xref="x",
    yref="paper",
    text="Today",
    showarrow=False,
    yshift=10,
    font=dict(color="gray"),
)

fig.update_layout(
    title="Bitcoin Price Forecast",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    hovermode="x unified",
    template=template,
    height=600,
    margin=dict(l=80, r=40, t=60, b=60),
)

# new API
st.plotly_chart(fig, width="stretch")

# ========================================
# MONTHLY TABLE (full horizon)
# ========================================
st.subheader("Monthly Breakdown")
monthly = []
cur = start_dt
price_idx = 0
while cur <= end_dt:
    h = block_from_date(cur)

    if price_model == "Monte Carlo (GBM)" and mc_df is not None:
        mc_row = mc_df[pd.to_datetime(mc_df["Date"]) == cur]
        if not mc_row.empty:
            price_val = float(mc_row["P50"].values[0])
        else:
            price_val = float(mc_df["P50"].iloc[-1])
    else:
        if price_idx < len(plot_df):
            price_val = float(plot_df.iloc[price_idx]["Model Price"])
        else:
            price_val = float(plot_df.iloc[-1]["Model Price"])

    monthly.append({
        "Month": cur.strftime("%Y-%m"),
        "Block": h,
        "Subsidy": subsidy_at(h),
        "Est. Fee": current_fee * (1 + fee_growth) ** ((cur - start_dt).days / 365.25),
        "Price": price_val,
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
    width="stretch",  # <- replaced use_container_width here too
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
