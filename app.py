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
BLOCK_TIME = 588  # long-term avg
SATOSHIS = 100_000_000
HALVING = 210_000
FEE_API = "https://mempool.space/api/v1/blocks"
PRICE_API = "https://blockchain.info/ticker"
SESSION_STATE_DARK = "dark"
SESSION_STATE_LIGHT = "light"
THEME_KEY = "theme"

st.set_page_config(page_title="Future Bitcoin Calculator", layout="wide")
st.title("Future Bitcoin Calculator")
st.markdown("Forecast **mining rewards**, **fees**, **price**, and **network growth**.")

# ========================================
# HELPERS
# ========================================
@st.cache_data(ttl=3600)  # Cache 1 hour to respect rate limits
def get_current_price():
    """Fetch BTC/USD from Blockchain.info (FOSS, reliable since 2011)."""
    try:
        response = requests.get(PRICE_API, timeout=10)
        response.raise_for_status()
        data = response.json()
        usd_data = data.get("USD")
        if not usd_data:
            raise ValueError("No USD data in response")
        price = usd_data["last"]
        return int(price)
    except Exception as e:
        st.warning(f"Price fetch failed: {e} — using fallback")
        return 99321  # fallback

@st.cache_data(ttl=1800)
def get_avg_fee_btc(blocks=2016):
    try:
        data = requests.get(FEE_API, timeout=10).json()[:blocks]
        total_fees = sum(b["extras"]["totalFees"] for b in data)
        return total_fees / len(data) / SATOSHIS
    except Exception as e:
        st.warning(f"BTC fee fetch failed: {e}")
        return 0.061

def block_from_date(date):
    return int((date - GENESIS).total_seconds() / BLOCK_TIME)

def date_from_block(height):
    return GENESIS + timedelta(seconds=height * BLOCK_TIME)

def subsidy_at(h):
    return 50 / (2 ** (h // HALVING))

# ========================================
# SIDEBAR WITH DARK MODE TOGGLE
# ========================================
with st.sidebar:
    st.header("Forecast Settings")

    # --- Auto Date Range - Start date to today and End date in four years time. ---
    today = datetime.today().date()
    four_years_later = today + timedelta(days=4*365)

    # initialise theme in session
    if THEME_KEY not in st.session_state:
        st.session_state[THEME_KEY] = SESSION_STATE_LIGHT

    # make toggle reflect current state
    dark_mode = st.toggle(
        "Dark Mode",
        value=st.session_state.get("dark_mode", False),
        key="dark_mode"
    )

    # inputs
    start_date = st.date_input("Start Date", today)
    end_date   = st.date_input("End Date", four_years_later)
    hashrate_growth = st.slider("Hashrate Growth (%/yr)", 10, 150, 50) / 100
    fee_growth      = st.slider("Fee Growth (%/yr)", -20, 100, 20) / 100
    price_model = st.selectbox("Price Model", ["Stock-to-Flow", "Power-Law", "Custom"])
    if price_model == "Custom":
        custom_price = st.number_input("Custom Price (USD)", 50000, 5000000, 250000)
        
# ========================================
# APPLY THEME VIA CSS + PLOTLY TEMPLATE
# ========================================
if st.session_state.get("dark_mode", False):
    st.markdown(
        """
        <style>
        /* app background + text */
        .stApp, body {
            background-color: #0e1117 !important;
            color: #f7f7f7 !important;
        }

        /* catch all known sidebar wrappers */
        [data-testid="stSidebar"],
        section[data-testid="stSidebar"],
        div[data-testid="stSidebar"] > div:first-child {
            background-color: #141820 !important;
            color: #f7f7f7 !important;
        }

        /* text inside sidebar */
        [data-testid="stSidebar"] * {
            color: #f7f7f7 !important;
        }

        /* inputs in sidebar (boxes) */
        [data-testid="stSidebar"] .stTextInput > div > div input,
        [data-testid="stSidebar"] .stDateInput input,
        [data-testid="stSidebar"] .stNumberInput input,
        [data-testid="stSidebar"] .stSelectbox > div > div {
            background-color: #1b1f2a !important;
            color: #f7f7f7 !important;
            border-color: #1b1f2a !important;
        }

        /* slider track */
        [data-testid="stSidebar"] .stSlider > div > div {
            background-color: rgba(247, 247, 247, 0.15) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    plotly_template = "plotly_dark"
else:
    st.markdown(
        """
        <style>
        .stApp, body {
            background-color: #ffffff !important;
            color: #0e1117 !important;
        }

        /* light sidebar */
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
    plotly_template = "plotly_white"

# ========================================
# CALCULATIONS
# ========================================
start_dt = datetime.combine(start_date, datetime.min.time())
end_dt = datetime.combine(end_date, datetime.min.time())

h1 = block_from_date(start_dt)
h2 = block_from_date(end_dt)
blocks = h2 - h1 + 1

# Subsidy
era1, era2 = h1 // HALVING, h2 // HALVING
if era1 == era2:
    subsidy_btc = blocks * subsidy_at(h1)
else:
    # cross-era calculation (simple but OK for now)
    subsidy_btc = sum(subsidy_at(h) for h in range(h1, h2 + 1))

# Fees
current_fee = get_avg_fee_btc()
years = (end_dt - start_dt).days / 365.25
avg_fee_forecast = current_fee * ((1 + fee_growth) ** (years / 2))  # avg over period
fees_btc = blocks * avg_fee_forecast

total_btc = subsidy_btc + fees_btc
current_price = get_current_price()
usd_value = total_btc * current_price

# Price Forecast
dates = [start_dt + timedelta(days=x) for x in range(0, (end_dt - start_dt).days + 1, 30)]
blocks_list = [block_from_date(d) for d in dates]
price_forecast = []

for h in blocks_list:
    years_since_genesis = (date_from_block(h) - GENESIS).days / 365.25
    if price_model == "Stock-to-Flow":
        # (your rough S2F — kept as-is)
        sf = 1.0 * (h * 50) / (210000 * 50)
        price = 0.4 * np.exp(3.3 * np.log(sf + 1))
    elif price_model == "Power-Law":
        price = 0.00002 * (years_since_genesis ** 5.8)
    else:
        # custom model: use the custom price directly
        price = custom_price / current_price
    price_forecast.append(price * current_price)

# ========================================
# DISPLAY RESULTS
# ========================================
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total BTC Mined", f"{total_btc:,.1f}")
with col2:
    st.metric("Subsidy", f"{subsidy_btc:,.1f} BTC")
with col3:
    st.metric("Fees", f"{fees_btc:,.1f} BTC")

col4, col5 = st.columns(2)
with col4:
    st.metric("Avg Fee per Block", f"{avg_fee_forecast:.4f} BTC")
with col5:
    st.metric("Value at Current Price", f"${usd_value:,.0f}")

# ========================================
# CHARTS
# ========================================
df = pd.DataFrame({
    "Date": [d.date() for d in dates],
    "Price (USD)": price_forecast,
    "Subsidy": [subsidy_at(h) for h in blocks_list],
    "Est. Fee": [
        current_fee * (1 + fee_growth) ** ((date_from_block(h) - start_dt).days / 365.25)
        for h in blocks_list
    ],
})

fig = go.Figure()
fig.add_trace(go.Scatter(x=df["Date"], y=df["Price (USD)"], name="Price Forecast", line=dict(color="#f7931a")))
fig.add_trace(go.Bar(x=df["Date"], y=df["Subsidy"], name="Subsidy per Block", yaxis="y2"))
fig.add_trace(go.Bar(x=df["Date"], y=df["Est. Fee"], name="Est. Fee per Block", yaxis="y2", opacity=0.6))
fig.update_layout(
    title=f"Bitcoin Price & Revenue Forecast ({start_date} → {end_date})",
    yaxis=dict(title="Price (USD)"),
    yaxis2=dict(title="BTC per Block", overlaying="y", side="right"),
    hovermode="x unified",
    template=plotly_template,   # 👈 use the toggle here
)
st.plotly_chart(fig, width="stretch")

# ========================================
# DATA TABLE & EXPORT
# ========================================
st.subheader("Monthly Breakdown")
monthly = []
current = start_dt
while current <= end_dt:
    h = block_from_date(current)
    monthly.append({
        "Month": current.strftime("%Y-%m"),
        "Block": h,
        "Subsidy": subsidy_at(h),
        "Est. Fee": current_fee * (1 + fee_growth) ** ((current - start_dt).days / 365.25),
        "Price": price_forecast[min(len(price_forecast)-1, (current - start_dt).days // 30)],
    })
    current += timedelta(days=30)

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

# ---- FOOTER -------------------------------------------
st.markdown("---")
try:
    blockchain_price = get_current_price()
except:
    blockchain_price = None

price_str = f"${blockchain_price:,} USD" if blockchain_price else "Price: — (offline)"

col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    st.caption(f"**Data:** mempool.space • blockchain.info • **Price:** {price_str}")
with col2:
    st.caption("Block time: **588 s** • Halving: **210 000** blocks")
with col3:
    st.caption(f"Updated: **{datetime.now():%Y-%m-%d %H:%M}**")
# -------------------------------------------------------
