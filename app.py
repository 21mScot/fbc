# app.py — Future Bitcoin Calculator (refactored)
import streamlit as st
import pandas as pd
from datetime import datetime

from config import HISTORY_RATIO
from core.blockchain import block_from_date, subsidy_at
from core.data_sources import get_historical_prices, get_avg_fee_btc, get_current_price
from core.models import build_price_series
from core.reports import build_monthly_table
from ui.inputs import get_user_inputs
from ui.charts import build_price_chart


st.set_page_config(page_title="Future Bitcoin Calculator", layout="wide")
st.title("Future Bitcoin Calculator")
st.markdown("Forecast **mining rewards**, **fees**, **price**, and **network growth**.")

# 1) sidebar
inputs = get_user_inputs()
mode = inputs["mode"]
hist_from = inputs["hist_from"]
hist_to = inputs["hist_to"]
forecast_start = inputs["forecast_start"]
forecast_end = inputs["forecast_end"]
fee_growth = inputs["fee_growth"]
price_model = inputs["price_model"]
custom_price = inputs["custom_price"]

# 2) active dates
if mode == "Forecast":
    active_start_date = forecast_start
    active_end_date = forecast_end
else:
    active_start_date = hist_from
    active_end_date = hist_to

start_dt = datetime.combine(active_start_date, datetime.min.time())
end_dt = datetime.combine(active_end_date, datetime.min.time())
backcast = (mode == "Historical")

# 3) mining side
h1 = block_from_date(active_start_date)
h2 = block_from_date(active_end_date)
blocks = h2 - h1 + 1
era1, era2 = h1 // 210_000, h2 // 210_000
if era1 == era2:
    subsidy_btc = blocks * subsidy_at(h1)
else:
    subsidy_btc = sum(subsidy_at(h) for h in range(h1, h2 + 1))

current_fee = get_avg_fee_btc()
years = (active_start_date - active_end_date).days / 365.25
avg_fee_forecast = current_fee * ((1 + fee_growth) ** (years / 2))
fees_btc = blocks * avg_fee_forecast
total_btc = subsidy_btc + fees_btc

current_price = get_current_price()
usd_value = total_btc * current_price

# 4) price series
model_df = build_price_series(start_dt, end_dt, backcast=backcast)
hist_df = get_historical_prices()

if backcast and not hist_df.empty:
    hist_df["date"] = pd.to_datetime(hist_df["date"])
    plot_df = model_df.merge(hist_df, left_on="Date", right_on="date", how="left")
    plot_df["Actual Price"] = plot_df["price"]
else:
    plot_df = model_df.copy()
    plot_df["Actual Price"] = pd.NA

# re-scale model to last actual
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

# custom price override
if price_model == "Custom":
    first_model = plot_df["Model Price"].iloc[0]
    plot_df["Model Price"] = plot_df["Model Price"] * (custom_price / first_model)

# 5) metrics
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

# 6) chart (first pass - get mc_df)
fig, mc_df = build_price_chart(
    plot_df=plot_df,
    start_date=start_dt,
    end_date=end_dt,
    backcast=backcast,
    price_model=price_model,
    hist_df=hist_df,
)

# 7) monthly breakdown (needs mc_df)
df_monthly = build_monthly_table(
    start_dt=start_dt,
    end_dt=end_dt,
    plot_df=plot_df,
    price_model=price_model,
    mc_df=mc_df,
    current_fee=current_fee,
    fee_growth=fee_growth,
)

# make it available to charts.py
st.session_state["monthly_breakdown"] = df_monthly

# 6b) chart (second pass - now histogram is possible)
fig, _ = build_price_chart(
    plot_df=plot_df,
    start_date=start_dt,
    end_date=end_dt,
    backcast=backcast,
    price_model=price_model,
    hist_df=hist_df,
)

st.plotly_chart(fig, width="stretch")

# 8) show table
st.subheader("Monthly Breakdown")
st.dataframe(
    df_monthly.style.format(
        {
            "Subsidy": "{:.6f}",
            "Est. Fee": "{:.6f}",
            "Price": "${:,.0f}",
        }
    ),
    width="stretch",
)

csv = df_monthly.to_csv(index=False).encode()
st.download_button("Download CSV", csv, "future_bitcoin_forecast.csv", "text/csv")


# 8) footer
st.markdown("---")
price_str = f"${current_price:,} USD"
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    st.caption(f"**Data:** mempool.space • blockchain.info • **Price:** {price_str}")
with col2:
    st.caption("Block time: **588 s** • Halving: **210 000** blocks")
with col3:
    st.caption(f"Updated: **{datetime.now():%Y-%m-%d %H:%M}**")
