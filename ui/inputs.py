import streamlit as st
from datetime import datetime, timedelta

from config import (
    EARLIEST_FROM_DATE,
    LATEST_FROM_DATE,
    EARLIEST_TO_DATE,
    LATEST_TO_DATE,
    NUMBER_DAYS_SEPARATION_BTWN_DATES,
)


def get_user_inputs():
    today = datetime.today().date()
    four_years = today + timedelta(days=4 * 365)

    with st.sidebar:
        st.header("Data mode")

        mode = st.radio(
            "Select data mode",
            ["Historical", "Forecast"],
            index=0,
            help="Work with past (CSV) data or future projections.",
        )

        st.markdown("---")
        
        # ---------- Historical ----------
        st.markdown("### Historical data inputs")
        hist_disabled = mode != "Historical"

        hist_from = st.date_input(
            "Historical from date",
            value=EARLIEST_FROM_DATE,
            min_value=EARLIEST_FROM_DATE,
            max_value=LATEST_FROM_DATE,
            disabled=hist_disabled,
            key="hist_from",
        )

        min_to_by_separation = hist_from + timedelta(days=NUMBER_DAYS_SEPARATION_BTWN_DATES)
        hist_to_min = max(EARLIEST_TO_DATE, min_to_by_separation)

        hist_to = st.date_input(
            "Historical to date",
            value=LATEST_TO_DATE,
            min_value=hist_to_min,
            max_value=LATEST_TO_DATE,
            disabled=hist_disabled,
            key="hist_to",
        )

        st.caption(
            "Source: data/BTCUSD_daily.csv NB Minimum "
            + str(NUMBER_DAYS_SEPARATION_BTWN_DATES)
            + " days between Historic from/to dates."
        )

        st.markdown("---")

        # ---------- Forecast ----------
        st.markdown("### Forecast data inputs")
        forecast_disabled = mode != "Forecast"

        forecast_start = st.date_input(
            "Forecast start date",
            value=today,
            min_value=today,
            disabled=forecast_disabled,
            key="forecast_start",
        )

        forecast_end = st.date_input(
            "Forecast end date",
            value=four_years,
            min_value=today,
            disabled=forecast_disabled,
            key="forecast_end",
        )

        fee_growth = st.slider(
            "Fee growth (%/yr)",
            min_value=0,
            max_value=100,
            value=5,
            disabled=forecast_disabled,
            key="fee_growth",
        )

        st.markdown("### Price")
        price_model = st.selectbox(
            "BTC price model",
            ["Spot/auto", "Custom", "Monte Carlo (GBM)"],
            index=0,
            disabled=(mode == "Historical"),
            key="price_model",
        )

        custom_price = st.number_input(
            "Custom BTC price (USD)",
            min_value=0.0,
            value=60000.0,
            step=100.0,
            disabled=(mode == "Historical" or price_model != "Custom"),
            key="custom_price",
        )

    # validation
    date_error = None
    if mode == "Historical":
        if hist_to < hist_from:
            date_error = (
                "Your historical **to** date is earlier than your **from** date. "
                "Please pick a later 'to' date."
            )
    else:
        if forecast_end < forecast_start:
            date_error = (
                "Your forecast **end** date is earlier than your **start** date. "
                "Please pick a later end date."
            )

    if date_error:
        st.error(date_error)
        st.stop()

    return {
        "mode": mode,
        "hist_from": hist_from,
        "hist_to": hist_to,
        "forecast_start": forecast_start,
        "forecast_end": forecast_end,
        "fee_growth": fee_growth,
        "price_model": price_model,
        "custom_price": custom_price,
    }
