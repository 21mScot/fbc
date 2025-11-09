# app_main.py
import streamlit as st
from datetime import datetime

from core.reports import build_monthly_table
from core.services.mining import compute_mining_economics
from core.services.pricing import build_price_data
from ui.inputs import get_user_inputs
from ui.charts import build_price_chart


def _get_active_dates(mode, hist_from, hist_to, forecast_start, forecast_end):
    if mode == "Forecast":
        active_start_date = forecast_start
        active_end_date = forecast_end
    else:
        active_start_date = hist_from
        active_end_date = hist_to

    start_dt = datetime.combine(active_start_date, datetime.min.time())
    end_dt = datetime.combine(active_end_date, datetime.min.time())
    backcast = mode == "Historical"
    return active_start_date, active_end_date, start_dt, end_dt, backcast


def _render_metrics(mining_ctx: dict):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total BTC Mined", f"{mining_ctx['total_btc']:,.1f}")
    with c2:
        st.metric("Subsidy", f"{mining_ctx['subsidy_btc']:,.1f} BTC")
    with c3:
        st.metric("Fees", f"{mining_ctx['fees_btc']:,.1f} BTC")

    c4, c5 = st.columns(2)
    with c4:
        st.metric("Avg Fee per Block", f"{mining_ctx['avg_fee_forecast']:.4f} BTC")
    with c5:
        st.metric(
            "Value at Current Price",
            f"${mining_ctx['usd_value']:,.0f}",
        )


def _render_footer(current_price: float):
    st.markdown("---")
    price_str = f"${current_price:,} USD"
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.caption(
            f"**Data:** mempool.space • blockchain.info • **Price:** {price_str}"
        )
    with col2:
        st.caption("Block time: **588 s** • Halving: **210 000** blocks")
    with col3:
        st.caption(f"Updated: **{datetime.now():%Y-%m-%d %H:%M}**")


def run_app():
    st.title("Future Bitcoin Calculator")
    st.markdown(
        "Forecast **mining rewards**, **fees**, **price**, and **network growth**."
    )

    # 1) inputs
    inputs = get_user_inputs()

    (
        active_start_date,
        active_end_date,
        start_dt,
        end_dt,
        backcast,
    ) = _get_active_dates(
        inputs["mode"],
        inputs["hist_from"],
        inputs["hist_to"],
        inputs["forecast_start"],
        inputs["forecast_end"],
    )

    # 2) mining (service)
    mining_ctx = compute_mining_economics(
        active_start_date,
        active_end_date,
        inputs["fee_growth"],
    )

    # 3) pricing/model/historical (service)
    plot_df, hist_df = build_price_data(
        start_dt=start_dt,
        end_dt=end_dt,
        backcast=backcast,
        price_model=inputs["price_model"],
        custom_price=inputs["custom_price"],
    )

    # 4) metrics
    _render_metrics(mining_ctx)

    # 5) chart (first pass) — get mc_df ONLY, don't render
    _, mc_df = build_price_chart(
        plot_df=plot_df,
        start_date=start_dt,
        end_date=end_dt,
        backcast=backcast,
        price_model=inputs["price_model"],
        hist_df=hist_df,
    )

    # 6) monthly breakdown (now we have mc_df)
    df_monthly = build_monthly_table(
        start_dt=start_dt,
        end_dt=end_dt,
        plot_df=plot_df,
        price_model=inputs["price_model"],
        mc_df=mc_df,
        current_fee=mining_ctx["current_fee"],
        fee_growth=inputs["fee_growth"],
    )
    st.session_state["monthly_breakdown"] = df_monthly

    # 7) chart (second pass) — now render, with key
    final_fig, _ = build_price_chart(
        plot_df=plot_df,
        start_date=start_dt,
        end_date=end_dt,
        backcast=backcast,
        price_model=inputs["price_model"],
        hist_df=hist_df,
    )
    st.plotly_chart(final_fig, width="stretch", key="price_and_histogram")

    # 8) table
    st.subheader("Monthly Breakdown")
    st.dataframe(df_monthly, hide_index=True)

    csv = df_monthly.to_csv(index=False).encode()
    st.download_button(
        "Download CSV",
        csv,
        "future_bitcoin_forecast.csv",
        "text/csv",
    )

    # 9) footer
    _render_footer(mining_ctx["current_price"])
