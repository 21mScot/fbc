import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import HISTORY_RATIO
from core.models import estimate_mu_sigma_from_history, monte_carlo_gbm


def build_price_chart(
    plot_df: pd.DataFrame,
    start_date,
    end_date,
    backcast: bool,
    price_model: str,
    hist_df: pd.DataFrame | None,
):
    fig = go.Figure()
    dark = st.session_state.get("dark_mode", False)
    template = "plotly_dark" if dark else "plotly_white"

    today_ts = pd.Timestamp.today().normalize()
    mc_df = None

    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)

    plot_df = plot_df.copy()
    plot_df["Date"] = pd.to_datetime(plot_df["Date"])

    plot_df_window = plot_df[(plot_df["Date"] >= start_ts) & (plot_df["Date"] <= end_ts)].copy()

    # trim forecast to HISTORY_RATIO if we’re backcasting
    plot_df_trim = plot_df_window.copy()
    history_mask = plot_df_trim["Actual Price"].notna()
    if backcast and history_mask.any():
        hist_start = plot_df_trim.loc[history_mask, "Date"].min()
        hist_end = plot_df_trim.loc[history_mask, "Date"].max()
        hist_days = max((hist_end - hist_start).days, 1)
        forecast_share = 1 - HISTORY_RATIO
        max_forecast_days = int(hist_days * forecast_share)
        forecast_cutoff = hist_end + pd.Timedelta(days=max_forecast_days)
        plot_df_trim = plot_df_trim[plot_df_trim["Date"] <= forecast_cutoff]

    if price_model == "Monte Carlo (GBM)":
        # actuals
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
        start_price = float(plot_df_trim["Model Price"].iloc[0])
        mc_df = monte_carlo_gbm(
            start_price=start_price,
            start_dt=start_ts,
            end_dt=end_ts,
            mu=daily_mu,
            sigma=daily_sigma,
            n_sims=500,
        )

        # median
        fig.add_trace(
            go.Scatter(
                x=mc_df["Date"],
                y=mc_df["P50"],
                name="MC P50",
                mode="lines",
                line=dict(width=2),
            )
        )
        # 25-75
        fig.add_trace(
            go.Scatter(
                x=pd.concat([mc_df["Date"], mc_df["Date"][::-1]]),
                y=pd.concat([mc_df["P25"], mc_df["P75"][::-1]]),
                fill="toself",
                name="MC 25-75%",
                line=dict(width=0),
                opacity=0.2,
            )
        )
        # 5-95
        fig.add_trace(
            go.Scatter(
                x=pd.concat([mc_df["Date"], mc_df["Date"][::-1]]),
                y=pd.concat([mc_df["P05"], mc_df["P95"][::-1]]),
                fill="toself",
                name="MC 5-95%",
                line=dict(width=0),
                opacity=0.1,
            )
        )
    else:
        # draw actuals
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


        # draw model
        model_part = plot_df_trim[plot_df_trim["Actual Price"].isna()]
        if not model_part.empty:
            fig.add_trace(
                go.Scatter(
                    x=model_part["Date"],
                    y=model_part["Model Price"],
                    name="Model Price",
                    mode="lines",
                    line=dict(color="#f7931a", width=2, dash="dash"),
                )
            )


    # today marker
    if start_ts <= today_ts <= end_ts:
        fig.add_vline(
            x=today_ts,
            line_width=1,
            line_dash="dot",
            line_color="gray",
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
        title="Bitcoin Price Actual / Forecast",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x unified",
        template=template,
        height=600,
        margin=dict(l=80, r=40, t=60, b=60),
        xaxis=dict(range=[start_ts, end_ts]),
    )

    return fig, mc_df
