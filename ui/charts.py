# ui/charts.py

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from config import (
    HISTORY_RATIO,
    HISTOGRAM_HEIGHT_RATIO,
    HISTOGRAM_COLOR,
    HISTOGRAM_OPACITY,
    ACTUAL_PRICE_LEGENDRANK,
    MODEL_PRICE_LEGENDRANK,
    HISTOGRAM_LEGENDRANK,
)
from core.models import estimate_mu_sigma_from_history, monte_carlo_gbm


# ---------- helpers ----------

def _derive_blocks_from_monthly(mb: pd.DataFrame) -> pd.DataFrame | None:
    if mb is None or mb.empty:
        return None
    if "Month" not in mb.columns or "Block" not in mb.columns:
        return None

    mb = mb.copy()
    mb["Date"] = pd.to_datetime(mb["Month"])
    mb = mb.sort_values("Date")

    mb["Block"] = (
        mb["Block"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
    )
    mb["Block"] = pd.to_numeric(mb["Block"], errors="coerce")

    mb["Blocks (derived)"] = mb["Block"].diff()
    if len(mb):
        mb.loc[mb.index[0], "Blocks (derived)"] = float("nan")

    out = mb[["Date", "Blocks (derived)"]]
    return out if not out.empty else None


def _monthly_from_session() -> tuple[str | None, pd.DataFrame | None]:
    mb = st.session_state.get("monthly_breakdown")
    if not isinstance(mb, pd.DataFrame):
        return None, None
    if "Month" not in mb.columns:
        return None, None

    mb = mb.copy()
    mb["Date"] = pd.to_datetime(mb["Month"])

    # best: subsidy (shows halvings)
    if "Subsidy" in mb.columns:
        mb["Subsidy"] = pd.to_numeric(mb["Subsidy"], errors="coerce")
        out = mb[["Date", "Subsidy"]].sort_values("Date")
        if not out.empty:
            return "Subsidy (BTC)", out

    # fallback: blocks derived
    out = _derive_blocks_from_monthly(mb)
    if out is not None:
        return "Blocks (derived)", out

    return None, None


def _monthly_from_df(df: pd.DataFrame, candidates: list[str]) -> tuple[str | None, pd.DataFrame | None]:
    if df is None or df.empty:
        return None, None

    df = df.copy()

    # find a date-like column
    date_col = None
    for col in df.columns:
        if col.lower() in ("date", "dt", "timestamp"):
            date_col = col
            break
    if date_col is None:
        return None, None

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    if df.empty:
        return None, None

    df = df.set_index(date_col)

    for col in candidates:
        if col in df.columns:
            monthly = df[col].resample("M").sum().reset_index()
            if not monthly.empty:
                monthly = monthly.rename(columns={date_col: "Date"})
                return col, monthly

    return None, None


def _get_histogram_series(plot_df: pd.DataFrame, hist_df: pd.DataFrame | None):
    # 1) session
    name, series = _monthly_from_session()
    if name and series is not None and not series.empty:
        return name, series

    # 2) plot df
    name, series = _monthly_from_df(plot_df, ["Blocks", "block_count", "blocks", "Blocks Mined"])
    if name and series is not None and not series.empty:
        return name, series

    # 3) hist df
    if hist_df is not None and not hist_df.empty:
        name, series = _monthly_from_df(hist_df, ["Blocks", "block_count", "blocks", "Blocks Mined"])
        if name and series is not None and not series.empty:
            return name, series

    return None, None


# ---------- main entrypoint ----------

def build_price_chart(
    plot_df: pd.DataFrame,
    start_date,
    end_date,
    backcast: bool,
    price_model: str,
    hist_df: pd.DataFrame | None,
):
    dark = st.session_state.get("dark_mode", False)
    template = "plotly_dark" if dark else "plotly_white"

    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    today_ts = pd.Timestamp.today().normalize()
    mc_df = None

    # base df
    plot_df = plot_df.copy()
    plot_df["Date"] = pd.to_datetime(plot_df["Date"], errors="coerce")
    plot_df = plot_df.dropna(subset=["Date"])
    plot_df = plot_df.sort_values("Date")

    # window
    window_df = plot_df[(plot_df["Date"] >= start_ts) & (plot_df["Date"] <= end_ts)].copy()

    # trim forecast
    if backcast:
        hist_mask = window_df["Actual Price"].notna()
        if hist_mask.any():
            hist_start = window_df.loc[hist_mask, "Date"].min()
            hist_end = window_df.loc[hist_mask, "Date"].max()
            hist_days = max((hist_end - hist_start).days, 1)
            max_forecast_days = int(hist_days * (1 - HISTORY_RATIO))
            cutoff = hist_end + pd.Timedelta(days=max_forecast_days)
            window_df = window_df[window_df["Date"] <= cutoff]

    # prepare price dfs ONCE
    actual = window_df[window_df["Actual Price"].notna()]
    model = window_df[window_df["Actual Price"].isna()]

    # get histogram series (may be None)
    hist_name, hist_df_monthly = _get_histogram_series(window_df, hist_df)

    # figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 1) draw histogram FIRST (so lines overlay)
    max_bar_val = None
    if (
        hist_name is not None
        and hist_df_monthly is not None
        and not hist_df_monthly.empty
    ):
        hist_df_monthly = hist_df_monthly[
            (hist_df_monthly["Date"] >= start_ts) & (hist_df_monthly["Date"] <= end_ts)
        ]
        if not hist_df_monthly.empty:
            val_col = hist_df_monthly.columns[1]
            max_bar_val = hist_df_monthly[val_col].max(skipna=True)

            fig.add_trace(
                go.Bar(
                    x=hist_df_monthly["Date"],
                    y=hist_df_monthly[val_col],
                    name=hist_name,
                    marker_color=HISTOGRAM_COLOR,
                    opacity=HISTOGRAM_OPACITY,
                    legendrank=HISTOGRAM_LEGENDRANK,
                ),
                secondary_y=True,
            )

    # 2) add price lines ONCE, in the order you want
    if not actual.empty:
        fig.add_trace(
            go.Scatter(
                x=actual["Date"],
                y=actual["Actual Price"],
                name="Actual Price",
                mode="lines",
                line=dict(color="#f7931a", width=2),
                legendrank=ACTUAL_PRICE_LEGENDRANK,
            ),
            secondary_y=False,
        )

    if not model.empty:
        fig.add_trace(
            go.Scatter(
                x=model["Date"],
                y=model["Model Price"],
                name="Model Price",
                mode="lines",
                line=dict(color="#f7931a", width=2, dash="dash"),
                legendrank=MODEL_PRICE_LEGENDRANK,
            ),
            secondary_y=False,
        )

    # 3) today line
    if start_ts <= today_ts <= end_ts:
        fig.add_vline(
            x=today_ts,
            line_width=1,
            line_dash="dot",
            line_color="gray",
        )
        fig.add_annotation(
            x=1,
            y=1.02,
            xref="paper",
            yref="paper",
            text="Key",
            showarrow=False,
            xanchor="right",
        )

    # axes
    fig.update_yaxes(title_text="Price (USD)", secondary_y=False)

    if max_bar_val is not None:
        top = max_bar_val / HISTOGRAM_HEIGHT_RATIO
        fig.update_yaxes(title_text=hist_name, secondary_y=True, range=[0, top])
    else:
        fig.update_yaxes(title_text="", secondary_y=True)

    # layout
    fig.update_layout(
        title="Bitcoin Price (actual/forecast) & block histogram",
        xaxis_title="Date",
        hovermode="x unified",
        template=template,
        height=600,
        margin=dict(l=80, r=40, t=60, b=60),
        xaxis=dict(range=[start_ts, end_ts]),
        barmode="overlay",
        legend=dict(
            title_text="Key",
            title_font=dict(size=12),
            traceorder="normal",
            x=1,
            y=1,
            xanchor="right",
            yanchor="top",
        ),
    )

    return fig, mc_df
