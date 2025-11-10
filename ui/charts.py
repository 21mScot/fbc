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
    # 👇 new imports from config.py
    ACTUAL_PRICE_COLOR,
    MODEL_PRICE_COLOR,
    ACTUAL_PRICE_LINESTYLE,
    MODEL_PRICE_LINESTYLE,
    ACTUAL_PRICE_LINEWIDTH,
    MODEL_PRICE_LINEWIDTH,
)
from core.models import estimate_mu_sigma_from_history, monte_carlo_gbm


# ---------- helpers ----------

def _derive_blocks_from_monthly(mb: pd.DataFrame) -> pd.DataFrame | None:
    """fallback: compute block deltas from the monthly table already in session"""
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
    """try to get the monthly breakdown the app already computed"""
    mb = st.session_state.get("monthly_breakdown")
    if not isinstance(mb, pd.DataFrame):
        return None, None
    if "Month" not in mb.columns:
        return None, None

    mb = mb.copy()
    mb["Date"] = pd.to_datetime(mb["Month"])

    # prefer subsidy because it shows halvings
    if "Subsidy" in mb.columns:
        mb["Subsidy"] = pd.to_numeric(mb["Subsidy"], errors="coerce")
        out = mb[["Date", "Subsidy"]].sort_values("Date")
        if not out.empty:
            return "Subsidy (BTC)", out

    # fallback: blocks
    out = _derive_blocks_from_monthly(mb)
    if out is not None:
        return "Blocks (derived)", out

    return None, None


def _monthly_from_df(df: pd.DataFrame, candidates: list[str]) -> tuple[str | None, pd.DataFrame | None]:
    """resample any df we have (plot_df / hist_df) to monthly."""
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
    """unified way to get the monthly series the histogram should show"""
    # 1) from session/monthly breakdown
    name, series = _monthly_from_session()
    if name and series is not None and not series.empty:
        return name, series

    # 2) from plot df
    name, series = _monthly_from_df(plot_df, ["Blocks", "block_count", "blocks", "Blocks Mined"])
    if name and series is not None and not series.empty:
        return name, series

    # 3) from historical df
    if hist_df is not None and not hist_df.empty:
        name, series = _monthly_from_df(hist_df, ["Blocks", "block_count", "blocks", "Blocks Mined"])
        if name and series is not None and not series.empty:
            return name, series

    return None, None


def _bar_centers_from_starts(starts: list[pd.Timestamp]) -> list[pd.Timestamp]:
    """
    given month start dates, return centers;
    for the last one, stay clearly inside the bar (half gap - 2 days)
    """
    centers: list[pd.Timestamp] = []
    if not starts:
        return centers

    for i, s in enumerate(starts):
        if i < len(starts) - 1:
            nxt = starts[i + 1]
            centers.append(s + (nxt - s) / 2)
        else:
            # last bar: use previous gap, but pull it a bit left so it doesn't hug the edge
            if len(starts) > 1:
                gap = starts[i] - starts[i - 1]
                center = s + gap / 2 - pd.Timedelta(days=2)
                centers.append(center)
            else:
                centers.append(s + pd.Timedelta(days=15))
    return centers


def _pick_price_per_bar(
    window_df: pd.DataFrame,
    bar_starts: list[pd.Timestamp],
    end_ts: pd.Timestamp,
) -> list[float | None]:
    """
    For each bar (month) look at the actual windowed price data in that exact span
    and take the *last* price we have (Actual preferred, else Model).
    """
    out: list[float | None] = []

    df = window_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    for i, start in enumerate(bar_starts):
        if i < len(bar_starts) - 1:
            stop = bar_starts[i + 1]
        else:
            stop = end_ts + pd.Timedelta(days=1)

        span = df[(df["Date"] >= start) & (df["Date"] < stop)].sort_values("Date")

        if span.empty:
            out.append(None)
            continue

        # prefer actual
        actual = span["Actual Price"].dropna()
        if not actual.empty:
            out.append(float(actual.iloc[-1]))
            continue

        model = span["Model Price"].dropna()
        if not model.empty:
            out.append(float(model.iloc[-1]))
        else:
            out.append(None)

    return out


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

    # histogram series
    hist_name, hist_df_monthly = _get_histogram_series(window_df, hist_df)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    max_bar_val = None
    bar_starts: list[pd.Timestamp] = []
    bar_centers: list[pd.Timestamp] = []

    # 1) histogram (bars)
    if (
        hist_name is not None
        and hist_df_monthly is not None
        and not hist_df_monthly.empty
    ):
        hist_df_monthly = (
            hist_df_monthly[
                (hist_df_monthly["Date"] >= start_ts)
                & (hist_df_monthly["Date"] <= end_ts)
            ]
            .sort_values("Date")
        )

        if not hist_df_monthly.empty:
            bar_starts = hist_df_monthly["Date"].tolist()
            bar_centers = _bar_centers_from_starts(bar_starts)

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

    # 2) price line — 1 price per bar
    if bar_starts:
        prices_per_bar = _pick_price_per_bar(window_df, bar_starts, end_ts)

        # decide which style to use
        has_any_actual = window_df["Actual Price"].notna().any()
        if has_any_actual:
            line_color = ACTUAL_PRICE_COLOR
            line_width = ACTUAL_PRICE_LINEWIDTH
            line_dash = ACTUAL_PRICE_LINESTYLE
            line_name = "Actual Price"
            legend_rank = ACTUAL_PRICE_LEGENDRANK
        else:
            line_color = MODEL_PRICE_COLOR
            line_width = MODEL_PRICE_LINEWIDTH
            line_dash = MODEL_PRICE_LINESTYLE
            line_name = "Model Price"
            legend_rank = MODEL_PRICE_LEGENDRANK

        fig.add_trace(
            go.Scatter(
                x=bar_centers,
                y=prices_per_bar,
                name=line_name,
                mode="lines",
                line=dict(
                    color=line_color,
                    width=line_width,
                    dash=line_dash,
                ),
                connectgaps=False,
                legendrank=legend_rank,
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

    # y-axes
    fig.update_yaxes(title_text="Price (USD)", secondary_y=False)
    if max_bar_val is not None:
        top = max_bar_val / HISTOGRAM_HEIGHT_RATIO
        fig.update_yaxes(title_text=hist_name, secondary_y=True, range=[0, top])
    else:
        fig.update_yaxes(title_text="", secondary_y=True)

    # pad x so first bar isn't cut
    axis_start = start_ts - pd.Timedelta(days=15)

    fig.update_layout(
        title="Bitcoin Price (actual/forecast) & block histogram",
        xaxis_title="Date",
        hovermode="x unified",
        template=template,
        height=600,
        margin=dict(l=80, r=40, t=60, b=60),
        xaxis=dict(range=[axis_start, end_ts]),
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
