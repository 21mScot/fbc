import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from config import (
    HISTORY_RATIO,
    HISTOGRAM_HEIGHT_RATIO,
    HISTOGRAM_COLOR,
    HISTOGRAM_OPACITY,
)
from core.models import estimate_mu_sigma_from_history, monte_carlo_gbm


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _hist_from_monthly_breakdown(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Turn your monthly table (Month | Block | Subsidy | ...) into a
    time series we can plot. If we can't, return None.
    """
    if df is None or df.empty:
        return None

    df = df.copy()
    if "Month" not in df.columns or "Block" not in df.columns:
        return None

    df["Date"] = pd.to_datetime(df["Month"])
    df = df.sort_values("Date")

    # clean numeric
    df["Block"] = (
        df["Block"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
    )
    df["Block"] = pd.to_numeric(df["Block"], errors="coerce")

    df["Blocks (derived)"] = df["Block"].diff()

    # first one is huge, don't plot it
    if len(df):
        df.loc[df.index[0], "Blocks (derived)"] = float("nan")

    out = df[["Date", "Blocks (derived)"]]
    if out.empty:
        return None
    return out


def _find_histogram_source(plot_df: pd.DataFrame, hist_df: pd.DataFrame | None):
    """
    Try to find something to plot as bars.

    Priority:
    1. session monthly_breakdown -> Subsidy (best for halvings)
    2. session monthly_breakdown -> Blocks (derived)
    3. plot_df / hist_df -> block-like columns
    """
    # 1) explicit monthly_breakdown in session
    if "monthly_breakdown" in st.session_state:
        mb = st.session_state["monthly_breakdown"]
        if isinstance(mb, pd.DataFrame) and "Month" in mb.columns:
            mb = mb.copy()
            mb["Date"] = pd.to_datetime(mb["Month"])
            # prefer Subsidy
            if "Subsidy" in mb.columns:
                mb["Subsidy"] = pd.to_numeric(mb["Subsidy"], errors="coerce")
                mb = mb[["Date", "Subsidy"]].sort_values("Date")
                if not mb.empty:
                    return "Subsidy (BTC)", mb
            # fallback: derive from Block
            monthly = _hist_from_monthly_breakdown(mb)
            if monthly is not None and not monthly.empty:
                return "Blocks (derived)", monthly

    # 2) scan other session dfs
    for v in st.session_state.values():
        if isinstance(v, pd.DataFrame) and "Month" in v.columns and "Block" in v.columns:
            monthly = _hist_from_monthly_breakdown(v)
            if monthly is not None and not monthly.empty:
                return "Blocks (derived)", monthly

    # 3) look in plot_df for block-ish stuff
    if plot_df is not None and not plot_df.empty:
        for cand in ["Blocks", "block_count", "blocks", "Blocks Mined"]:
            if cand in plot_df.columns:
                tmp = plot_df.copy()
                tmp["Date"] = pd.to_datetime(tmp["Date"])
                monthly = (
                    tmp.set_index("Date")[cand]
                    .resample("M")
                    .sum()
                    .reset_index()
                )
                if not monthly.empty:
                    return cand, monthly

    # 4) look in hist_df
    if hist_df is not None and not hist_df.empty:
        for cand in ["Blocks", "block_count", "blocks", "Blocks Mined"]:
            if cand in hist_df.columns:
                tmp = hist_df.copy()
                tmp["Date"] = pd.to_datetime(tmp["Date"])
                monthly = (
                    tmp.set_index("Date")[cand]
                    .resample("M")
                    .sum()
                    .reset_index()
                )
                if not monthly.empty:
                    return cand, monthly

    # nothing found
    return None, None


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

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

    today_ts = pd.Timestamp.today().normalize()
    mc_df = None

    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)

    # normalise incoming
    plot_df = plot_df.copy()
    plot_df["Date"] = pd.to_datetime(plot_df["Date"])
    plot_df_window = plot_df[(plot_df["Date"] >= start_ts) & (plot_df["Date"] <= end_ts)].copy()

    # trim forecast if backcasting
    plot_df_trim = plot_df_window.copy()
    history_mask = plot_df_trim["Actual Price"].notna()
    if backcast and history_mask.any():
        hist_start = plot_df_trim.loc[history_mask, "Date"].min()
        hist_end = plot_df_trim.loc[history_mask, "Date"].max()
        hist_days = max((hist_end - hist_start).days, 1)
        max_forecast_days = int(hist_days * (1 - HISTORY_RATIO))
        forecast_cutoff = hist_end + pd.Timedelta(days=max_forecast_days)
        plot_df_trim = plot_df_trim[plot_df_trim["Date"] <= forecast_cutoff]

    # try to get bars
    hist_col, hist_monthly = _find_histogram_source(plot_df_window, hist_df)

    # 2-axis fig
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # -------------------- PRICE ON LEFT --------------------
    if price_model == "Monte Carlo (GBM)":
        hist_part = plot_df_trim[plot_df_trim["Actual Price"].notna()]
        if not hist_part.empty:
            fig.add_trace(
                go.Scatter(
                    x=hist_part["Date"],
                    y=hist_part["Actual Price"],
                    name="Actual Price",
                    mode="lines",
                    line=dict(color="#f7931a", width=2),
                ),
                secondary_y=False,
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

        fig.add_trace(
            go.Scatter(
                x=mc_df["Date"],
                y=mc_df["P50"],
                name="MC P50",
                mode="lines",
                line=dict(width=2),
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=pd.concat([mc_df["Date"], mc_df["Date"][::-1]]),
                y=pd.concat([mc_df["P25"], mc_df["P75"][::-1]]),
                name="MC 25-75%",
                fill="toself",
                line=dict(width=0),
                opacity=0.2,
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=pd.concat([mc_df["Date"], mc_df["Date"][::-1]]),
                y=pd.concat([mc_df["P05"], mc_df["P95"][::-1]]),
                name="MC 5-95%",
                fill="toself",
                line=dict(width=0),
                opacity=0.1,
            ),
            secondary_y=False,
        )
    else:
        hist_part = plot_df_trim[plot_df_trim["Actual Price"].notna()]
        if not hist_part.empty:
            fig.add_trace(
                go.Scatter(
                    x=hist_part["Date"],
                    y=hist_part["Actual Price"],
                    name="Actual Price",
                    mode="lines",
                    line=dict(color="#f7931a", width=2),
                ),
                secondary_y=False,
            )
        model_part = plot_df_trim[plot_df_trim["Actual Price"].isna()]
        if not model_part.empty:
            fig.add_trace(
                go.Scatter(
                    x=model_part["Date"],
                    y=model_part["Model Price"],
                    name="Model Price",
                    mode="lines",
                    line=dict(color="#f7931a", width=2, dash="dash"),
                ),
                secondary_y=False,
            )

    # -------------------- BARS ON RIGHT --------------------
    max_bar_val = None
    if (
        hist_col is not None
        and hist_monthly is not None
        and not hist_monthly.empty
    ):
        # apply date window
        hist_monthly = hist_monthly[
            (hist_monthly["Date"] >= start_ts) & (hist_monthly["Date"] <= end_ts)
        ]
        if not hist_monthly.empty:
            val_col = hist_monthly.columns[1]
            max_bar_val = hist_monthly[val_col].max(skipna=True)

            fig.add_trace(
                go.Bar(
                    x=hist_monthly["Date"],
                    y=hist_monthly[val_col],
                    name=hist_col,
                    marker_color=HISTOGRAM_COLOR,
                    opacity=HISTOGRAM_OPACITY,
                ),
                secondary_y=True,
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

    # axes/layout
    fig.update_yaxes(title_text="Price (USD)", secondary_y=False)

    if max_bar_val is not None and pd.notna(max_bar_val):
        top = max_bar_val / HISTOGRAM_HEIGHT_RATIO
        fig.update_yaxes(title_text=hist_col, secondary_y=True, range=[0, top])
    else:
        fig.update_yaxes(title_text="", secondary_y=True)

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
            title_text="",
            traceorder="normal",
        ),
    )

    return fig, mc_df
