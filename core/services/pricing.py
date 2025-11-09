# core/services/pricing.py

from __future__ import annotations
import pandas as pd

from core.models import build_price_series
from core.data_sources import get_historical_prices


def build_price_data(
    start_dt,
    end_dt,
    backcast: bool,
    price_model: str,
    custom_price: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the price dataframe(s) used by the UI and charts.

    Returns:
        plot_df: the main dataframe with Model Price (and Actual Price if backcast)
        hist_df: the raw historical prices dataframe (may be empty)
    """
    # 1) model series
    model_df = build_price_series(start_dt, end_dt, backcast=backcast)

    # 2) historical
    hist_df = get_historical_prices()

    # 3) merge actuals when backcasting
    if backcast and not hist_df.empty:
        hist_df["date"] = pd.to_datetime(hist_df["date"])
        plot_df = model_df.merge(
            hist_df, left_on="Date", right_on="date", how="left"
        )
        plot_df["Actual Price"] = plot_df["price"]
    else:
        plot_df = model_df.copy()
        plot_df["Actual Price"] = pd.NA

    # 4) rescale model to last actual (best-effort, silent on failure)
    if backcast and not hist_df.empty and "price" in hist_df.columns:
        try:
            last_actual_date = hist_df["date"].max()
            last_actual_price = float(
                hist_df.loc[hist_df["date"] == last_actual_date, "price"].iloc[0]
            )
            join_mask = plot_df["Date"] >= pd.to_datetime(last_actual_date)
            if join_mask.any():
                model_at_join = float(
                    plot_df.loc[join_mask, "Model Price"].iloc[0]
                )
                if model_at_join > 0:
                    scale_factor = last_actual_price / model_at_join
                    plot_df["Model Price"] = plot_df["Model Price"] * scale_factor
        except Exception:
            # keep going even if scaling fails
            pass

    # 5) custom price override
    if price_model == "Custom":
        first_model = plot_df["Model Price"].iloc[0]
        if first_model > 0:
            plot_df["Model Price"] = plot_df["Model Price"] * (custom_price / first_model)

    return plot_df, hist_df
