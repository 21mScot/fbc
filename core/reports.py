from datetime import date, datetime
import pandas as pd
from core.blockchain import block_from_date, subsidy_at


def _to_date(d) -> date:
    """Accept date or datetime and return date."""
    if isinstance(d, datetime):
        return d.date()
    return d


def _first_of_next_month(d: date) -> date:
    """Return the first day of the month after d."""
    if d.month == 12:
        return date(d.year + 1, 1, 1)
    return date(d.year, d.month + 1, 1)


def build_monthly_table(
    start_dt,
    end_dt,
    plot_df: pd.DataFrame,
    price_model: str,
    mc_df: pd.DataFrame | None,
    current_fee: float,
    fee_growth: float
) -> pd.DataFrame:
    """
    Build the Monthly Breakdown table between two dates.
    Returns a plain DataFrame (Streamlit will style/hide index).
    """

    # normalise to dates
    start_dt = _to_date(start_dt)
    end_dt = _to_date(end_dt)

    monthly = []

    # align to first of month
    cur = start_dt.replace(day=1)
    start_month = cur
    price_idx = 0

    while cur <= end_dt:
        # block height
        h = block_from_date(cur)

        # price selection
        if price_model == "Monte Carlo (GBM)" and mc_df is not None:
            if price_idx < len(mc_df):
                price_val = float(mc_df.iloc[price_idx]["price"])
            else:
                price_val = float(mc_df.iloc[-1]["price"])
        else:
            if price_idx < len(plot_df):
                # adjust name if different in your DF
                price_val = float(plot_df.iloc[price_idx]["Model Price"])
            else:
                price_val = float(plot_df.iloc[-1]["Model Price"])

        # fee growth
        days_from_start = (cur - start_month).days
        est_fee = current_fee * (1 + fee_growth) ** (days_from_start / 365.25)

        monthly.append(
            {
                "Month": cur.strftime("%Y-%m"),
                "Block": h,
                "Subsidy": subsidy_at(h),
                "Est. Fee": est_fee,
                "Price": price_val,
            }
        )

        # next month
        cur = _first_of_next_month(cur)
        price_idx += 1

    df = pd.DataFrame(monthly)
    df = df.reset_index(drop=True)
    return df
