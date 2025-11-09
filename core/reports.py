import pandas as pd
from datetime import timedelta

from core.blockchain import block_from_date, subsidy_at


def build_monthly_table(start_dt,
                        end_dt,
                        plot_df: pd.DataFrame,
                        price_model: str,
                        mc_df: pd.DataFrame | None,
                        current_fee: float,
                        fee_growth: float) -> pd.DataFrame:
    monthly = []
    cur = start_dt
    price_idx = 0
    while cur <= end_dt:
        h = block_from_date(cur)

        if price_model == "Monte Carlo (GBM)" and mc_df is not None:
            mc_row = mc_df[pd.to_datetime(mc_df["Date"]) == cur]
            if not mc_row.empty:
                price_val = float(mc_row["P50"].values[0])
            else:
                price_val = float(mc_df["P50"].iloc[-1])
        else:
            if price_idx < len(plot_df):
                price_val = float(plot_df.iloc[price_idx]["Model Price"])
            else:
                price_val = float(plot_df.iloc[-1]["Model Price"])

        monthly.append({
            "Month": cur.strftime("%Y-%m"),
            "Block": h,
            "Subsidy": subsidy_at(h),
            "Est. Fee": current_fee * (1 + fee_growth) ** ((cur - start_dt).days / 365.25),
            "Price": price_val,
        })

        cur += timedelta(days=30)
        price_idx += 1

    return pd.DataFrame(monthly)
