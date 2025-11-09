import pandas as pd
import requests
import streamlit as st

from config import FEE_API, PRICE_API, SATOSHIS


def get_historical_prices() -> pd.DataFrame:
    """Load historical BTC daily price CSV if available."""
    try:
        df = pd.read_csv("data/BTCUSD_daily.csv", skiprows=1)
        df["date"] = pd.to_datetime(df["Date"]).dt.date
        df = df[["date", "Close"]].rename(columns={"Close": "price"})
        df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()


def get_avg_fee_btc(blocks: int = 2016) -> float:
    try:
        data = requests.get(FEE_API, timeout=10).json()[:blocks]
        total_fees = sum(b["extras"]["totalFees"] for b in data)
        return total_fees / len(data) / SATOSHIS
    except Exception as e:
        st.warning(f"Fee API failed ({e}) → using fallback 0.061 BTC")
        return 0.061


def get_current_price() -> int:
    try:
        data = requests.get(PRICE_API, timeout=10).json()
        return int(data["USD"]["last"])
    except Exception as e:
        st.warning(f"Price fetch failed: {e} — using fallback 99321")
        return 99321

