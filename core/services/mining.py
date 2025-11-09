# core/services/mining.py
from datetime import date
from core.blockchain import block_from_date, subsidy_at
from core.data_sources import get_avg_fee_btc, get_current_price


def compute_mining_economics(
    start_date: date,
    end_date: date,
    fee_growth: float,
) -> dict:
    """
    Calculate blocks, subsidy, fees and current USD value for the period.

    Returns a dict with:
    - blocks
    - subsidy_btc
    - current_fee
    - avg_fee_forecast
    - fees_btc
    - total_btc
    - current_price
    - usd_value
    """

    # blocks in range
    h1 = block_from_date(start_date)
    h2 = block_from_date(end_date)
    blocks = h2 - h1 + 1

    # handle halving boundary
    era1, era2 = h1 // 210_000, h2 // 210_000
    if era1 == era2:
        subsidy_btc = blocks * subsidy_at(h1)
    else:
        # span across a halving -> sum block by block
        subsidy_btc = sum(subsidy_at(h) for h in range(h1, h2 + 1))

    # network fee today
    current_fee = get_avg_fee_btc()

    # fee growth over the period
    # keep your original sign/logic even though dates are descending
    years = (start_date - end_date).days / 365.25
    avg_fee_forecast = current_fee * ((1 + fee_growth) ** (years / 2))

    # total fees in BTC
    fees_btc = blocks * avg_fee_forecast

    # total BTC
    total_btc = subsidy_btc + fees_btc

    # value it
    current_price = get_current_price()
    usd_value = total_btc * current_price

    return {
        "blocks": blocks,
        "subsidy_btc": subsidy_btc,
        "current_fee": current_fee,
        "avg_fee_forecast": avg_fee_forecast,
        "fees_btc": fees_btc,
        "total_btc": total_btc,
        "current_price": current_price,
        "usd_value": usd_value,
    }
