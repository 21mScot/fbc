from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from config import GENESIS, EARLIEST_DATA


def power_law_price(days_since_genesis: int) -> float:
    # crude descriptive model — we’ll re-scale it later
    return 10_000 * (days_since_genesis ** 0.3)


def build_price_series(start_date: datetime, end_date: datetime, backcast: bool) -> pd.DataFrame:
    """Monthly-ish model series from start → end."""
    genesis = GENESIS
    today = datetime.today().date()

    if backcast:
        series_start = min(start_date.date(), EARLIEST_DATA)
        series_start = datetime.combine(series_start, datetime.min.time())
    else:
        series_start = start_date

    dates = []
    model_prices = []
    is_hist = []

    cur = series_start
    while cur <= end_date:
        days = (cur - genesis).days
        if days < 365:
            cur += timedelta(days=30)
            continue

        price = power_law_price(days)
        dates.append(cur)
        model_prices.append(price)
        is_hist.append(cur.date() <= today)
        cur += timedelta(days=30)

    return pd.DataFrame({
        "Date": pd.to_datetime(dates),
        "Model Price": model_prices,
        "Historical": is_hist,
    })


def monte_carlo_gbm(start_price: float, start_dt: datetime, end_dt: datetime,
                    mu: float, sigma: float, n_sims: int = 500) -> pd.DataFrame:
    days = (end_dt - start_dt).days
    if days <= 0:
        days = 1

    dates = [start_dt + timedelta(days=i) for i in range(days + 1)]
    dt = 1 / 365

    sims = np.zeros((len(dates), n_sims), dtype=float)
    sims[0, :] = start_price

    for t in range(1, len(dates)):
        rand = np.random.normal(0, 1, n_sims)
        sims[t, :] = sims[t - 1, :] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * rand)

    p5 = np.percentile(sims, 5, axis=1)
    p25 = np.percentile(sims, 25, axis=1)
    p50 = np.percentile(sims, 50, axis=1)
    p75 = np.percentile(sims, 75, axis=1)
    p95 = np.percentile(sims, 95, axis=1)

    return pd.DataFrame({
        "Date": pd.to_datetime(dates),
        "P05": p5,
        "P25": p25,
        "P50": p50,
        "P75": p75,
        "P95": p95,
    })


def estimate_mu_sigma_from_history(hist_df: pd.DataFrame) -> tuple[float, float]:
    if hist_df is None or hist_df.empty:
        annual_mu = 0.35
        annual_sigma = 0.75
        daily_mu = (1 + annual_mu) ** (1 / 365) - 1
        daily_sigma = annual_sigma / np.sqrt(252)
        return daily_mu, daily_sigma

    prices = hist_df["price"].astype(float).values
    returns = np.diff(np.log(prices))
    daily_mu = returns.mean()
    daily_sigma = returns.std()
    return daily_mu, daily_sigma
