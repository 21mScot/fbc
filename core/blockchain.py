from datetime import datetime, timedelta
from config import GENESIS, BLOCK_TIME, HALVING


def subsidy_at(h: int) -> float:
    return 50 / (2 ** (h // HALVING))


def block_from_date(date_input) -> int:
    """Convert date/datetime to block height. Handles both date and datetime inputs."""
    if isinstance(date_input, datetime):
        dt = date_input
    else:
        dt = datetime.combine(date_input, datetime.min.time())
    return int((dt - GENESIS).total_seconds() / BLOCK_TIME)


def date_from_block(height: int) -> datetime:
    return GENESIS + timedelta(seconds=height * BLOCK_TIME)
