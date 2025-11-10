from datetime import datetime, date

GENESIS = datetime(2009, 1, 3, 18, 15, 5)
BLOCK_TIME = 588
SATOSHIS = 100_000_000
HALVING = 210_000

# INTERNET DEPENDANT DATA
FEE_API = "https://mempool.space/api/v1/blocks"
PRICE_API = "https://blockchain.info/ticker"

# CSV FILE LOCATION !!!!???!!!!

# !!! SHOULD THIS NOT BE THE SAME AS EARLIEST_FROM_DATE !!!! or does it need to be the value of the first date in the file?
EARLIEST_DATA = datetime(2017, 8, 17).date()
NUMBER_DAYS_SEPARATION_BTWN_DATES = 180

# date bounds for historical CSV
EARLIEST_FROM_DATE = date(2017, 9, 1)     # 1 September 2017
LATEST_FROM_DATE   = date(2025, 10, 1)    # 1 October 2025

EARLIEST_TO_DATE   = date(2018, 1, 1)     # 1 January 2018
LATEST_TO_DATE     = date(2025, 11, 1)    # 1 November 2025

# x-axis, how much of the chart should be history vs forecast
HISTORY_RATIO = 0.85   # show ~85% history and 15% forecast

# ---------------------------------------------------------------------------
# CHART STYLE CONSTANTS
# ---------------------------------------------------------------------------

# Controls how tall the tallest histogram bar appears relative to its axis.
HISTOGRAM_HEIGHT_RATIO = 0.6   # e.g. 0.8 = top bar reaches 80% of right axis

# Controls the bar colour for the histogram (any Plotly-compatible colour string).
HISTOGRAM_COLOR = "rgba(128,128,128,0.5)"  # semi-transparent light grey

# --- price line styles (for ui/charts.py) ---
# Bitcoin orange for actual
ACTUAL_PRICE_COLOR = "#f7931a"
# slightly lighter/or warmer for model so they’re distinguishable
MODEL_PRICE_COLOR = "#ffb347"

# Plotly dash options: "solid", "dash", "dot", "dashdot"
ACTUAL_PRICE_LINESTYLE = "solid"
MODEL_PRICE_LINESTYLE = "dash"

# widths (kept as constants so you can tweak globally)
ACTUAL_PRICE_LINEWIDTH = 2
MODEL_PRICE_LINEWIDTH = 2

# Controls how opaque the histogram bars are (0 = fully transparent, 1 = solid).
HISTOGRAM_OPACITY = 0.5

# ---------------------------------------------------------------------------
# CHART LEGEND ORDER CONSTANTS
# ---------------------------------------------------------------------------

# Lower numbers appear earlier in the legend.
ACTUAL_PRICE_LEGENDRANK = 1
MODEL_PRICE_LEGENDRANK = 2
HISTOGRAM_LEGENDRANK = 30
