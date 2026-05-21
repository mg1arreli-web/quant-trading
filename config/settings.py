"""
Centralized configuration for the Quant Trading system.
All magic numbers and tunable parameters live here.
"""

# ──────────────────────────────────────────────
# Trading Universe
# ──────────────────────────────────────────────
SYMBOLS = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA']

# ──────────────────────────────────────────────
# Backtest Parameters
# ──────────────────────────────────────────────
BACKTEST_START = '2020-01-01'
BACKTEST_END = '2020-12-31'
RISK_FREE_RATE = 0.045  # Annualized (~4.5% for 10Y Treasury)

# ──────────────────────────────────────────────
# Transaction Cost Model
# ──────────────────────────────────────────────
SLIPPAGE_BPS = 5            # 0.05% per trade (5 basis points)
COMMISSION_PER_SHARE = 0.005
COMMISSION_MINIMUM = 1.0

# ──────────────────────────────────────────────
# Execution
# ──────────────────────────────────────────────
TWAP_THRESHOLD = 1000       # Shares above which TWAP is used
CACHE_TTL = 5.0             # Seconds for broker cache TTL
DEFAULT_PRICE_FALLBACK = 100.0
MAX_RETRIES = 3

# ──────────────────────────────────────────────
# Alpha Factors
# ──────────────────────────────────────────────
MOMENTUM_WINDOW = 20
VALUE_WEIGHT = 0.4
MOMENTUM_WEIGHT = 0.3
NFLO_WEIGHT = 0.3
PC_RATIO_THRESHOLD = 1.2

# ──────────────────────────────────────────────
# Risk Management
# ──────────────────────────────────────────────
VIX_CRISIS_THRESHOLD = 0.15
CRISIS_COOLDOWN_DAYS = 20
CIRCUIT_BREAKER_THRESHOLD = -0.07
KELLY_FRACTION_CAP = 1.0
TRAILING_STOP_LOSS_PCT = 0.10
MAX_SLIPPAGE_PCT = 0.05

# ──────────────────────────────────────────────
# Scheduling (Daemon)
# ──────────────────────────────────────────────
SCHEDULE_TIME = "09:30"
SCHEDULE_TIMEZONE = "US/Eastern"
