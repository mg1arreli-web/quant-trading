# Quant Trading

A systematic, multi-factor equity trading system that combines **regime-aware macro modeling**, **NLP-derived sentiment**, and **quantitative alpha signals** into a single automated pipeline — from signal generation through order execution on Alpaca.

Built for the US large-cap universe (Magnificent 7 by default). Runs as a daily scheduled daemon or as a standalone backtester.

---

## How It Works

```
┌─────────────┐    ┌─────────────┐    ┌──────────────┐    ┌───────────┐
│  Data Fetch  │───▶│  Alpha Calc  │───▶│  Risk Overlay │───▶│  Execute  │
│  (yfinance,  │    │  (Value,     │    │  (HMM Regime, │    │  (Alpaca  │
│   AV, FMP)   │    │   Mom, NFLO) │    │   Stop Loss,  │    │   Paper/  │
│              │    │              │    │   Kelly, CB)   │    │   Live)   │
└─────────────┘    └─────────────┘    └──────────────┘    └───────────┘
```

**Alpha factors** (momentum, fundamental value, earnings NLP, options skew) are z-scored and blended into target weights. A **Hidden Markov Model** trained on VIX/TNX returns classifies the macro regime as Risk-On or Risk-Off, zeroing exposure during systemic stress. A **circuit breaker** monitors intraday SPY drawdowns and halts execution if a flash crash is detected. Weights are scaled by **Kelly criterion** and filtered through **trailing stop losses** before being sent as market orders to Alpaca.

The backtester applies the same pipeline to historical data with a **realistic transaction cost model** (configurable slippage + per-share commission) and reports Sharpe, Sortino, Calmar, Max Drawdown, and Annual Volatility.

---

## Project Structure

```
├── daemon.py                    # Production entry point (APScheduler, graceful shutdown)
├── live_trader.py               # Live trading: fetch → compute → execute pipeline
├── main.py                      # Backtest entry point with full metrics report
│
├── agents/
│   ├── macro_risk_manager.py    # HMM regime detection (hmmlearn / sklearn fallback)
│   ├── portfolio_manager.py     # Markowitz optimization, Kelly sizing, trailing stops
│   └── sentiment_analyst.py     # Residual Sentiment Momentum (RSM) factor
│
├── backtester/
│   ├── engine.py                # Backtester with transaction costs and 7 metrics
│   └── circuit_breaker.py       # Market-wide drawdown halt
│
├── broker/
│   └── alpaca.py                # Alpaca API with tenacity retries, TTL cache, TWAP
│
├── config/
│   ├── settings.py              # All tunable constants (25+)
│   └── logging_config.py        # Centralized logging setup
│
├── data_pipeline/
│   └── sec_transcript_fetcher.py  # FMP earnings transcript API
│
├── factors/
│   ├── fundamentals.py          # Value score (earnings yield + ROE + FCF yield)
│   ├── earnings_nlp.py          # NFLO — Net Forward-Looking Optimism from transcripts
│   ├── options_skew.py          # Put/Call ratio continuous penalty
│   └── pairs_trading.py         # Cointegration z-score (numba JIT when available)
│
├── strategy/
│   └── value_strategy.py        # Multi-factor alpha blender with z-score normalization
│
├── utils/
│   ├── state_manager.py         # Atomic JSON state persistence
│   └── rationale_generator.py   # Template-based trade rationale
│
├── tests/                       # 45 tests (pytest)
├── Dockerfile                   # Non-root, health-checked container
├── docker-compose.yml           # Resource limits, log rotation, volume mounts
├── requirements.txt             # Pinned dependencies
└── pyproject.toml               # pytest / mypy / ruff config
```

---

## Setup

### Prerequisites

- Python 3.10+
- API keys: [Alpaca](https://alpaca.markets/) (paper or live), optionally [AlphaVantage](https://www.alphavantage.co/) and [FMP](https://financialmodelingprep.com/)

### Install

```bash
git clone https://github.com/mg1arreli-web/quant-trading.git
cd quant-trading

python -m venv venv
source venv/bin/activate        # Linux/WSL
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### Configure

Create a `.env` file in the project root:

```env
ALPACA_API_KEY=your_key
ALPACA_API_SECRET=your_secret

# Optional — falls back to yfinance if not set
ALPHAVANTAGE_API_KEY=your_key
FMP_API_KEY=your_key
```

---

## Usage

### Backtest

Runs the full pipeline on historical data (default: Mag-7, 2020) with transaction costs and HMM regime overlay:

```bash
python main.py
```

Sample output:

```
  Total Return:          27.65%
  Annual Return:         27.78%
  Sharpe Ratio:            0.88
  Sortino Ratio:           1.16
  Calmar Ratio:            1.31
  Max Drawdown:         -21.23%
  Annual Volatility:     26.89%
```

Backtest parameters are configured in [`config/settings.py`](config/settings.py):

| Parameter | Default | Description |
|---|---|---|
| `SYMBOLS` | Mag-7 | Trading universe |
| `BACKTEST_START` | `2020-01-01` | Start date |
| `BACKTEST_END` | `2020-12-31` | End date |
| `SLIPPAGE_BPS` | `5` | Slippage per trade (basis points) |
| `COMMISSION_PER_SHARE` | `0.005` | Per-share commission |
| `RISK_FREE_RATE` | `0.045` | Annualized risk-free rate |

### Live Trading

Runs the daemon, which executes the full cycle daily at market open (9:30 ET) and sleeps between runs:

```bash
python daemon.py
```

Or run a single cycle manually:

```bash
python live_trader.py
```

### Docker

```bash
docker compose up --build -d
```

The container runs as a non-root user, mounts `./data` for state persistence, and includes a health check.

---

## Testing

```bash
# Run all 45 tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=. --cov-report=term-missing

# Lint
ruff check .

# Type check
mypy agents/ broker/ backtester/ strategy/ factors/ utils/ --ignore-missing-imports
```

---

## Architecture Notes

**Regime Model** — A 2-state Gaussian HMM (or sklearn GMM fallback) is trained on daily VIX and TNX returns from 2010–2022. The state with higher mean VIX return is labeled Risk-Off. During backtest, Risk-Off days have zero equity exposure.

**Alpha Blending** — Three z-scored factors (Value, Momentum, NFLO) are linearly combined with configurable weights (default: 0.4 / 0.3 / 0.3). Options skew applies a continuous penalty. When all blended alphas are negative, exposure is reduced to 50% equal-weight (partial cash position) rather than forcing full investment.

**Transaction Costs** — The backtester applies per-trade slippage (configurable in basis points) and per-share commission with a minimum floor. Execution prices are adjusted directionally: buys execute at `price × (1 + slippage)`, sells at `price × (1 - slippage)`.

**Execution** — Orders above `TWAP_THRESHOLD` shares are sliced into 10 tranches. The broker module uses `tenacity` exponential-backoff retries and an in-memory TTL cache to avoid redundant API calls within the same cycle.

---

## License

MIT
