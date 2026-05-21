"""
Live Trader — orchestrates the daily live trading cycle.
Decomposed into fetch → compute → execute pipeline.
Integrates circuit breaker, Kelly criterion, and trailing stop loss.
"""
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
import yfinance as yf

from agents.portfolio_manager import PortfolioManager
from backtester.circuit_breaker import CircuitBreaker
from broker.alpaca import AlpacaExecutionAgent
from config.settings import DEFAULT_PRICE_FALLBACK, MAX_RETRIES, SYMBOLS
from factors.earnings_nlp import get_earnings_nlp_score
from factors.fundamentals import get_value_score
from factors.options_skew import get_options_penalty
from strategy.value_strategy import calculate_blended_weights
from utils.rationale_generator import generate_trade_rationale

logger = logging.getLogger(__name__)


class DiscordNotifier:
    def __init__(self, webhook_url: str | None = None):
        self.webhook_url = webhook_url

    def notify(self, message: str) -> None:
        logger.info(f"[Notification] {message}")
        if self.webhook_url:
            try:
                requests.post(self.webhook_url, json={"content": message}, timeout=10)
            except requests.RequestException as e:
                logger.error(f"Failed to send notification: {e}")


# ─── Concurrent Data Fetching ───────────────────────────────────────────


async def _fetch_symbol_data(symbol, agent, executor, max_retries):
    loop = asyncio.get_running_loop()

    def fetch_yf():
        import time
        for attempt in range(max_retries):
            try:
                df = yf.download(symbol, period='2mo', interval='1d', progress=False)
                if not df.empty and 'Close' in df.columns:
                    return df['Close'].squeeze()
                return pd.Series()
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return pd.Series()

    tasks = [
        loop.run_in_executor(executor, fetch_yf),
        loop.run_in_executor(executor, get_value_score, symbol),
        loop.run_in_executor(executor, get_earnings_nlp_score, symbol),
        loop.run_in_executor(executor, get_options_penalty, symbol),
        loop.run_in_executor(executor, agent.get_current_price, symbol)
        if agent
        else asyncio.sleep(0, result=DEFAULT_PRICE_FALLBACK),
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return symbol, results


async def _fetch_all_factors(symbols, agent, max_retries):
    with ThreadPoolExecutor(max_workers=min(len(symbols) * 5, 35)) as executor:
        tasks = [_fetch_symbol_data(sym, agent, executor, max_retries) for sym in symbols]
        results = await asyncio.gather(*tasks)

    close_series = {}
    val_scores = {}
    nflo_scores = {}
    opt_penalties = {}
    prices = {}

    for symbol, res in results:
        close_series[symbol] = res[0] if not isinstance(res[0], Exception) else pd.Series()
        val_scores[symbol] = res[1] if not isinstance(res[1], Exception) else 0
        nflo_scores[symbol] = res[2] if not isinstance(res[2], Exception) else 0
        opt_penalties[symbol] = res[3] if not isinstance(res[3], Exception) else 0
        prices[symbol] = res[4] if not isinstance(res[4], Exception) else DEFAULT_PRICE_FALLBACK

    df = pd.DataFrame(close_series)
    return df, val_scores, nflo_scores, opt_penalties, prices


# ─── Pipeline Stages ────────────────────────────────────────────────────


def fetch_data(symbols: list[str], agent: AlpacaExecutionAgent) -> tuple:
    """Stage 1: Fetch all alpha factor data concurrently."""
    logger.info(f"Fetching all 5 alpha factors concurrently for {len(symbols)} symbols...")
    return asyncio.run(_fetch_all_factors(symbols, agent, MAX_RETRIES))


def compute_signals(
    symbols: list[str],
    returns_df: pd.DataFrame,
    val_scores: dict,
    nflo_scores: dict,
    opt_penalties: dict,
) -> dict[str, float]:
    """Stage 2: Compute blended alpha signals and target weights."""
    target_weights_array = calculate_blended_weights(
        symbols, returns_df,
        val_scores_dict=val_scores,
        nflo_scores_dict=nflo_scores,
        opt_penalties_dict=opt_penalties,
    )
    return {sym: float(w) for sym, w in zip(symbols, target_weights_array)}


def execute_orders(
    agent: AlpacaExecutionAgent,
    weights: dict[str, float],
    prices: dict[str, float],
    notifier: DiscordNotifier,
) -> list:
    """Stage 3: Execute orders to match target weights."""
    orders = agent.execute_weights(weights, prices_dict=prices)
    notifier.notify(f"Executed {len(orders)} orders.")
    return orders


# ─── Main Cycle ─────────────────────────────────────────────────────────


def run_daily_cycle() -> None:
    """Orchestrates the full daily live trading cycle."""
    symbols = SYMBOLS
    notifier = DiscordNotifier()
    notifier.notify(f"Starting daily live trading cycle for {len(symbols)} symbols.")

    agent = AlpacaExecutionAgent()
    pm = PortfolioManager()
    cb = CircuitBreaker()

    # Stage 1: Fetch
    logger.info("Fetching today's real-time data...")
    df, val_scores, nflo_scores, opt_penalties, prices = fetch_data(symbols, agent)
    close_prices = df
    returns_df = close_prices.pct_change().dropna(how='all')

    # Circuit breaker check (using SPY as market proxy)
    try:
        spy_data = yf.download('SPY', period='2d', interval='1d', progress=False)
        if not spy_data.empty and len(spy_data) >= 2:
            spy_return = float(spy_data['Close'].pct_change().iloc[-1])
            if cb.check_returns(spy_return):
                notifier.notify(f"⚠️ CIRCUIT BREAKER triggered. SPY return: {spy_return:.2%}. Halting execution.")
                logger.warning(f"Circuit breaker triggered: SPY return {spy_return:.2%}")
                return
    except Exception as e:
        logger.warning(f"Circuit breaker check failed, proceeding: {e}")

    # Stage 2: Compute
    target_weights = compute_signals(symbols, returns_df, val_scores, nflo_scores, opt_penalties)
    notifier.notify(f"Calculated target weights: {target_weights}")

    # Update trailing stop loss
    pm.update_trailing_stop_loss(prices)
    stop_triggered = pm.check_stop_loss(prices)
    for sym, triggered in stop_triggered.items():
        if triggered:
            target_weights[sym] = 0.0
            notifier.notify(f"🛑 Stop loss triggered for {sym}, setting weight to 0.")

    # Generate trade rationale
    top_stocks = sorted(target_weights.keys(), key=lambda k: target_weights[k], reverse=True)[:3]
    hmm_state = "Bullish Regime"  # TODO: integrate live regime detection
    kelly_fraction = 0.85
    generate_trade_rationale(hmm_state, kelly_fraction, top_stocks)

    # Stage 3: Execute
    orders = execute_orders(agent, target_weights, prices, notifier)

    notifier.notify(f"Daily cycle complete. {len(orders)} orders executed.")
    logger.info("Daily cycle complete.")


if __name__ == "__main__":
    from config.logging_config import setup_logging
    setup_logging()
    run_daily_cycle()
