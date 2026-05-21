"""
Fundamental Value Factor — computes a value score from P/E, ROE, and FCF yield.
"""
import logging

import yfinance as yf

logger = logging.getLogger(__name__)


def get_value_score(symbol: str) -> float:
    """
    Fetch fundamental metrics and calculate a value score.
    Sub-factors are on comparable scales (no arbitrary multipliers).
    Z-scoring across symbols happens downstream in value_strategy.py.
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        pe = info.get('forwardPE')
        roe = info.get('returnOnEquity')
        fcf = info.get('freeCashflow')
        mcap = info.get('marketCap')

        score = 0.0

        # Earnings yield (1/PE) — typically 0.02 to 0.10
        if pe is not None and pe > 0:
            score += 1.0 / pe

        # ROE — typically 0.05 to 0.40
        if roe is not None:
            score += roe

        # FCF yield — typically 0.01 to 0.10
        if fcf is not None and mcap is not None and mcap > 0:
            score += fcf / mcap

        return score
    except Exception as e:
        logger.warning(f"Error fetching fundamentals for {symbol}: {e}")
        return 0.0
