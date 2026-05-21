"""
Options Skew Factor — continuous Put/Call volume ratio penalty.
"""
import logging

import yfinance as yf

logger = logging.getLogger(__name__)


def get_options_penalty(symbol: str) -> float:
    """
    Fetch nearest-term options chain, compute P/C volume ratio,
    and return a continuous penalty scaled linearly above threshold.
    """
    try:
        ticker = yf.Ticker(symbol)
        expirations = ticker.options
        if not expirations:
            return 0.0

        chain = ticker.option_chain(expirations[0])
        calls = chain.calls
        puts = chain.puts

        call_vol = calls['volume'].fillna(0).sum() if 'volume' in calls.columns else 0
        put_vol = puts['volume'].fillna(0).sum() if 'volume' in puts.columns else 0

        if call_vol == 0:
            return 0.0

        pc_ratio = put_vol / call_vol

        # Continuous penalty: scales linearly above 1.0
        penalty = -max(0.0, (pc_ratio - 1.0)) * 0.5
        return penalty

    except Exception as e:
        logger.warning(f"Options data unavailable for {symbol}: {e}")
        return 0.0
