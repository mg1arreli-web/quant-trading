"""
Trade Rationale Generator — template-based rationale for trade decisions.
"""
import logging

logger = logging.getLogger(__name__)


def generate_trade_rationale(hmm_state: str, kelly_fraction: float, top_stocks: list[str]) -> str:
    """
    Returns a template-based rationale explaining the trade based on the given parameters.
    """
    stocks_str = ", ".join(top_stocks) if top_stocks else "none"
    rationale = (
        f"Market state is {hmm_state}. Kelly fraction at {kelly_fraction:.2f}. "
        f"Increasing exposure to {stocks_str} due to converging bullish alpha signals "
        f"and favorable risk-reward profiling under current macroeconomic conditions."
    )
    logger.info(f"Trade Rationale: {rationale}")
    return rationale
