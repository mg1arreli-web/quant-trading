"""
Value Strategy — multi-factor alpha blender with z-score normalization.
"""
import logging

import numpy as np
import pandas as pd

from config.settings import MOMENTUM_WEIGHT, MOMENTUM_WINDOW, NFLO_WEIGHT, VALUE_WEIGHT

logger = logging.getLogger(__name__)


def calculate_blended_weights(
    symbols: list[str],
    returns_df: pd.DataFrame,
    val_scores_dict: dict[str, float],
    nflo_scores_dict: dict[str, float],
    opt_penalties_dict: dict[str, float],
    momentum_window: int = MOMENTUM_WINDOW,
    value_weight: float = VALUE_WEIGHT,
    momentum_weight: float = MOMENTUM_WEIGHT,
    nflo_weight: float = NFLO_WEIGHT,
) -> np.ndarray:
    """
    Blends momentum, value, and NFLO scores into target portfolio weights.
    When all alphas are negative, reduces exposure (50% cash) instead of equal weight.
    """
    # 1. Momentum scores
    recent_returns = returns_df.tail(momentum_window)
    mom_scores = recent_returns.mean()

    # 2. Value scores
    val_scores = pd.Series(val_scores_dict)

    # 3. NFLO scores
    nflo_scores = pd.Series(nflo_scores_dict)

    # Fill NAs
    mom_scores = mom_scores.fillna(0)
    val_scores = val_scores.fillna(0)
    nflo_scores = nflo_scores.fillna(0)

    # Z-score normalization for blending
    def z_score(series: pd.Series) -> pd.Series:
        if len(series) <= 1 or series.std() == 0:
            return pd.Series(0, index=series.index)
        return (series - series.mean()) / series.std()

    mom_z = z_score(mom_scores)
    val_z = z_score(val_scores)
    nflo_z = z_score(nflo_scores)

    # 4. Options Penalties (continuous)
    opt_penalties = pd.Series(opt_penalties_dict)
    opt_penalties = opt_penalties.fillna(0)

    # Blended Alpha
    blended_alpha = (value_weight * val_z) + (momentum_weight * mom_z) + (nflo_weight * nflo_z)
    blended_alpha = blended_alpha + opt_penalties

    # Convert alpha to long-only weights
    shifted_alpha = blended_alpha.copy()
    shifted_alpha[shifted_alpha < 0] = 0

    if shifted_alpha.sum() == 0:
        # All signals bearish — reduce exposure to 50% (partial cash position)
        logger.warning("All blended alphas are non-positive. Using reduced 50% equal-weight allocation.")
        return np.ones(len(symbols)) / len(symbols) * 0.5

    weights = shifted_alpha / shifted_alpha.sum()

    return weights.values
