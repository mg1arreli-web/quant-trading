"""
Pairs Trading Factor — cointegration-based statistical arbitrage.
Uses numba JIT when available, falls back to pure numpy.
"""
import logging

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

logger = logging.getLogger(__name__)

# Conditional numba import
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def jit(*args, **kwargs):
        """No-op decorator when numba is not installed."""
        def wrapper(func):
            return func
        return wrapper


@jit(nopython=True)
def numba_rolling_stats(arr, window):
    n = len(arr)
    mean_res = np.empty(n)
    mean_res[:] = np.nan
    std_res = np.empty(n)
    std_res[:] = np.nan

    for i in range(window - 1, n):
        window_slice = arr[i - window + 1: i + 1]
        m = np.mean(window_slice)
        mean_res[i] = m

        var = 0.0
        for val in window_slice:
            var += (val - m) ** 2
        std_res[i] = np.sqrt(var / (window - 1))

    return mean_res, std_res


@jit(nopython=True)
def numba_simple_ols(y, x):
    """Simple OLS returning both slope and intercept."""
    n = len(x)
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    cov_xy = 0.0
    var_x = 0.0
    for i in range(n):
        cov_xy += (x[i] - mean_x) * (y[i] - mean_y)
        var_x += (x[i] - mean_x) ** 2

    slope = cov_xy / var_x if var_x > 0 else 0.0
    intercept = mean_y - slope * mean_x
    return slope, intercept


def calculate_pairs_zscore(
    price_series_1: pd.Series,
    price_series_2: pd.Series,
    window: int = 20,
) -> tuple:
    """
    Calculate pairs trading z-score and signals based on cointegration spread.

    Returns:
        tuple: (cointegration p-value, z-score series, signal series)
    """
    score, pvalue, _ = coint(price_series_1, price_series_2)

    beta, alpha = numba_simple_ols(price_series_1.values, price_series_2.values)

    # Spread now includes intercept
    spread = price_series_1 - beta * price_series_2 - alpha

    rolling_mean, rolling_std = numba_rolling_stats(spread.values, window)

    zscore = (spread - rolling_mean) / rolling_std

    signals = pd.Series(0, index=zscore.index, dtype=int)
    signals[zscore > 2.0] = -1
    signals[zscore < -2.0] = 1

    return pvalue, zscore, signals
