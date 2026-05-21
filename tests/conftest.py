"""
Shared test fixtures for the Quant Trading test suite.
"""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def mock_returns_df():
    """Create a deterministic returns DataFrame for 7 symbols over 60 days."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=60, freq='B')
    symbols = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA']
    data = {sym: np.random.normal(0.001, 0.02, 60) for sym in symbols}
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def mock_prices_df(mock_returns_df):
    """Create a deterministic close prices DataFrame from returns."""
    return (1 + mock_returns_df).cumprod() * 100


@pytest.fixture
def mock_weights(mock_returns_df):
    """Equal-weight portfolio weights."""
    n = len(mock_returns_df.columns)
    return np.ones(n) / n


@pytest.fixture
def mock_score_dicts():
    """Mock score dicts for value, NFLO, and options penalty."""
    symbols = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA']
    return {
        'val': {s: np.random.uniform(0, 0.3) for s in symbols},
        'nflo': {s: np.random.uniform(-0.5, 0.5) for s in symbols},
        'opt': {s: 0.0 for s in symbols},
    }
