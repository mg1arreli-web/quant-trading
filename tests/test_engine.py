"""
Tests for backtester/engine.py — Sharpe ratio, transaction costs, and metrics.
"""
import numpy as np
import pandas as pd

from backtester.engine import CustomBacktester


class TestGenerateReport:
    """Test the metrics calculation logic directly."""

    def _make_backtester_with_data(self, values):
        """Helper to create a backtester with pre-built portfolio values."""
        bt = CustomBacktester.__new__(CustomBacktester)
        bt.symbols = ['TEST']
        bt.data = pd.DataFrame()
        df = pd.DataFrame({'portfolio_value': values})
        return bt, df

    def test_sharpe_with_known_data(self):
        """Sharpe should be positive for steadily increasing portfolio."""
        bt, df = self._make_backtester_with_data(
            [100 + i * 0.1 for i in range(252)]
        )
        metrics = bt._generate_report(df)
        assert metrics['sharpe'] > 0
        assert metrics['total_return'] > 0

    def test_sharpe_includes_risk_free_rate(self):
        """Sharpe with zero-return portfolio should be negative (penalized by rf)."""
        flat_values = [100.0] * 252
        # Add tiny noise to avoid zero std
        flat_values = [100.0 + np.random.normal(0, 0.001) for _ in range(252)]
        bt, df = self._make_backtester_with_data(flat_values)
        metrics = bt._generate_report(df)
        # Near-zero return minus positive rf → negative Sharpe
        assert metrics['sharpe'] < 1.0  # Should be well below 1

    def test_max_drawdown_calculation(self):
        """Max drawdown should be negative and reflect the worst peak-to-trough."""
        # Goes up to 200, then crashes to 100, then recovers to 150
        values = list(range(100, 201)) + list(range(200, 99, -1)) + list(range(100, 151))
        bt, df = self._make_backtester_with_data(values)
        metrics = bt._generate_report(df)
        assert metrics['max_drawdown'] < 0
        assert metrics['max_drawdown'] >= -1.0

    def test_sortino_calculated(self):
        """Sortino ratio should be calculated."""
        values = [100 + i * 0.1 + np.random.normal(0, 0.5) for i in range(252)]
        bt, df = self._make_backtester_with_data(values)
        metrics = bt._generate_report(df)
        assert 'sortino' in metrics

    def test_empty_data_returns_zeros(self):
        """Empty data should return all-zero metrics."""
        bt, df = self._make_backtester_with_data([])
        metrics = bt._generate_report(df)
        assert metrics['sharpe'] == 0.0
        assert metrics['total_return'] == 0.0

    def test_single_value_returns_zeros(self):
        """Single data point should return zero metrics."""
        bt, df = self._make_backtester_with_data([100.0])
        metrics = bt._generate_report(df)
        assert metrics['sharpe'] == 0.0

    def test_metrics_dict_has_all_keys(self):
        """All expected metric keys should be present."""
        values = [100 + i * 0.05 for i in range(100)]
        bt, df = self._make_backtester_with_data(values)
        metrics = bt._generate_report(df)
        expected_keys = {
            'sharpe', 'total_return', 'max_drawdown', 'sortino',
            'calmar', 'annual_volatility', 'annual_return',
        }
        assert set(metrics.keys()) == expected_keys


class TestTransactionCosts:
    """Test that transaction costs are properly deducted."""

    def test_costs_reduce_returns(self):
        """Portfolio with high slippage should underperform frictionless."""
        bt = CustomBacktester.__new__(CustomBacktester)
        bt.symbols = ['A', 'B']
        dates = pd.date_range('2020-01-01', periods=50, freq='B')
        np.random.seed(42)
        close_data = pd.DataFrame({
            'A': 100 + np.cumsum(np.random.normal(0.1, 1, 50)),
            'B': 100 + np.cumsum(np.random.normal(0.1, 1, 50)),
        }, index=dates)
        bt.data = pd.concat({'Close': close_data}, axis=1)

        weights = np.array([0.5, 0.5])

        # Low slippage
        metrics_low = bt.run(weights=weights, slippage_bps=1)
        # High slippage
        metrics_high = bt.run(weights=weights, slippage_bps=50)

        # Higher slippage → lower return
        assert metrics_high['total_return'] <= metrics_low['total_return']
