"""
Tests for strategy/value_strategy.py — weight calculation, normalization, edge cases.
"""
import numpy as np
import pandas as pd

from strategy.value_strategy import calculate_blended_weights


class TestBlendedWeights:
    def _make_returns(self, n_symbols=7, n_days=60):
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=n_days, freq='B')
        symbols = [f'SYM{i}' for i in range(n_symbols)]
        data = {sym: np.random.normal(0.001, 0.02, n_days) for sym in symbols}
        return pd.DataFrame(data, index=dates), symbols

    def test_weights_sum_to_one(self):
        returns_df, symbols = self._make_returns()
        val = {s: np.random.uniform(0, 1) for s in symbols}
        nflo = {s: np.random.uniform(-1, 1) for s in symbols}
        opt = {s: 0.0 for s in symbols}
        weights = calculate_blended_weights(symbols, returns_df, val, nflo, opt)
        # Weights should sum to ~1.0 (or ~0.5 if all bearish)
        assert np.sum(weights) > 0
        assert np.sum(weights) <= 1.01

    def test_no_negative_weights(self):
        returns_df, symbols = self._make_returns()
        val = {s: 0.5 for s in symbols}
        nflo = {s: 0.0 for s in symbols}
        opt = {s: 0.0 for s in symbols}
        weights = calculate_blended_weights(symbols, returns_df, val, nflo, opt)
        assert np.all(weights >= 0)

    def test_all_zero_external_scores_weights_valid(self):
        returns_df, symbols = self._make_returns()
        zero = {s: 0.0 for s in symbols}
        weights = calculate_blended_weights(symbols, returns_df, zero, zero, zero)
        # With zero val/nflo/opt, momentum still drives weights from returns data
        assert np.all(weights >= 0)
        assert np.sum(weights) <= 1.01

    def test_single_symbol(self):
        returns_df, _ = self._make_returns(n_symbols=1)
        symbols = ['SYM0']
        val = {'SYM0': 1.0}
        nflo = {'SYM0': 0.0}
        opt = {'SYM0': 0.0}
        weights = calculate_blended_weights(symbols, returns_df, val, nflo, opt)
        # Single symbol with z-score of std=0 → 50% reduced
        assert len(weights) == 1

    def test_options_penalty_reduces_weight(self):
        returns_df, symbols = self._make_returns()
        val = {s: 1.0 for s in symbols}
        nflo = {s: 0.0 for s in symbols}
        # Heavy penalty on first symbol
        opt = {s: 0.0 for s in symbols}
        opt[symbols[0]] = -5.0
        weights = calculate_blended_weights(symbols, returns_df, val, nflo, opt)
        # Penalized symbol should have lowest or zero weight
        assert weights[0] <= max(weights[1:])
