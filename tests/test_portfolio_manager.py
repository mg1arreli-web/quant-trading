"""
Tests for agents/portfolio_manager.py — Kelly criterion, Markowitz, trailing stop.
"""
import numpy as np
import pandas as pd
import pytest

from agents.portfolio_manager import PortfolioManager


class TestKellyCriterion:
    def test_normal_case(self):
        pm = PortfolioManager()
        # win_rate=0.6, WLR=1.0 → 0.6 - 0.4/1.0 = 0.2
        assert pytest.approx(pm.calculate_kelly_fraction(0.6, 1.0)) == 0.2

    def test_negative_kelly_bounded_to_zero(self):
        pm = PortfolioManager()
        # win_rate=0.4, WLR=1.0 → 0.4 - 0.6/1.0 = -0.2 → 0.0
        assert pm.calculate_kelly_fraction(0.4, 1.0) == 0.0

    def test_exceeds_cap(self):
        pm = PortfolioManager()
        # Impossible win_rate for math test: 1.5, WLR=1 → 1.5 - (-0.5)/1 = 2.0 → capped at 1.0
        assert pm.calculate_kelly_fraction(1.5, 1.0) == 1.0

    def test_zero_wlr(self):
        pm = PortfolioManager()
        assert pm.calculate_kelly_fraction(0.6, 0.0) == 0.0

    def test_negative_wlr(self):
        pm = PortfolioManager()
        assert pm.calculate_kelly_fraction(0.6, -1.0) == 0.0


class TestMarkowitzWeights:
    def test_weights_sum_to_one(self):
        pm = PortfolioManager()
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=60, freq='B')
        returns_df = pd.DataFrame({
            'A': np.random.normal(0.001, 0.02, 60),
            'B': np.random.normal(0.002, 0.025, 60),
            'C': np.random.normal(0.0005, 0.015, 60),
        }, index=dates)
        weights = pm.calculate_markowitz_weights(returns_df)
        assert pytest.approx(np.sum(weights), abs=0.01) == 1.0

    def test_weights_non_negative(self):
        pm = PortfolioManager()
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=60, freq='B')
        returns_df = pd.DataFrame({
            'A': np.random.normal(0.001, 0.02, 60),
            'B': np.random.normal(0.002, 0.025, 60),
        }, index=dates)
        weights = pm.calculate_markowitz_weights(returns_df)
        assert np.all(weights >= -0.01)  # Allow tiny float error

    def test_empty_returns(self):
        pm = PortfolioManager()
        returns_df = pd.DataFrame()
        weights = pm.calculate_markowitz_weights(returns_df)
        assert len(weights) == 0


class TestTrailingStopLoss:
    def test_stop_loss_triggers(self):
        pm = PortfolioManager()
        pm.high_water_marks = {'AAPL': 200.0}
        # Price dropped 15% from HWM (threshold is 10%)
        triggered = pm.check_stop_loss({'AAPL': 170.0})
        assert triggered['AAPL'] is True

    def test_stop_loss_not_triggered(self):
        pm = PortfolioManager()
        pm.high_water_marks = {'AAPL': 200.0}
        # Price only dropped 5% (within threshold)
        triggered = pm.check_stop_loss({'AAPL': 195.0})
        assert triggered['AAPL'] is False

    def test_no_hwm_no_trigger(self):
        pm = PortfolioManager()
        pm.high_water_marks = {}
        triggered = pm.check_stop_loss({'AAPL': 150.0})
        assert triggered['AAPL'] is False


class TestScaleWeightsByKelly:
    def test_scaling(self):
        pm = PortfolioManager()
        weights = np.array([0.5, 0.5])
        scaled = pm.scale_weights_by_kelly(weights, win_rate=0.6, win_loss_ratio=1.0)
        # Kelly = 0.2, so weights should be 0.5 * 0.2 = 0.1
        np.testing.assert_array_almost_equal(scaled, [0.1, 0.1])
