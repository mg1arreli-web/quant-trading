"""
Portfolio Manager — Markowitz optimization, Kelly criterion, and trailing stop logic.
"""
import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config.settings import KELLY_FRACTION_CAP, TRAILING_STOP_LOSS_PCT
from utils.state_manager import StateManager

logger = logging.getLogger(__name__)


class PortfolioManager:
    def __init__(self) -> None:
        self.kelly_cap: float = float(KELLY_FRACTION_CAP)
        self.trailing_stop_loss_pct: float = float(TRAILING_STOP_LOSS_PCT)

        self.state_manager = StateManager()
        state = self.state_manager.load_state()
        self.high_water_marks: dict[str, float] = state.get('high_water_marks', {})

    def update_trailing_stop_loss(self, current_prices: dict[str, float]) -> None:
        """Update trailing stop loss high-water marks for active positions."""
        updated = False
        for ticker, price in current_prices.items():
            if ticker not in self.high_water_marks or price > self.high_water_marks[ticker]:
                self.high_water_marks[ticker] = price
                updated = True

        if updated:
            self.state_manager.save_state({'high_water_marks': self.high_water_marks})

    def check_stop_loss(self, current_prices: dict[str, float]) -> dict[str, bool]:
        """
        Check which symbols have triggered their trailing stop loss.
        Returns a dict of {symbol: triggered_bool}.
        """
        triggered: dict[str, bool] = {}
        for symbol, price in current_prices.items():
            hwm = self.high_water_marks.get(symbol)
            if hwm is not None:
                stop_price = hwm * (1 - self.trailing_stop_loss_pct)
                is_triggered = price < stop_price
                if is_triggered:
                    logger.warning(
                        f"STOP LOSS triggered for {symbol}: "
                        f"price={price:.2f}, HWM={hwm:.2f}, stop={stop_price:.2f}"
                    )
                triggered[symbol] = is_triggered
            else:
                triggered[symbol] = False
        return triggered

    def calculate_markowitz_weights(self, returns_df: pd.DataFrame) -> np.ndarray:
        """Calculate optimal portfolio weights using Markowitz max-Sharpe optimization."""
        cov_matrix = returns_df.cov().values * 252
        expected_returns = returns_df.mean().values * 252
        num_assets = len(cov_matrix)

        if num_assets == 0:
            return np.array([])

        init_guess = np.ones(num_assets) / num_assets

        def objective(weights: np.ndarray) -> float:
            p_var = weights.T @ cov_matrix @ weights
            p_ret = weights @ expected_returns
            return float(-1 * (p_ret / np.sqrt(p_var + 1e-9)))

        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0.0, 1.0) for _ in range(num_assets))

        res = minimize(objective, init_guess, method='SLSQP', bounds=bounds, constraints=cons)

        if res.success:
            return res.x
        else:
            logger.error(f"Markowitz optimization failed: {res.message}")
            return init_guess

    def calculate_kelly_fraction(self, win_rate: float, win_loss_ratio: float) -> float:
        """Calculate the Kelly Criterion fraction, capped at kelly_cap."""
        if win_loss_ratio <= 0:
            return 0.0
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
        return float(max(0.0, min(self.kelly_cap, kelly)))

    def scale_weights_by_kelly(
        self, weights: np.ndarray, win_rate: float, win_loss_ratio: float,
    ) -> np.ndarray:
        """Scale portfolio weights by Kelly fraction for position sizing."""
        kelly_f = self.calculate_kelly_fraction(win_rate, win_loss_ratio)
        logger.info(f"Kelly fraction: {kelly_f:.4f} (win_rate={win_rate:.2f}, WLR={win_loss_ratio:.2f})")
        return weights * kelly_f
