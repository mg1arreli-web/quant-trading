"""
Main Backtest Entry Point — runs the full HMM-aware backtest pipeline.
"""
import logging

import numpy as np
import pandas as pd

from agents.macro_risk_manager import MacroRegimeSwitch
from backtester.engine import CustomBacktester
from config.logging_config import setup_logging
from config.settings import BACKTEST_END, BACKTEST_START, SYMBOLS
from strategy.value_strategy import calculate_blended_weights

setup_logging()
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    symbols = SYMBOLS

    logger.info(f"Running backtest for {symbols} from {BACKTEST_START} to {BACKTEST_END}")
    bt = CustomBacktester(symbols=symbols, start_date=BACKTEST_START, end_date=BACKTEST_END)

    close_prices = bt.data['Close'].dropna(axis=1, how='all')
    returns_df = close_prices.ffill().pct_change().dropna(how='all')

    # Use only symbols that actually have data
    available_symbols = list(close_prices.columns)
    if len(available_symbols) < len(symbols):
        missing = set(symbols) - set(available_symbols)
        logger.warning(f"Missing data for symbols: {missing}. Proceeding with {available_symbols}")

    # Fit HMM regime model
    macro_manager = MacroRegimeSwitch()
    logger.info("Fitting Macro HMM...")
    macro_manager.fit()

    states = macro_manager.model.predict(macro_manager.data[['VIX_Ret', 'TNX_Ret']].values)
    risk_states = [macro_manager.map_regime_to_risk(s) for s in states]
    macro_risk_df = pd.DataFrame({'Risk_State': risk_states}, index=macro_manager.data.index)

    # Compute base weights using available symbols
    val_mock = {sym: 0.0 for sym in available_symbols}
    nflo_mock = {sym: 0.0 for sym in available_symbols}
    opt_mock = {sym: 0.0 for sym in available_symbols}
    optimal_weights = calculate_blended_weights(available_symbols, returns_df, val_mock, nflo_mock, opt_mock)

    # Replace NaN weights with 0
    optimal_weights = np.nan_to_num(optimal_weights, nan=0.0)
    # Re-normalize if weights don't sum to ~1
    weight_sum = np.sum(optimal_weights)
    if weight_sum > 0:
        optimal_weights = optimal_weights / weight_sum

    logger.info("Optimal Blended Allocations (Base):")
    for sym, weight in zip(available_symbols, optimal_weights):
        logger.info(f"  {sym}: {weight:.4f}")

    # Build time-varying weight matrix aligned with close_prices columns
    weights_df = pd.DataFrame(index=close_prices.index, columns=close_prices.columns, dtype=float)
    for i, col in enumerate(close_prices.columns):
        if i < len(optimal_weights):
            weights_df[col] = optimal_weights[i]
        else:
            weights_df[col] = 0.0

    # Apply regime overlay
    risk_off_days = 0
    for date in weights_df.index:
        state = "Risk-On"
        available_dates = macro_risk_df.index[macro_risk_df.index <= date]
        if len(available_dates) > 0:
            state = macro_risk_df.loc[available_dates[-1], 'Risk_State']

        if state == "Risk-Off":
            weights_df.loc[date] = 0.0
            risk_off_days += 1

    logger.info(f"Total Risk-Off days during backtest: {risk_off_days}")

    # Run backtest with transaction costs
    metrics = bt.run(weights=weights_df)

    # Print comprehensive results
    logger.info("=" * 60)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Total Return:      {metrics['total_return']:>10.2%}")
    logger.info(f"  Annual Return:     {metrics['annual_return']:>10.2%}")
    logger.info(f"  Sharpe Ratio:      {metrics['sharpe']:>10.2f}")
    logger.info(f"  Sortino Ratio:     {metrics['sortino']:>10.2f}")
    logger.info(f"  Calmar Ratio:      {metrics['calmar']:>10.2f}")
    logger.info(f"  Max Drawdown:      {metrics['max_drawdown']:>10.2%}")
    logger.info(f"  Annual Volatility: {metrics['annual_volatility']:>10.2%}")
    logger.info("=" * 60)
