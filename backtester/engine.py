"""
Custom Backtester Engine with realistic transaction costs, proper Sharpe ratio,
and comprehensive performance metrics.
"""
import logging
import os

import numpy as np
import pandas as pd
import requests
import requests_cache
import yfinance as yf

from config.settings import (
    COMMISSION_MINIMUM,
    COMMISSION_PER_SHARE,
    RISK_FREE_RATE,
    SLIPPAGE_BPS,
)

logger = logging.getLogger(__name__)

session = requests_cache.CachedSession('yfinance.cache')
session.headers['User-agent'] = 'quant-trading/1.0'


class CustomBacktester:
    def __init__(self, symbols: list[str], start_date: str, end_date: str):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.data: pd.DataFrame = pd.DataFrame()
        logger.info(f"Downloading data for {symbols}...")
        self._preload_data(symbols, start_date, end_date)
        self.data = self.data.ffill().bfill()

    def _preload_data(self, symbols: list[str], start_date: str, end_date: str) -> None:
        dfs: dict[str, pd.DataFrame] = {}
        av_failed = False

        av_key = os.getenv('ALPHAVANTAGE_API_KEY')
        if av_key:
            logger.info("Attempting to fetch data via AlphaVantage...")
            for symbol in symbols:
                url = (
                    f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY"
                    f"&symbol={symbol}&apikey={av_key}&outputsize=compact"
                )
                try:
                    r = requests.get(url, verify=True, timeout=30)
                    data = r.json()
                    if "Time Series (Daily)" not in data:
                        logger.warning(f"AlphaVantage failed for {symbol}: {data.get('Information', data)}")
                        av_failed = True
                        break

                    df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
                    df = df.rename(columns={
                        '1. open': 'Open', '2. high': 'High',
                        '3. low': 'Low', '4. close': 'Close',
                        '5. volume': 'Volume',
                    })
                    df = df.astype(float)
                    df.index = pd.to_datetime(df.index)
                    df = df.sort_index()
                    df = df.loc[start_date:end_date]
                    dfs[symbol] = df
                except Exception as e:
                    logger.warning(f"AlphaVantage exception for {symbol}: {e}")
                    av_failed = True
                    break

            if not av_failed and len(dfs) == len(symbols):
                if len(symbols) > 1:
                    self.data = pd.concat(dfs.values(), axis=1, keys=dfs.keys())
                    self.data = self.data.swaplevel(0, 1, axis=1).sort_index(axis=1)
                else:
                    self.data = dfs[symbols[0]]
                logger.info("AlphaVantage data loaded successfully.")
                return
        else:
            logger.info("ALPHAVANTAGE_API_KEY not set, skipping AlphaVantage.")

        logger.info("Falling back to yfinance...")
        try:
            self.data = yf.download(symbols, start=start_date, end=end_date)
        except Exception:
            self.data = yf.download(symbols, start=start_date, end=end_date)

    def get_last_price(self, symbol: str | None = None) -> float:
        if symbol is not None and isinstance(self.data.columns, pd.MultiIndex):
            return float(self.data['Close'][symbol].iloc[-1])
        return float(self.data['Close'].iloc[-1])

    def _generate_report(self, df: pd.DataFrame) -> dict:
        """Generate comprehensive performance metrics."""
        if 'portfolio_value' not in df.columns:
            return self._empty_metrics()

        daily_returns = df['portfolio_value'].pct_change().dropna()

        if len(daily_returns) == 0:
            return self._empty_metrics()

        std_dev = np.std(daily_returns)
        if std_dev == 0:
            return self._empty_metrics()

        # Risk-free rate (daily)
        rf_daily = (1 + RISK_FREE_RATE) ** (1 / 252) - 1

        # Sharpe Ratio (with risk-free rate)
        sharpe = (np.mean(daily_returns) - rf_daily) / std_dev * np.sqrt(252)

        # Total Return
        total_return = (1 + daily_returns).prod() - 1

        # Max Drawdown
        cummax = df['portfolio_value'].cummax()
        drawdown = (df['portfolio_value'] - cummax) / cummax
        max_drawdown = float(drawdown.min())

        # Annual Volatility
        annual_vol = float(std_dev * np.sqrt(252))

        # Sortino Ratio (downside deviation)
        downside_returns = daily_returns[daily_returns < rf_daily]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else std_dev
        sortino = float((np.mean(daily_returns) - rf_daily) / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0

        # Calmar Ratio
        years = len(daily_returns) / 252
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        calmar = float(annual_return / abs(max_drawdown)) if max_drawdown != 0 else 0.0

        return {
            'sharpe': float(sharpe),
            'total_return': float(total_return),
            'max_drawdown': max_drawdown,
            'sortino': sortino,
            'calmar': calmar,
            'annual_volatility': annual_vol,
            'annual_return': float(annual_return),
        }

    @staticmethod
    def _empty_metrics() -> dict:
        return {
            'sharpe': 0.0, 'total_return': 0.0, 'max_drawdown': 0.0,
            'sortino': 0.0, 'calmar': 0.0, 'annual_volatility': 0.0,
            'annual_return': 0.0,
        }

    def run(
        self,
        weights=None,
        initial_capital: float = 100000.0,
        slippage_bps: int | None = None,
        commission_per_share: float | None = None,
        commission_minimum: float | None = None,
    ) -> dict:
        """
        Run the backtest with realistic transaction costs.

        Returns a dict of performance metrics.
        """
        _slippage = slippage_bps if slippage_bps is not None else SLIPPAGE_BPS
        _comm_ps = commission_per_share if commission_per_share is not None else COMMISSION_PER_SHARE
        _comm_min = commission_minimum if commission_minimum is not None else COMMISSION_MINIMUM

        df = pd.DataFrame(index=self.data.index)

        if isinstance(self.data.columns, pd.MultiIndex):
            if weights is not None:
                close_prices = self.data['Close']
                portfolio_values = []
                cash = initial_capital
                num_assets = len(close_prices.columns)
                holdings = np.zeros(num_assets)

                if isinstance(weights, pd.DataFrame):
                    # Align both rows AND columns with close_prices
                    aligned_weights = weights.reindex(
                        index=close_prices.index, columns=close_prices.columns, fill_value=0.0,
                    ).ffill().bfill().fillna(0.0)
                    weights_arr = aligned_weights.values
                elif isinstance(weights, pd.Series):
                    weights_arr = np.tile(weights.values, (len(close_prices), 1))
                elif isinstance(weights, np.ndarray):
                    if weights.ndim == 1:
                        weights_arr = np.tile(weights, (len(close_prices), 1))
                    else:
                        weights_arr = weights
                else:
                    weights_arr = np.tile(np.array(weights), (len(close_prices), 1))

                for i in range(len(close_prices)):
                    current_prices = close_prices.iloc[i].fillna(0.0).values
                    current_target_weights = weights_arr[i] if weights_arr.ndim == 2 else weights_arr

                    # Current portfolio value
                    portfolio_val = cash + np.sum(holdings * current_prices)

                    # Target values
                    target_values = portfolio_val * current_target_weights
                    current_values = holdings * current_prices
                    trade_values = target_values - current_values

                    # Apply slippage
                    slippage_mult_buy = 1.0 + _slippage / 10000.0
                    slippage_mult_sell = 1.0 - _slippage / 10000.0
                    exec_prices = np.where(
                        trade_values > 0,
                        current_prices * slippage_mult_buy,
                        current_prices * slippage_mult_sell,
                    )

                    # Calculate shares to trade
                    shares_to_trade = np.zeros_like(trade_values)
                    mask = exec_prices > 0
                    shares_to_trade[mask] = trade_values[mask] / exec_prices[mask]

                    # Commission
                    abs_shares = np.abs(shares_to_trade)
                    commissions = np.where(
                        abs_shares > 1e-4,
                        np.maximum(_comm_min, _comm_ps * abs_shares),
                        0.0,
                    )

                    # Execute trades
                    cost_of_trades = np.sum(shares_to_trade * exec_prices) + np.sum(commissions)
                    cash -= cost_of_trades
                    holdings += shares_to_trade

                    # End-of-day portfolio value
                    end_of_day_value = cash + np.sum(holdings * current_prices)
                    portfolio_values.append(end_of_day_value)

                df['portfolio_value'] = portfolio_values
            else:
                df['portfolio_value'] = self.data['Close'].sum(axis=1)
        else:
            df['portfolio_value'] = self.data['Close']

        metrics = self._generate_report(df)
        logger.info(
            f"Backtest finished. Total Return: {metrics['total_return']:.2%}, "
            f"Sharpe: {metrics['sharpe']:.2f}, Max DD: {metrics['max_drawdown']:.2%}, "
            f"Sortino: {metrics['sortino']:.2f}, Calmar: {metrics['calmar']:.2f}"
        )
        return metrics
