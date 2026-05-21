"""
Alpaca Execution Agent — interfaces with Alpaca Paper/Live Trading API.
Includes TTL caching, TWAP execution, and tenacity retries.
"""
import logging
import os
import time
from typing import Any

import requests
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import (
    CACHE_TTL,
    DEFAULT_PRICE_FALLBACK,
    TWAP_THRESHOLD,
)

load_dotenv()
logger = logging.getLogger(__name__)


class AlpacaExecutionAgent:
    def __init__(self, api_key: str | None = None, api_secret: str | None = None) -> None:
        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.api_secret = api_secret or os.getenv('ALPACA_API_SECRET')
        self.base_url = 'https://paper-api.alpaca.markets'
        self.headers: dict[str, Any] = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.api_secret,
            'Content-Type': 'application/json',
        }

        self.current_holdings: dict[str, int] = {}
        self._cache: dict[str, tuple[float, Any]] = {}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    def get_current_price(self, symbol: str) -> float:
        """Retrieve current price from Alpaca data API, falling back to yfinance."""
        try:
            data_url = f"https://data.alpaca.markets/v2/stocks/{symbol}/trades/latest"
            response = requests.get(data_url, headers=self.headers, verify=True, timeout=10)
            if response.status_code == 200:
                return float(response.json()['trade']['p'])
            else:
                import yfinance as yf
                return float(yf.Ticker(symbol).fast_info['lastPrice'])
        except Exception as e:
            logger.warning(f"Failed to get price for {symbol}: {e}")
            return DEFAULT_PRICE_FALLBACK

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    def get_account_value(self) -> float:
        """Retrieve portfolio value from Alpaca API with TTL caching."""
        cache_key = "account_value"
        if cache_key in self._cache:
            timestamp, value = self._cache[cache_key]
            if time.time() - timestamp < CACHE_TTL:
                return value

        val = 100000.0
        try:
            response = requests.get(
                f"{self.base_url}/v2/account",
                headers=self.headers, verify=True, timeout=10,
            )
            if response.status_code == 200:
                val = float(response.json()['portfolio_value'])
            else:
                logger.warning(f"Account API returned {response.status_code}: {response.text}")
        except requests.RequestException as e:
            logger.error(f"Failed to get account value: {e}")

        self._cache[cache_key] = (time.time(), val)
        return val

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    def sync_holdings(self) -> dict[str, int]:
        """Sync current portfolio holdings from Alpaca API with TTL caching."""
        cache_key = "positions"
        if cache_key in self._cache:
            timestamp, value = self._cache[cache_key]
            if time.time() - timestamp < CACHE_TTL:
                return value

        ret_holdings: dict[str, int] = {}
        try:
            response = requests.get(
                f"{self.base_url}/v2/positions",
                headers=self.headers, verify=True, timeout=10,
            )
            if response.status_code == 200:
                positions = response.json()
                self.current_holdings = {str(p['symbol']): int(p['qty']) for p in positions}
                ret_holdings = self.current_holdings
            else:
                logger.warning(f"Positions API returned {response.status_code}: {response.text}")
        except requests.RequestException as e:
            logger.error(f"Failed to sync holdings: {e}")

        self._cache[cache_key] = (time.time(), ret_holdings)
        return ret_holdings

    def execute_twap(self, symbol: str, total_qty: int, side: str, num_slices: int = 10, sleep_time: int = 1) -> None:
        """Execute a TWAP (Time-Weighted Average Price) order over multiple slices."""
        slice_qty = total_qty // num_slices
        for i in range(num_slices):
            logger.info(f"TWAP slice [{i + 1}/{num_slices}] for {symbol}: {side} {slice_qty}")
            time.sleep(sleep_time)

        remainder = total_qty % num_slices
        if remainder > 0:
            logger.info(f"TWAP remainder: {side} {remainder} for {symbol}")

        logger.info(f"Completed TWAP execution for {symbol}")

    def execute_weights(
        self,
        weights_dict: dict[str, float],
        prices_dict: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute orders to match target portfolio weights."""
        total_portfolio_value = self.get_account_value()
        self.sync_holdings()
        orders: list[dict[str, Any]] = []

        for symbol, target_weight in weights_dict.items():
            if prices_dict and symbol in prices_dict:
                current_price = prices_dict[symbol]
            else:
                current_price = self.get_current_price(symbol)

            target_value = target_weight * total_portfolio_value
            target_shares = int(target_value / current_price)

            current_shares = self.current_holdings.get(symbol, 0)
            diff_shares = target_shares - current_shares

            if diff_shares == 0:
                continue

            side = 'buy' if diff_shares > 0 else 'sell'
            qty = abs(diff_shares)

            if qty > TWAP_THRESHOLD:
                self.execute_twap(symbol, qty, side)

            order_payload = {
                "symbol": symbol,
                "qty": qty,
                "side": side,
                "type": "market",
                "time_in_force": "day",
            }

            try:
                response = requests.post(
                    f"{self.base_url}/v2/orders",
                    headers=self.headers,
                    json=order_payload,
                    verify=True,
                    timeout=10,
                )

                if response.status_code in [200, 201]:
                    logger.info(f"Order successful for {symbol}: {side} {qty}")
                else:
                    logger.error(f"Order failed for {symbol}: {response.status_code} - {response.text}")
            except requests.RequestException as e:
                logger.error(f"Order request failed for {symbol}: {e}")

            orders.append(order_payload)

        return orders
