"""
Tests for broker/alpaca.py — mocked HTTP calls, TWAP, share calculations.
"""
from unittest.mock import MagicMock, patch

from broker.alpaca import AlpacaExecutionAgent


class TestAlpacaExecutionAgent:

    def _make_agent(self):
        return AlpacaExecutionAgent(api_key='test_key', api_secret='test_secret')

    @patch('broker.alpaca.requests.get')
    def test_get_current_price_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'trade': {'p': 150.25}}
        mock_get.return_value = mock_response

        agent = self._make_agent()
        price = agent.get_current_price('AAPL')
        assert price == 150.25

    @patch('broker.alpaca.requests.get')
    def test_get_account_value_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'portfolio_value': '250000.50'}
        mock_get.return_value = mock_response

        agent = self._make_agent()
        agent._cache.clear()
        val = agent.get_account_value()
        assert val == 250000.50

    @patch('broker.alpaca.requests.get')
    def test_get_account_value_uses_cache(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'portfolio_value': '250000.50'}
        mock_get.return_value = mock_response

        agent = self._make_agent()
        agent._cache.clear()
        val1 = agent.get_account_value()
        val2 = agent.get_account_value()
        # Second call should use cache (only 1 HTTP call)
        assert val1 == val2
        assert mock_get.call_count == 1

    @patch('broker.alpaca.requests.get')
    def test_sync_holdings_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {'symbol': 'AAPL', 'qty': '100'},
            {'symbol': 'MSFT', 'qty': '50'},
        ]
        mock_get.return_value = mock_response

        agent = self._make_agent()
        agent._cache.clear()
        holdings = agent.sync_holdings()
        assert holdings == {'AAPL': 100, 'MSFT': 50}

    @patch('broker.alpaca.requests.post')
    @patch('broker.alpaca.requests.get')
    def test_execute_weights_places_orders(self, mock_get, mock_post):
        # Mock account value
        account_response = MagicMock()
        account_response.status_code = 200
        account_response.json.return_value = {'portfolio_value': '100000'}

        # Mock positions (empty)
        positions_response = MagicMock()
        positions_response.status_code = 200
        positions_response.json.return_value = []

        mock_get.side_effect = [account_response, positions_response]

        # Mock order placement
        order_response = MagicMock()
        order_response.status_code = 201
        mock_post.return_value = order_response

        agent = self._make_agent()
        agent._cache.clear()
        orders = agent.execute_weights(
            {'AAPL': 0.5, 'MSFT': 0.5},
            prices_dict={'AAPL': 150.0, 'MSFT': 300.0},
        )
        assert len(orders) == 2
        assert all(o['side'] == 'buy' for o in orders)

    @patch('broker.alpaca.requests.get')
    def test_api_error_returns_default(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = 'Unauthorized'
        mock_get.return_value = mock_response

        agent = self._make_agent()
        agent._cache.clear()
        val = agent.get_account_value()
        # Should fall back to default 100000.0
        assert val == 100000.0
