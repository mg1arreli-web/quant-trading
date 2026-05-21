"""
SEC Earnings Transcript Fetcher — fetches quarterly earnings call transcripts from FMP API.
"""
import logging
import os

import requests

logger = logging.getLogger(__name__)


def fetch_earnings_transcript(symbol: str, year: int = 2024) -> str:
    """Fetch earnings call transcript text for a symbol from FMP API."""
    api_key = os.getenv('FMP_API_KEY')
    if not api_key:
        logger.error("FMP_API_KEY environment variable is not set")
        return ""

    url = f"https://financialmodelingprep.com/api/v4/batch_earning_call_transcript/{symbol}?year={year}&apikey={api_key}"
    try:
        response = requests.get(url, verify=True, timeout=30)
        response.raise_for_status()
        data = response.json()

        if isinstance(data, list) and len(data) > 0:
            if 'content' in data[0]:
                return data[0]['content']
            elif 'transcript' in data[0]:
                return data[0]['transcript']

        return ""
    except requests.RequestException as e:
        logger.error(f"Error fetching transcript for {symbol}: {e}")
        return ""
