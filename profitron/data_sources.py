"""Module for data sources to be used with the prediction manager."""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from .live_data_fetcher import LiveDataFetcher
from binance.client import Client

class BinanceDataSource:
    """Data source for fetching price data from Binance API."""

    def __init__(self, symbol: str = 'ETHUSDT', client: Optional[Client] = None):
        """Initialize BinanceDataSource with symbol and optional client.

        Args:
            symbol: Trading pair symbol (default: ETHUSDT)
            client: Optional Binance client instance
        """
        self.symbol = symbol
        self.fetcher = LiveDataFetcher(client)
        self.last_update = None

    def get_live_data(self) -> Dict:
        """Get current market data.

        Returns:
            Dictionary with OHLCV data for the current price
        """
        price = self.fetcher.get_latest_price(self.symbol)
        self.last_update = datetime.now()
        
        # Return in OHLCV format for compatibility
        return {
            'datetime': self.last_update,
            'open': price,
            'high': price,
            'low': price,
            'close': price,
            'volume': 0  # Not available for current price
        }

    def get_historical_data(self, past_hours: int = 24) -> List[Dict]:
        """Get historical OHLCV data.

        Args:
            past_hours: Number of hours of historical data to fetch (default: 24)

        Returns:
            List of dictionaries containing OHLCV data
        """
        # Determine appropriate interval based on past_hours
        if past_hours <= 24:
            interval = '1h'
        elif past_hours <= 168:  # 7 days
            interval = '4h'
        else:
            interval = '1d'
            
        # Calculate limit to ensure we get enough data
        limit = min(1000, past_hours + 10)  # Add buffer, cap at API limit
        
        return self.fetcher.get_latest_candles(
            symbol=self.symbol,
            interval=interval,
            limit=limit
        )