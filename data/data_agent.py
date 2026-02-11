"""
Data Agent Module for Chloe AI
Handles data collection from exchanges and market data sources
"""

import asyncio
import logging
import ccxt
import yfinance as yf
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DataAgent:
    """
    Data agent responsible for collecting market data from various sources
    """
    
    def __init__(self):
        self.exchanges = {}
        self._setup_exchanges()
        
    def _setup_exchanges(self):
        """Initialize exchange connections"""
        try:
            # Initialize major exchanges with rate limiting
            self.exchanges['binance'] = ccxt.binance({
                'enableRateLimit': True,
                'options': {'adjustForTimeDifference': True}
            })
            
            self.exchanges['coinbase'] = ccxt.coinbase({
                'enableRateLimit': True
            })
            
            logger.info("✅ Exchanges initialized successfully")
        except Exception as e:
            logger.error(f"❌ Error initializing exchanges: {e}")
    
    async def fetch_crypto_ohlcv(
        self, 
        symbol: str, 
        timeframe: str = '1d', 
        limit: int = 365,
        exchange_name: str = 'binance'
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for cryptocurrency
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe ('1m', '5m', '1h', '1d', etc.)
            limit: Number of candles to fetch
            exchange_name: Exchange to fetch from
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            exchange = self.exchanges.get(exchange_name)
            if not exchange:
                logger.error(f"Exchange {exchange_name} not found")
                return None
                
            # Fetch OHLCV data
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"✅ Fetched {len(df)} {timeframe} candles for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error fetching crypto data for {symbol}: {e}")
            return None
    
    async def fetch_stock_ohlcv(
        self, 
        symbol: str, 
        period: str = '1y',
        interval: str = '1d'
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for stock
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=interval)
            
            if hist.empty:
                logger.warning(f"No data found for stock {symbol}")
                return None
            
            logger.info(f"✅ Fetched {len(hist)} {interval} candles for stock {symbol}")
            return hist
            
        except Exception as e:
            logger.error(f"❌ Error fetching stock data for {symbol}: {e}")
            return None
    
    async def save_data(self, df: pd.DataFrame, filename: str, format: str = 'parquet'):
        """
        Save data to file
        
        Args:
            df: DataFrame to save
            filename: Output filename (without extension)
            format: Format to save ('parquet', 'csv')
        """
        try:
            filepath = f"data/{filename}"
            
            if format == 'parquet':
                df.to_parquet(f"{filepath}.parquet")
            elif format == 'csv':
                df.to_csv(f"{filepath}.csv")
                
            logger.info(f"✅ Data saved to {filepath}.{format}")
        except Exception as e:
            logger.error(f"❌ Error saving data: {e}")
    
    async def load_data(self, filename: str, format: str = 'parquet') -> Optional[pd.DataFrame]:
        """
        Load data from file
        
        Args:
            filename: Filename to load (without extension)
            format: Format to load ('parquet', 'csv')
            
        Returns:
            DataFrame with loaded data
        """
        try:
            filepath = f"data/{filename}"
            
            if format == 'parquet':
                df = pd.read_parquet(f"{filepath}.parquet")
            elif format == 'csv':
                df = pd.read_csv(f"{filepath}.csv", index_col=0, parse_dates=True)
                
            logger.info(f"✅ Data loaded from {filepath}.{format}")
            return df
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            return None

# Example usage
async def main():
    """Example usage of the Data Agent"""
    agent = DataAgent()
    
    # Fetch Bitcoin data
    btc_data = await agent.fetch_crypto_ohlcv('BTC/USDT', timeframe='1d', limit=365)
    if btc_data is not None:
        print(f"BTC Data shape: {btc_data.shape}")
        print(btc_data.head())
        
        # Save data
        await agent.save_data(btc_data, 'btc_daily', 'parquet')
    
    # Fetch Ethereum data
    eth_data = await agent.fetch_crypto_ohlcv('ETH/USDT', timeframe='1d', limit=365)
    if eth_data is not None:
        print(f"ETH Data shape: {eth_data.shape}")
        await agent.save_data(eth_data, 'eth_daily', 'parquet')
    
    # Fetch Apple stock data
    aapl_data = await agent.fetch_stock_ohlcv('AAPL', period='1y', interval='1d')
    if aapl_data is not None:
        print(f"AAPL Data shape: {aapl_data.shape}")
        await agent.save_data(aapl_data, 'aapl_daily', 'parquet')

if __name__ == "__main__":
    asyncio.run(main())