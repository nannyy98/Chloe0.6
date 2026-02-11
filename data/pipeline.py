"""
Professional Data Pipeline
Modern data ingestion, cleaning, and feature engineering
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import yfinance as yf
import ccxt
import os
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

logger = logging.getLogger(__name__)

class DataStorage:
    """Parquet-based data storage system"""
    
    def __init__(self, data_dir: str = "data/storage"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def save_data(self, df: pd.DataFrame, symbol: str, data_type: str = "ohlcv") -> str:
        """Save DataFrame to parquet with proper partitioning"""
        # Create partitioned directory structure
        partition_path = self.data_dir / data_type / symbol
        partition_path.mkdir(parents=True, exist_ok=True)
        
        # Add metadata
        df_with_meta = df.copy()
        df_with_meta.attrs = {
            'symbol': symbol,
            'data_type': data_type,
            'saved_at': datetime.now().isoformat(),
            'rows': len(df)
        }
        
        # Save to parquet
        filename = f"{data_type}_{symbol}_{datetime.now().strftime('%Y%m%d')}.parquet"
        filepath = partition_path / filename
        
        # Convert to pyarrow and save
        table = pa.Table.from_pandas(df_with_meta)
        pq.write_table(table, filepath)
        
        logger.info(f"💾 Saved {len(df)} rows to {filepath}")
        return str(filepath)
        
    def load_data(self, symbol: str, data_type: str = "ohlcv", 
                  start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        """Load data from parquet storage"""
        partition_path = self.data_dir / data_type / symbol
        
        if not partition_path.exists():
            return None
            
        # Find latest file
        parquet_files = list(partition_path.glob("*.parquet"))
        if not parquet_files:
            return None
            
        latest_file = max(parquet_files, key=os.path.getctime)
        
        # Load and filter by date if specified
        table = pq.read_table(latest_file)
        df = table.to_pandas()
        
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        logger.info(f"📂 Loaded {len(df)} rows from {latest_file}")
        return df

class DataCleaner:
    """Professional data cleaning and validation"""
    
    @staticmethod
    def clean_ohlcv_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean OHLCV data"""
        df_clean = df.copy()
        
        # Remove rows with missing OHLC values
        required_cols = ['open', 'high', 'low', 'close']
        if all(col in df_clean.columns for col in required_cols):
            df_clean = df_clean.dropna(subset=required_cols)
        
        # Ensure high >= low >= close >= open relationships
        if all(col in df_clean.columns for col in ['high', 'low', 'close']):
            # Fix high/low relationships
            df_clean['high'] = df_clean[['high', 'low', 'close']].max(axis=1)
            df_clean['low'] = df_clean[['high', 'low', 'close']].min(axis=1)
            
        # Remove extreme outliers (price changes > 50% in one period)
        if 'close' in df_clean.columns:
            price_changes = df_clean['close'].pct_change().abs()
            df_clean = df_clean[price_changes < 0.5]  # Remove >50% changes
            
        # Forward fill remaining NaN values
        df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')
        
        # Remove duplicate timestamps
        df_clean = df_clean[~df_clean.index.duplicated(keep='first')]
        
        # Sort by timestamp
        df_clean = df_clean.sort_index()
        
        logger.info(f"🧹 Cleaned data: {len(df)} → {len(df_clean)} rows")
        return df_clean
        
    @staticmethod
    def detect_lookahead_bias(df: pd.DataFrame, features: List[str]) -> bool:
        """Detect potential lookahead bias in features"""
        # Check if any feature uses future data
        for feature in features:
            if feature in df.columns:
                # Simple check: feature shouldn't have perfect correlation with future returns
                if 'close' in df.columns and len(df) > 10:
                    future_returns = df['close'].shift(-1).pct_change()
                    correlation = df[feature].corr(future_returns)
                    if abs(correlation) > 0.9:  # Suspiciously high correlation
                        logger.warning(f"⚠️ Potential lookahead bias in {feature}: correlation={correlation:.3f}")
                        return True
        return False

class DataPipeline:
    """Main data pipeline orchestrator"""
    
    def __init__(self, storage_dir: str = "data/storage"):
        self.storage = DataStorage(storage_dir)
        self.cleaner = DataCleaner()
        self.exchanges = {
            'binance': ccxt.binance({'enableRateLimit': True}),
            'coinbase': ccxt.coinbase({'enableRateLimit': True})
        }
        
    async def fetch_crypto_data(self, symbol: str, timeframe: str = '1d', 
                               limit: int = 1000, exchange: str = 'binance') -> Optional[pd.DataFrame]:
        """Fetch cryptocurrency data from exchange"""
        try:
            exchange_obj = self.exchanges.get(exchange)
            if not exchange_obj:
                raise ValueError(f"Exchange {exchange} not supported")
                
            # Fetch OHLCV data
            ohlcv = await asyncio.get_event_loop().run_in_executor(
                None, exchange_obj.fetch_ohlcv, symbol, timeframe, None, limit
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Clean data
            df_clean = self.cleaner.clean_ohlcv_data(df)
            
            # Save to storage
            self.storage.save_data(df_clean, symbol, 'crypto_ohlcv')
            
            logger.info(f"📊 Fetched {len(df_clean)} rows of {symbol} data from {exchange}")
            return df_clean
            
        except Exception as e:
            logger.error(f"❌ Error fetching {symbol} data: {e}")
            return None
            
    def fetch_stock_data(self, symbol: str, period: str = '1y', 
                        interval: str = '1d') -> Optional[pd.DataFrame]:
        """Fetch stock data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                return None
                
            # Clean data
            df_clean = self.cleaner.clean_ohlcv_data(df)
            
            # Save to storage
            self.storage.save_data(df_clean, symbol, 'stock_ohlcv')
            
            logger.info(f"📊 Fetched {len(df_clean)} rows of {symbol} stock data")
            return df_clean
            
        except Exception as e:
            logger.error(f"❌ Error fetching {symbol} stock data: {e}")
            return None
            
    def create_features(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """Create advanced technical features"""
        df_features = df.copy()
        
        # Price-based features
        df_features['returns'] = df_features['close'].pct_change()
        df_features['log_returns'] = np.log(df_features['close'] / df_features['close'].shift(1))
        
        # Moving averages
        for window in [5, 10, 20, 50, 200]:
            df_features[f'sma_{window}'] = df_features['close'].rolling(window=window).mean()
            df_features[f'ema_{window}'] = df_features['close'].ewm(span=window).mean()
            
        # Volatility features
        df_features['volatility_20'] = df_features['returns'].rolling(window=20).std()
        df_features['volatility_60'] = df_features['returns'].rolling(window=60).std()
        
        # RSI
        df_features['rsi_14'] = self._calculate_rsi(df_features['close'], 14)
        
        # MACD
        df_features['macd_line'], df_features['macd_signal'], df_features['macd_histogram'] = \
            self._calculate_macd(df_features['close'])
            
        # Bollinger Bands
        df_features['bb_upper'], df_features['bb_lower'], df_features['bb_percent'] = \
            self._calculate_bollinger_bands(df_features['close'])
            
        # Volume features
        df_features['volume_sma_20'] = df_features['volume'].rolling(window=20).mean()
        df_features['volume_ratio'] = df_features['volume'] / df_features['volume_sma_20']
        
        # Momentum features
        for period in [1, 3, 5, 10]:
            df_features[f'momentum_{period}'] = df_features['close'] / df_features['close'].shift(period) - 1
            
        # Price position features
        df_features['price_position_20'] = (df_features['close'] - df_features['low'].rolling(20).min()) / \
                                          (df_features['high'].rolling(20).max() - df_features['low'].rolling(20).min())
                                          
        # Lagged features (for ML models)
        for lag in [1, 2, 3, 5]:
            df_features[f'close_lag_{lag}'] = df_features['close'].shift(lag)
            df_features[f'volume_lag_{lag}'] = df_features['volume'].shift(lag)
            
        # Target variables (future returns - ensure no lookahead bias)
        for horizon in [1, 3, 5, 10]:
            df_features[f'target_{horizon}'] = df_features['close'].shift(-horizon) / df_features['close'] - 1
            
        logger.info(f"⚙️ Created {len([col for col in df_features.columns if col not in df.columns])} features")
        return df_features
        
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, 
                       signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal).mean()
        macd_histogram = macd_line - macd_signal
        return macd_line, macd_signal, macd_histogram
        
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, 
                                  num_std: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        bb_percent = (prices - lower_band) / (upper_band - lower_band)
        return upper_band, lower_band, bb_percent
        
    def get_data_for_analysis(self, symbol: str, data_type: str = 'crypto_ohlcv',
                            start_date: str = None, end_date: str = None,
                            create_features: bool = True) -> Optional[pd.DataFrame]:
        """Get cleaned, feature-enriched data for analysis"""
        # Try to load from storage first
        df = self.storage.load_data(symbol, data_type, start_date, end_date)
        
        if df is None:
            # Fetch new data
            if data_type == 'crypto_ohlcv':
                df = self.fetch_crypto_data(symbol)
            elif data_type == 'stock_ohlcv':
                df = self.fetch_stock_data(symbol)
                
        if df is not None and create_features:
            df = self.create_features(df, symbol)
            
        return df

# Global data pipeline instance
data_pipeline = None

def initialize_data_pipeline(storage_dir: str = "data/storage") -> DataPipeline:
    """Initialize global data pipeline"""
    global data_pipeline
    data_pipeline = DataPipeline(storage_dir)
    logger.info("📊 Data pipeline initialized with parquet storage")
    return data_pipeline