"""
Professional Market Data Processor
Advanced data cleaning, validation, and feature engineering
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import yfinance as yf
import ccxt
from dataclasses import dataclass
import warnings

logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Container for market data"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread: Optional[float] = None

class DataValidator:
    """Professional data validation and quality control"""
    
    @staticmethod
    def validate_ohlcv_structure(df: pd.DataFrame) -> bool:
        """Validate OHLCV data structure"""
        required_columns = {'open', 'high', 'low', 'close', 'volume'}
        actual_columns = set(df.columns)
        
        # Check for lowercase or uppercase variants
        has_required = any(col.lower() in required_columns for col in df.columns)
        
        if not has_required:
            logger.error(f"❌ Missing required OHLCV columns: {required_columns}")
            return False
            
        return True
        
    @staticmethod
    def detect_outliers(df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.Series:
        """Detect outliers using Z-score method"""
        prices = df[column]
        z_scores = np.abs((prices - prices.mean()) / prices.std())
        return z_scores > threshold
        
    @staticmethod
    def validate_price_relationships(df: pd.DataFrame) -> pd.DataFrame:
        """Validate that OHLC prices follow logical relationships"""
        df_valid = df.copy()
        
        # Identify columns regardless of case
        col_map = {col.lower(): col for col in df.columns}
        
        open_col = col_map.get('open')
        high_col = col_map.get('high')
        low_col = col_map.get('low')
        close_col = col_map.get('close')
        
        if all([open_col, high_col, low_col, close_col]):
            # Ensure high >= max(open, close) and low <= min(open, close)
            df_valid[high_col] = df_valid[[high_col, open_col, close_col]].max(axis=1)
            df_valid[low_col] = df_valid[[low_col, open_col, close_col]].min(axis=1)
            
            # Ensure high >= low
            df_valid.loc[df_valid[high_col] < df_valid[low_col], high_col] = df_valid[low_col]
            
        return df_valid
        
    @staticmethod
    def detect_lookahead_bias(df: pd.DataFrame, feature_columns: List[str]) -> Dict[str, float]:
        """Detect potential lookahead bias in features"""
        lookahead_warnings = {}
        
        if 'close' in df.columns:
            future_returns = df['close'].shift(-1) / df['close'] - 1  # Future 1-period return
            future_returns = future_returns[:-1]  # Remove last NaN
            
            for feature in feature_columns:
                if feature in df.columns:
                    feature_values = df[feature].iloc[:-1]  # Align with future returns
                    
                    if len(feature_values) == len(future_returns):
                        correlation = np.corrcoef(feature_values, future_returns)[0, 1]
                        if not np.isnan(correlation) and abs(correlation) > 0.7:
                            lookahead_warnings[feature] = correlation
                            
        return lookahead_warnings

class FeatureEngineer:
    """Advanced feature engineering for trading models"""
    
    @staticmethod
    def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        df_features = df.copy()
        
        # Identify price columns
        col_map = {col.lower(): col for col in df.columns}
        close_col = col_map.get('close', col_map.get('Close', 'close'))
        high_col = col_map.get('high', col_map.get('High', 'high'))
        low_col = col_map.get('low', col_map.get('Low', 'low'))
        open_col = col_map.get('open', col_map.get('Open', 'open'))
        volume_col = col_map.get('volume', col_map.get('Volume', 'volume'))
        
        # Price-based features
        if close_col in df_features.columns:
            df_features[f'{close_col}_returns'] = df_features[close_col].pct_change()
            df_features[f'{close_col}_log_returns'] = np.log(df_features[close_col] / df_features[close_col].shift(1))
            
            # Moving averages
            for window in [5, 10, 20, 50, 100, 200]:
                df_features[f'{close_col}_sma_{window}'] = df_features[close_col].rolling(window=window).mean()
                df_features[f'{close_col}_ema_{window}'] = df_features[close_col].ewm(span=window).mean()
                
            # Volatility measures
            for window in [10, 20, 30]:
                df_features[f'{close_col}_volatility_{window}'] = df_features[f'{close_col}_returns'].rolling(window=window).std()
                
            # Momentum indicators
            for period in [1, 3, 5, 10, 20]:
                df_features[f'{close_col}_momentum_{period}'] = df_features[close_col] / df_features[close_col].shift(period) - 1
                
            # Price position indicators
            for window in [5, 10, 20]:
                rolling_min = df_features[low_col].rolling(window=window).min() if low_col in df_features.columns else df_features[close_col].rolling(window=window).min()
                rolling_max = df_features[high_col].rolling(window=window).max() if high_col in df_features.columns else df_features[close_col].rolling(window=window).max()
                df_features[f'{close_col}_position_{window}'] = (df_features[close_col] - rolling_min) / (rolling_max - rolling_min)
                
        # Volume-based features (if volume column exists)
        if volume_col in df_features.columns:
            for window in [5, 10, 20]:
                df_features[f'{volume_col}_sma_{window}'] = df_features[volume_col].rolling(window=window).mean()
                df_features[f'{volume_col}_ratio_{window}'] = df_features[volume_col] / df_features[f'{volume_col}_sma_{window}']
                
        # Advanced indicators
        if all(col in df_features.columns for col in [high_col, low_col, close_col]):
            df_features = FeatureEngineer._calculate_rsi(df_features, close_col)
            df_features = FeatureEngineer._calculate_macd(df_features, close_col)
            df_features = FeatureEngineer._calculate_bollinger_bands(df_features, close_col)
            df_features = FeatureEngineer._calculate_stochastic(df_features, high_col, low_col, close_col)
            df_features = FeatureEngineer._calculate_atr(df_features, high_col, low_col, close_col)
            
        # Lagged features for ML models
        for lag in [1, 2, 3, 5, 10]:
            if close_col in df_features.columns:
                df_features[f'{close_col}_lag_{lag}'] = df_features[close_col].shift(lag)
            if volume_col in df_features.columns:
                df_features[f'{volume_col}_lag_{lag}'] = df_features[volume_col].shift(lag)
                
        # Target variables (future returns - ensure no lookahead bias)
        for horizon in [1, 3, 5, 10]:
            if close_col in df_features.columns:
                df_features[f'target_{horizon}_returns'] = df_features[close_col].shift(-horizon) / df_features[close_col] - 1
                
        return df_features
        
    @staticmethod
    def _calculate_rsi(df: pd.DataFrame, close_col: str, window: int = 14) -> pd.DataFrame:
        """Calculate Relative Strength Index"""
        df_copy = df.copy()
        delta = df_copy[close_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        df_copy[f'{close_col}_rsi_{window}'] = 100 - (100 / (1 + rs))
        return df_copy
        
    @staticmethod
    def _calculate_macd(df: pd.DataFrame, close_col: str, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Calculate MACD indicator"""
        df_copy = df.copy()
        ema_fast = df_copy[close_col].ewm(span=fast).mean()
        ema_slow = df_copy[close_col].ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal).mean()
        macd_histogram = macd_line - macd_signal
        
        df_copy[f'{close_col}_macd_line'] = macd_line
        df_copy[f'{close_col}_macd_signal'] = macd_signal
        df_copy[f'{close_col}_macd_histogram'] = macd_histogram
        
        return df_copy
        
    @staticmethod
    def _calculate_bollinger_bands(df: pd.DataFrame, close_col: str, window: int = 20, num_std: int = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        df_copy = df.copy()
        sma = df_copy[close_col].rolling(window=window).mean()
        std = df_copy[close_col].rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        df_copy[f'{close_col}_bb_upper'] = upper_band
        df_copy[f'{close_col}_bb_lower'] = lower_band
        df_copy[f'{close_col}_bb_width'] = upper_band - lower_band
        df_copy[f'{close_col}_bb_position'] = (df_copy[close_col] - lower_band) / (upper_band - lower_band)
        
        return df_copy
        
    @staticmethod
    def _calculate_stochastic(df: pd.DataFrame, high_col: str, low_col: str, close_col: str, k_window: int = 14) -> pd.DataFrame:
        """Calculate Stochastic Oscillator"""
        df_copy = df.copy()
        rolling_high = df_copy[high_col].rolling(window=k_window).max()
        rolling_low = df_copy[low_col].rolling(window=k_window).min()
        
        df_copy[f'{close_col}_stoch_k'] = 100 * (df_copy[close_col] - rolling_low) / (rolling_high - rolling_low)
        df_copy[f'{close_col}_stoch_d'] = df_copy[f'{close_col}_stoch_k'].rolling(window=3).mean()
        
        return df_copy
        
    @staticmethod
    def _calculate_atr(df: pd.DataFrame, high_col: str, low_col: str, close_col: str, window: int = 14) -> pd.DataFrame:
        """Calculate Average True Range"""
        df_copy = df.copy()
        
        prev_close = df_copy[close_col].shift(1)
        tr1 = df_copy[high_col] - df_copy[low_col]
        tr2 = abs(df_copy[high_col] - prev_close)
        tr3 = abs(df_copy[low_col] - prev_close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df_copy[f'{close_col}_atr_{window}'] = tr.rolling(window=window).mean()
        
        return df_copy

class MarketDataProvider:
    """Professional market data provider with multiple sources"""
    
    def __init__(self):
        self.exchanges = {
            'binance': ccxt.binance({'enableRateLimit': True}),
            'coinbase': ccxt.coinbase({'enableRateLimit': True}),
            'bybit': ccxt.bybit({'enableRateLimit': True})
        }
        self.validator = DataValidator()
        self.feature_engineer = FeatureEngineer()
        
    async def fetch_crypto_data(self, symbol: str, timeframe: str = '1h', 
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
            
            # Validate data
            if not self.validator.validate_ohlcv_structure(df):
                logger.error(f"❌ Invalid OHLCV structure for {symbol}")
                return None
                
            # Validate price relationships
            df = self.validator.validate_price_relationships(df)
            
            # Detect and handle outliers
            for col in ['open', 'high', 'low', 'close']:
                if col in df.columns:
                    outliers = self.validator.detect_outliers(df, col)
                    df.loc[outliers, col] = np.nan  # Mark outliers for cleaning
                    
            # Forward fill remaining NaN values
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # Remove duplicates
            df = df[~df.index.duplicated(keep='first')]
            
            # Sort by timestamp
            df = df.sort_index()
            
            logger.info(f"📊 Fetched {len(df)} rows of {symbol} crypto data from {exchange}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error fetching {symbol} crypto data: {e}")
            return None
            
    def fetch_stock_data(self, symbol: str, period: str = '1y', 
                        interval: str = '1d') -> Optional[pd.DataFrame]:
        """Fetch stock data from Yahoo Finance"""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period, interval=interval)
                
            if df.empty:
                logger.error(f"❌ No data found for stock {symbol}")
                return None
                
            # Rename columns to lowercase for consistency
            df.columns = df.columns.str.lower()
            
            # Validate data
            if not self.validator.validate_ohlcv_structure(df):
                logger.error(f"❌ Invalid OHLCV structure for {symbol}")
                return None
                
            # Validate price relationships
            df = self.validator.validate_price_relationships(df)
            
            # Detect and handle outliers
            for col in ['open', 'high', 'low', 'close']:
                if col in df.columns:
                    outliers = self.validator.detect_outliers(df, col)
                    df.loc[outliers, col] = np.nan  # Mark outliers for cleaning
                    
            # Forward fill remaining NaN values
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # Remove duplicates
            df = df[~df.index.duplicated(keep='first')]
            
            # Sort by timestamp
            df = df.sort_index()
            
            logger.info(f"📊 Fetched {len(df)} rows of {symbol} stock data")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error fetching {symbol} stock data: {e}")
            return None
            
    def clean_and_validate_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Clean and validate market data"""
        df_clean = df.copy()
        
        # Validate OHLCV structure
        if not self.validator.validate_ohlcv_structure(df_clean):
            raise ValueError(f"Invalid OHLCV structure for {symbol}")
            
        # Validate price relationships
        df_clean = self.validator.validate_price_relationships(df_clean)
        
        # Handle outliers
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df_clean.columns:
                outliers = self.validator.detect_outliers(df_clean, col)
                # Replace outliers with median of nearby values
                for idx in outliers[outliers].index:
                    window = df_clean[col].rolling(window=5, center=True).median()
                    df_clean.at[idx, col] = window.loc[idx]
                    
        # Forward fill any remaining NaN values
        df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')
        
        # Remove duplicates
        df_clean = df_clean[~df_clean.index.duplicated(keep='first')]
        
        # Sort by timestamp
        df_clean = df_clean.sort_index()
        
        # Validate that we still have reasonable data
        if len(df_clean) < 10:
            raise ValueError(f"Not enough valid data points after cleaning for {symbol}")
            
        logger.info(f"🧹 Cleaned data for {symbol}: {len(df)} → {len(df_clean)} rows")
        return df_clean
        
    def create_features(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """Create advanced features for trading models"""
        try:
            df_features = self.feature_engineer.calculate_technical_indicators(df)
            
            # Detect potential lookahead bias
            feature_cols = [col for col in df_features.columns if col not in df.columns]
            lookahead_warnings = self.validator.detect_lookahead_bias(df_features, feature_cols)
            
            if lookahead_warnings:
                logger.warning(f"⚠️ Potential lookahead bias detected in {symbol}: {lookahead_warnings}")
                
            logger.info(f"⚙️ Created {len(feature_cols)} features for {symbol}")
            return df_features
            
        except Exception as e:
            logger.error(f"❌ Error creating features for {symbol}: {e}")
            return df

class MarketDataProcessor:
    """Main market data processing orchestrator"""
    
    def __init__(self):
        self.provider = MarketDataProvider()
        self.validator = DataValidator()
        
    def get_processed_data(self, symbol: str, data_type: str = 'crypto', 
                          timeframe: str = '1d', days: int = 365) -> Optional[pd.DataFrame]:
        """Get fully processed market data with features"""
        logger.info(f"🔄 Processing market data for {symbol}")
        
        # Fetch raw data
        if data_type.lower() == 'crypto':
            df_raw = self.provider.fetch_crypto_data(symbol, timeframe=timeframe, limit=days)
        elif data_type.lower() == 'stock':
            period_map = {
                7: '1wk', 30: '1mo', 90: '3mo', 180: '6mo',
                365: '1y', 730: '2y', 1095: '3y', 1825: '5y'
            }
            period = period_map.get(min(days, max(period_map.keys())), '1y')
            df_raw = self.provider.fetch_stock_data(symbol, period=period, interval=timeframe)
        else:
            logger.error(f"❌ Unsupported data type: {data_type}")
            return None
            
        if df_raw is None or df_raw.empty:
            logger.error(f"❌ No raw data fetched for {symbol}")
            return None
            
        # Clean and validate data
        try:
            df_clean = self.provider.clean_and_validate_data(df_raw, symbol)
        except Exception as e:
            logger.error(f"❌ Error cleaning data for {symbol}: {e}")
            return None
            
        # Create features
        df_features = self.provider.create_features(df_clean, symbol)
        
        logger.info(f"✅ Processed {len(df_features)} rows of feature-rich data for {symbol}")
        return df_features
        
    def validate_data_quality(self, df: pd.DataFrame, symbol: str) -> Dict[str, Union[bool, str, float]]:
        """Validate overall data quality"""
        quality_report = {
            'symbol': symbol,
            'total_rows': len(df),
            'start_date': df.index.min().isoformat() if len(df) > 0 else None,
            'end_date': df.index.max().isoformat() if len(df) > 0 else None,
            'has_gaps': False,
            'outlier_percentage': 0.0,
            'valid_ohlc_structure': self.validator.validate_ohlcv_structure(df),
            'data_completeness': 0.0
        }
        
        if len(df) > 0:
            # Check for gaps in time series
            expected_frequency = pd.infer_freq(df.index)
            if expected_frequency:
                complete_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=expected_frequency)
                missing_dates = len(complete_range) - len(df)
                quality_report['has_gaps'] = missing_dates > 0
                quality_report['gap_percentage'] = (missing_dates / len(complete_range)) * 100
                
            # Calculate outlier percentage
            total_outliers = 0
            total_points = 0
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    outliers = self.validator.detect_outliers(df, col)
                    total_outliers += outliers.sum()
                    total_points += len(outliers)
                    
            quality_report['outlier_percentage'] = (total_outliers / total_points * 100) if total_points > 0 else 0.0
            
            # Data completeness (percentage of non-NaN values)
            total_values = df.count().sum()
            possible_values = np.prod(df.shape)
            quality_report['data_completeness'] = (total_values / possible_values * 100) if possible_values > 0 else 0.0
            
        return quality_report

# Global market data processor instance
market_data_processor = None

def initialize_market_data_processor() -> MarketDataProcessor:
    """Initialize global market data processor"""
    global market_data_processor
    market_data_processor = MarketDataProcessor()
    logger.info("📊 Market data processor initialized with professional validation")
    return market_data_processor