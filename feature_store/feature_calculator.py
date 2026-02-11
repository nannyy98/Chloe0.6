"""
Unified Feature Calculator
Combines all technical indicators and advanced features in one place
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class FeatureCalculator:
    """
    Unified feature calculator that combines all feature engineering logic
    Single source of truth for all market features
    """
    
    def __init__(self):
        self.feature_metadata = {}
        self.calculated_features = set()
        
    def calculate_all_features(self, df: pd.DataFrame, 
                             symbol: str = None,
                             correlated_assets: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """
        Calculate all features in correct order and without duplication
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Symbol name for logging
            correlated_assets: Dictionary of correlated asset data
            
        Returns:
            DataFrame with all calculated features
        """
        logger.info(f"🧮 Calculating unified features for {symbol or 'unknown'}...")
        
        # Step 1: Basic price transformations
        df = self._calculate_basic_transformations(df)
        
        # Step 2: Technical indicators (traditional)
        df = self._calculate_technical_indicators(df)
        
        # Step 3: Advanced statistical features
        df = self._calculate_statistical_features(df)
        
        # Step 4: Price pattern features
        df = self._calculate_price_patterns(df)
        
        # Step 5: Volume features
        df = self._calculate_volume_features(df)
        
        # Step 6: Market regime features
        df = self._calculate_market_regime_features(df)
        
        # Step 7: Time-based features
        df = self._calculate_time_features(df)
        
        # Step 8: Cross-asset features (if provided)
        if correlated_assets:
            df = self._calculate_cross_asset_features(df, correlated_assets)
        
        # Step 9: Momentum and trend features
        df = self._calculate_momentum_features(df)
        
        # Record calculated features
        self._record_feature_metadata(df, symbol)
        
        logger.info(f"✅ Unified feature calculation complete ({len(df.columns)} total columns)")
        return df
    
    def _calculate_basic_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic price transformations"""
        close_prices = df['close'] if 'close' in df.columns else df['Close']
        high_prices = df['high'] if 'high' in df.columns else df['High']
        low_prices = df['low'] if 'low' in df.columns else df['Low']
        open_prices = df['open'] if 'open' in df.columns else df['Open']
        volume = df['volume'] if 'volume' in df.columns else df['Volume']
        
        # Log returns
        df['log_return'] = np.log(close_prices / close_prices.shift(1))
        df['price_change_pct'] = close_prices.pct_change()
        
        # Price ranges
        df['high_low_range'] = (high_prices - low_prices) / close_prices
        df['open_close_range'] = abs(open_prices - close_prices) / close_prices
        
        # Volume normalization
        df['volume_norm'] = volume / volume.rolling(20).mean()
        
        return df
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate traditional technical indicators without duplication"""
        close_prices = df['close'] if 'close' in df.columns else df['Close']
        high_prices = df['high'] if 'high' in df.columns else df['High']
        low_prices = df['low'] if 'low' in df.columns else df['Low']
        volume = df['volume'] if 'volume' in df.columns else df['Volume']
        
        # RSI (14 period)
        delta = close_prices.diff()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        avg_gain = gains.rolling(window=14).mean()
        avg_loss = losses.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = close_prices.ewm(span=12).mean()
        ema_26 = close_prices.ewm(span=26).mean()
        df['macd_line'] = ema_12 - ema_26
        df['macd_signal'] = df['macd_line'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd_line'] - df['macd_signal']
        
        # EMAs
        for period in [20, 50, 200]:
            df[f'ema_{period}'] = close_prices.ewm(span=period).mean()
        
        # Bollinger Bands
        sma_20 = close_prices.rolling(window=20).mean()
        std_20 = close_prices.rolling(window=20).std()
        df['bb_upper'] = sma_20 + (std_20 * 2)
        df['bb_middle'] = sma_20
        df['bb_lower'] = sma_20 - (std_20 * 2)
        df['bb_position'] = (close_prices - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Stochastic Oscillator
        lowest_low = low_prices.rolling(window=14).min()
        highest_high = high_prices.rolling(window=14).max()
        df['stoch_k'] = 100 * (close_prices - lowest_low) / (highest_high - lowest_low)
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        return df
    
    def _calculate_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate statistical features"""
        close_prices = df['close'] if 'close' in df.columns else df['Close']
        returns = df['log_return']
        
        # Volatility measures
        df['volatility_20'] = returns.rolling(window=20).std() * np.sqrt(252)
        df['volatility_60'] = returns.rolling(window=60).std() * np.sqrt(252)
        df['volatility_ratio'] = df['volatility_20'] / df['volatility_60']
        
        # Rolling statistics
        for window in [5, 10, 20, 50]:
            df[f'return_mean_{window}'] = returns.rolling(window=window).mean()
            df[f'return_std_{window}'] = returns.rolling(window=window).std()
            df[f'price_zscore_{window}'] = (close_prices - close_prices.rolling(window=window).mean()) / close_prices.rolling(window=window).std()
        
        # Skewness and kurtosis approximations
        df['skewness_20'] = returns.rolling(window=20).apply(lambda x: np.mean((x - np.mean(x))**3) / (np.std(x)**3) if len(x) > 0 else 0)
        df['kurtosis_20'] = returns.rolling(window=20).apply(lambda x: np.mean((x - np.mean(x))**4) / (np.std(x)**4) - 3 if len(x) > 0 else 0)
        
        return df
    
    def _calculate_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price pattern features"""
        close_prices = df['close'] if 'close' in df.columns else df['Close']
        high_prices = df['high'] if 'high' in df.columns else df['High']
        low_prices = df['low'] if 'low' in df.columns else df['Low']
        open_prices = df['open'] if 'open' in df.columns else df['Open']
        
        # Candlestick features
        df['candle_body'] = abs(close_prices - open_prices)
        df['candle_upper_shadow'] = high_prices - np.maximum(close_prices, open_prices)
        df['candle_lower_shadow'] = np.minimum(close_prices, open_prices) - low_prices
        df['candle_body_ratio'] = df['candle_body'] / (high_prices - low_prices + 1e-8)
        
        # Price action momentum
        df['price_acceleration'] = close_prices.diff().diff()
        df['price_jerk'] = df['price_acceleration'].diff()
        
        # Support/Resistance levels
        df['support_level_20'] = low_prices.rolling(window=20).min()
        df['resistance_level_20'] = high_prices.rolling(window=20).max()
        df['price_to_support'] = (close_prices - df['support_level_20']) / (df['resistance_level_20'] - df['support_level_20'] + 1e-8)
        
        # Price clustering
        df['price_cluster_5'] = close_prices.rolling(window=5).std()
        df['price_cluster_20'] = close_prices.rolling(window=20).std()
        df['cluster_ratio'] = df['price_cluster_5'] / (df['price_cluster_20'] + 1e-8)
        
        return df
    
    def _calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based features"""
        volume = df['volume'] if 'volume' in df.columns else df['Volume']
        close_prices = df['close'] if 'close' in df.columns else df['Close']
        returns = df['log_return']
        
        # Volume momentum and patterns
        df['volume_momentum'] = volume.diff()
        df['volume_acceleration'] = df['volume_momentum'].diff()
        
        # Volume-price relationships
        df['price_volume_trend'] = (close_prices.diff() * volume.diff()) / (volume + 1e-8)
        df['volume_price_ratio'] = volume / volume.rolling(window=20).mean()
        
        # Volume clustering
        df['volume_cluster'] = volume.rolling(window=10).std() / (volume.rolling(window=10).mean() + 1e-8)
        
        # On-Balance Volume style indicator
        price_diff = close_prices.diff()
        df['volume_flow'] = np.where(price_diff > 0, volume, np.where(price_diff < 0, -volume, 0))
        df['cumulative_volume_flow'] = df['volume_flow'].cumsum()
        
        # Volume confirmation
        df['volume_confirmation'] = (volume > volume.rolling(window=20).mean()).astype(int)
        
        return df
    
    def _calculate_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate market regime detection features"""
        close_prices = df['close'] if 'close' in df.columns else df['Close']
        returns = df['log_return']
        volume = df['volume'] if 'volume' in df.columns else df['Volume']
        
        # Volatility-based regime detection
        df['volatility_5'] = returns.rolling(window=5).std()
        df['volatility_20'] = returns.rolling(window=20).std()
        df['volatility_60'] = returns.rolling(window=60).std()
        
        # Regime ratios
        df['volatility_regime'] = df['volatility_20'] / (df['volatility_60'] + 1e-8)
        df['short_term_regime'] = df['volatility_5'] / (df['volatility_20'] + 1e-8)
        
        # Trend strength indicators
        df['trend_strength_20'] = abs(close_prices.diff(20)) / (close_prices * 0.2 + 1e-8)
        df['trend_consistency'] = (returns.rolling(window=10).apply(lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0.5) - 0.5) * 2
        
        # Market stress indicators
        df['market_stress'] = df['volatility_20'] * df['trend_strength_20']
        df['regime_shift_probability'] = df['volatility_regime'].diff().abs()
        
        return df
    
    def _calculate_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate time-based features"""
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("DataFrame index is not datetime, skipping time features")
            return df
        
        # Basic time features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['is_month_end'] = (df.index.is_month_end).astype(int)
        df['is_month_start'] = (df.index.is_month_start).astype(int)
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def _calculate_cross_asset_features(self, df: pd.DataFrame, 
                                      correlated_assets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate cross-asset correlation features"""
        close_prices = df['close'] if 'close' in df.columns else df['Close']
        
        # Self-correlation features
        for lag in [1, 2, 3, 5, 10]:
            df[f'price_autocorr_{lag}'] = close_prices.autocorr(lag=lag)
        
        # Cross-asset correlation
        for asset_name, asset_df in correlated_assets.items():
            try:
                asset_close = asset_df['close'] if 'close' in asset_df.columns else asset_df['Close']
                
                # Align indices
                aligned_df = pd.concat([close_prices, asset_close], axis=1, join='inner')
                aligned_df.columns = ['main_asset', 'correlated_asset']
                
                if len(aligned_df) > 30:
                    correlation = aligned_df['main_asset'].corr(aligned_df['correlated_asset'])
                    df[f'{asset_name}_correlation'] = correlation
                    
                    # Relative strength
                    df[f'{asset_name}_relative_strength'] = close_prices / asset_close.iloc[-len(close_prices):].values
                    
            except Exception as e:
                logger.warning(f"Could not create features for {asset_name}: {e}")
        
        return df
    
    def _calculate_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced momentum and trend-following features"""
        close_prices = df['close'] if 'close' in df.columns else df['Close']
        
        # Multiple timeframe momentum
        timeframes = [3, 5, 8, 13, 21, 34, 55]  # Fibonacci numbers
        
        for tf in timeframes:
            if tf < len(close_prices):
                momentum = close_prices.pct_change(periods=tf)
                df[f'momentum_{tf}'] = momentum
                df[f'momentum_rank_{tf}'] = momentum.rank(pct=True)
        
        # Rate of Change (ROC) variations
        for tf in [10, 20, 50]:
            if tf < len(close_prices):
                df[f'roc_{tf}'] = (close_prices - close_prices.shift(tf)) / close_prices.shift(tf) * 100
                df[f'roc_acceleration_{tf}'] = df[f'roc_{tf}'].diff()
        
        # Price distance to moving averages
        for ma_period in [10, 20, 50, 100, 200]:
            if ma_period < len(close_prices):
                ma = close_prices.rolling(window=ma_period).mean()
                df[f'price_to_ma_{ma_period}'] = (close_prices - ma) / ma
                df[f'ma_slope_{ma_period}'] = ma.diff() / ma
        
        return df
    
    def _record_feature_metadata(self, df: pd.DataFrame, symbol: str):
        """Record metadata about calculated features"""
        self.feature_metadata[symbol] = {
            'total_features': len(df.columns),
            'calculation_timestamp': pd.Timestamp.now(),
            'feature_names': list(df.columns)
        }
        self.calculated_features.update(df.columns)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance (placeholder - would come from trained model)"""
        if self.calculated_features:
            importance = {feature: 1.0/len(self.calculated_features) for feature in self.calculated_features}
            return importance
        return {}

# Global feature calculator instance
feature_calculator = None

def get_feature_calculator() -> FeatureCalculator:
    """Get singleton feature calculator instance"""
    global feature_calculator
    if feature_calculator is None:
        feature_calculator = FeatureCalculator()
    return feature_calculator