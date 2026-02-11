"""
Advanced Feature Engineering for Chloe AI
Enhanced technical indicators and market features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class AdvancedFeatureEngineer:
    """
    Advanced feature engineering for market analysis
    """
    
    def __init__(self):
        self.feature_names = []
        self.feature_importance = {}
        
    def create_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced price pattern features
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional pattern features
        """
        logger.info("🔍 Creating price pattern features...")
        
        # Price action patterns
        close_prices = df['close'] if 'close' in df.columns else df['Close']
        high_prices = df['high'] if 'high' in df.columns else df['High']
        low_prices = df['low'] if 'low' in df.columns else df['Low']
        open_prices = df['open'] if 'open' in df.columns else df['Open']
        
        # Candlestick patterns (simplified)
        df['candle_body'] = abs(close_prices - open_prices)
        df['candle_upper_shadow'] = high_prices - np.maximum(close_prices, open_prices)
        df['candle_lower_shadow'] = np.minimum(close_prices, open_prices) - low_prices
        df['candle_body_ratio'] = df['candle_body'] / (high_prices - low_prices + 1e-8)
        
        # Price action momentum
        df['price_acceleration'] = close_prices.diff().diff()
        df['price_jerk'] = df['price_acceleration'].diff()
        
        # Support/Resistance levels (rolling)
        df['support_level'] = low_prices.rolling(window=20).min()
        df['resistance_level'] = high_prices.rolling(window=20).max()
        df['price_to_support'] = (close_prices - df['support_level']) / (df['resistance_level'] - df['support_level'] + 1e-8)
        
        # Price clustering
        df['price_cluster_5'] = close_prices.rolling(window=5).std()
        df['price_cluster_20'] = close_prices.rolling(window=20).std()
        df['cluster_ratio'] = df['price_cluster_5'] / (df['price_cluster_20'] + 1e-8)
        
        logger.info("✅ Price pattern features created")
        return df
    
    def create_volume_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced volume pattern features
        
        Args:
            df: DataFrame with volume data
            
        Returns:
            DataFrame with additional volume features
        """
        logger.info("📊 Creating volume pattern features...")
        
        volume = df['volume'] if 'volume' in df.columns else df['Volume']
        close_prices = df['close'] if 'close' in df.columns else df['Close']
        
        # Volume momentum and patterns
        df['volume_momentum'] = volume.diff()
        df['volume_acceleration'] = df['volume_momentum'].diff()
        
        # Volume-price relationships
        df['price_volume_trend'] = (close_prices.diff() * volume.diff()) / (volume + 1e-8)
        df['volume_price_ratio'] = volume / volume.rolling(window=20).mean()
        
        # Volume clustering
        df['volume_cluster'] = volume.rolling(window=10).std() / (volume.rolling(window=10).mean() + 1e-8)
        
        # On-Balance Volume (OBV) style indicator
        price_diff = close_prices.diff()
        df['volume_flow'] = np.where(price_diff > 0, volume, np.where(price_diff < 0, -volume, 0))
        df['cumulative_volume_flow'] = df['volume_flow'].cumsum()
        
        # Volume confirmation
        df['volume_confirmation'] = (volume > volume.rolling(window=20).mean()).astype(int)
        
        logger.info("✅ Volume pattern features created")
        return df
    
    def create_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for detecting market regimes
        
        Args:
            df: DataFrame with market data
            
        Returns:
            DataFrame with regime detection features
        """
        logger.info("🎯 Creating market regime features...")
        
        close_prices = df['close'] if 'close' in df.columns else df['Close']
        
        # Volatility-based regime detection
        returns = close_prices.pct_change()
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
        
        logger.info("✅ Market regime features created")
        return df
    
    def create_time_based_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based and seasonal features
        
        Args:
            df: DataFrame with datetime index
            
        Returns:
            DataFrame with time-based features
        """
        logger.info("⏰ Creating time-based features...")
        
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
        
        # Market session indicators (assuming 24/7 crypto market)
        # For traditional markets, you'd add specific session times
        df['is_active_trading'] = 1  # Always active for crypto
        
        logger.info("✅ Time-based features created")
        return df
    
    def create_cross_asset_features(self, df: pd.DataFrame, 
                                  correlated_assets: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """
        Create cross-asset correlation features
        
        Args:
            df: Main asset DataFrame
            correlated_assets: Dictionary of correlated asset DataFrames
            
        Returns:
            DataFrame with cross-asset features
        """
        logger.info("🔗 Creating cross-asset features...")
        
        if correlated_assets is None:
            correlated_assets = {}
            
        close_prices = df['close'] if 'close' in df.columns else df['Close']
        
        # Self-correlation features
        for lag in [1, 2, 3, 5, 10]:
            df[f'price_autocorr_{lag}'] = close_prices.autocorr(lag=lag)
        
        # Cross-asset correlation (if provided)
        for asset_name, asset_df in correlated_assets.items():
            try:
                asset_close = asset_df['close'] if 'close' in asset_df.columns else asset_df['Close']
                
                # Align indices
                aligned_df = pd.concat([close_prices, asset_close], axis=1, join='inner')
                aligned_df.columns = ['main_asset', 'correlated_asset']
                
                if len(aligned_df) > 30:  # Need minimum data
                    correlation = aligned_df['main_asset'].corr(aligned_df['correlated_asset'])
                    df[f'{asset_name}_correlation'] = correlation
                    
                    # Relative strength
                    df[f'{asset_name}_relative_strength'] = close_prices / asset_close.iloc[-len(close_prices):].values
                    
            except Exception as e:
                logger.warning(f"Could not create features for {asset_name}: {e}")
        
        logger.info("✅ Cross-asset features created")
        return df
    
    def create_advanced_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced momentum and trend-following features
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with advanced momentum features
        """
        logger.info("🚀 Creating advanced momentum features...")
        
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
        
        # Hurst exponent approximation (simplified)
        def hurst_exponent(ts, max_lag=20):
            lags = range(2, min(max_lag, len(ts)//2))
            if len(lags) < 2:
                return 0.5
            
            tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
            # Avoid log(0)
            tau = [t if t > 0 else 1e-10 for t in tau]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        
        # Apply to rolling windows
        df['hurst_50'] = close_prices.rolling(window=100).apply(lambda x: hurst_exponent(x) if len(x) > 20 else 0.5)
        
        logger.info("✅ Advanced momentum features created")
        return df
    
    def create_all_advanced_features(self, df: pd.DataFrame, 
                                   correlated_assets: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """
        Create all advanced features
        
        Args:
            df: DataFrame with market data
            correlated_assets: Dictionary of correlated assets
            
        Returns:
            DataFrame with all advanced features
        """
        logger.info("🔧 Creating all advanced features...")
        
        df = self.create_price_patterns(df)
        df = self.create_volume_patterns(df)
        df = self.create_market_regime_features(df)
        df = self.create_time_based_features(df)
        df = self.create_cross_asset_features(df, correlated_assets)
        df = self.create_advanced_momentum_features(df)
        
        # Record feature names
        self.feature_names = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        logger.info(f"✅ All advanced features created ({len(self.feature_names)} total features)")
        return df
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores (placeholder for actual importance)
        
        Returns:
            Dictionary of feature importance scores
        """
        # This would typically come from the trained model
        # For now, return uniform importance
        if self.feature_names:
            importance = {feature: 1.0/len(self.feature_names) for feature in self.feature_names}
            return importance
        return {}

# Example usage
def main():
    """Example usage of Advanced Feature Engineer"""
    print("Advanced Feature Engineering module ready")
    
    # This would be used in conjunction with the main pipeline
    engineer = AdvancedFeatureEngineer()
    print("Feature engineer initialized")

if __name__ == "__main__":
    main()