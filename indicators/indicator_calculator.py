"""
Indicator Calculator Module for Chloe AI
Calculates technical indicators for market analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class IndicatorCalculator:
    """
    Calculator for technical indicators used in market analysis
    """
    
    def __init__(self):
        pass
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            df: DataFrame with 'close' column
            period: RSI period (default 14)
            
        Returns:
            DataFrame with RSI column added
        """
        try:
            close_prices = df['close'] if 'close' in df.columns else df['Close']
            
            # Calculate price changes
            delta = close_prices.diff()
            
            # Separate gains and losses
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            
            # Calculate average gains and losses
            avg_gain = gains.rolling(window=period).mean()
            avg_loss = losses.rolling(window=period).mean()
            
            # Calculate RS and RSI
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            df[f'rsi_{period}'] = rsi
            logger.info(f"✅ RSI ({period}) calculated")
            return df
        except Exception as e:
            logger.error(f"❌ Error calculating RSI: {e}")
            return df
    
    def calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Moving Average Convergence Divergence (MACD)
        
        Args:
            df: DataFrame with 'close' column
            
        Returns:
            DataFrame with MACD columns added
        """
        try:
            close_prices = df['close'] if 'close' in df.columns else df['Close']
            
            # Calculate MACD line (12-day EMA - 26-day EMA)
            ema_12 = close_prices.ewm(span=12).mean()
            ema_26 = close_prices.ewm(span=26).mean()
            macd_line = ema_12 - ema_26
            
            # Calculate signal line (9-day EMA of MACD line)
            signal_line = macd_line.ewm(span=9).mean()
            
            # Calculate histogram
            histogram = macd_line - signal_line
            
            df['macd'] = macd_line
            df['macd_signal'] = signal_line
            df['macd_hist'] = histogram
            logger.info("✅ MACD calculated")
            return df
        except Exception as e:
            logger.error(f"❌ Error calculating MACD: {e}")
            return df
    
    def calculate_ema(self, df: pd.DataFrame, periods: list = [20, 50, 200]) -> pd.DataFrame:
        """
        Calculate Exponential Moving Averages (EMA)
        
        Args:
            df: DataFrame with 'close' column
            periods: List of periods for EMA calculation
            
        Returns:
            DataFrame with EMA columns added
        """
        try:
            close_prices = df['close'] if 'close' in df.columns else df['Close']
            
            for period in periods:
                ema_values = close_prices.ewm(span=period).mean()
                df[f'ema_{period}'] = ema_values
            
            logger.info(f"✅ EMAs calculated for periods: {periods}")
            return df
        except Exception as e:
            logger.error(f"❌ Error calculating EMAs: {e}")
            return df
    
    def calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: int = 2) -> pd.DataFrame:
        """
        Calculate Bollinger Bands
        
        Args:
            df: DataFrame with 'close' column
            period: Period for moving average
            std_dev: Standard deviation multiplier
            
        Returns:
            DataFrame with Bollinger Band columns added
        """
        try:
            close_prices = df['close'] if 'close' in df.columns else df['Close']
            
            # Calculate middle band (SMA)
            middle_band = close_prices.rolling(window=period).mean()
            
            # Calculate standard deviation
            std = close_prices.rolling(window=period).std()
            
            # Calculate upper and lower bands
            upper_band = middle_band + (std * std_dev)
            lower_band = middle_band - (std * std_dev)
            
            df['bb_upper'] = upper_band
            df['bb_middle'] = middle_band
            df['bb_lower'] = lower_band
            logger.info("✅ Bollinger Bands calculated")
            return df
        except Exception as e:
            logger.error(f"❌ Error calculating Bollinger Bands: {e}")
            return df
    
    def calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """
        Calculate Stochastic Oscillator
        
        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            k_period: %K period
            d_period: %D period
            
        Returns:
            DataFrame with Stochastic columns added
        """
        try:
            high_prices = df['high'] if 'high' in df.columns else df['High']
            low_prices = df['low'] if 'low' in df.columns else df['Low']
            close_prices = df['close'] if 'close' in df.columns else df['Close']
            
            # Calculate %K
            lowest_low = low_prices.rolling(window=k_period).min()
            highest_high = high_prices.rolling(window=k_period).max()
            
            k_percent = 100 * (close_prices - lowest_low) / (highest_high - lowest_low)
            
            # Calculate %D (3-period SMA of %K)
            d_percent = k_percent.rolling(window=d_period).mean()
            
            df['stoch_k'] = k_percent
            df['stoch_d'] = d_percent
            logger.info("✅ Stochastic Oscillator calculated")
            return df
        except Exception as e:
            logger.error(f"❌ Error calculating Stochastic Oscillator: {e}")
            return df
    
    def calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume-based indicators
        
        Args:
            df: DataFrame with 'volume' column
            
        Returns:
            DataFrame with volume indicator columns added
        """
        try:
            volume = df['volume'] if 'volume' in df.columns else df['Volume']
            
            # Volume moving average
            volume_ma = volume.rolling(window=20).mean()
            df['volume_ma'] = volume_ma
            
            # Volume ratio
            df['volume_ratio'] = volume / volume_ma
            
            logger.info("✅ Volume indicators calculated")
            return df
        except Exception as e:
            logger.error(f"❌ Error calculating volume indicators: {e}")
            return df
    
    def calculate_volatility(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Calculate price volatility (standard deviation)
        
        Args:
            df: DataFrame with 'close' column
            period: Period for volatility calculation
            
        Returns:
            DataFrame with volatility column added
        """
        try:
            close_prices = df['close'].values if 'close' in df.columns else df['Close'].values
            returns = np.diff(np.log(close_prices))
            returns = np.append([np.nan], returns)  # Add NaN for first value
            
            # Rolling volatility
            df['volatility'] = pd.Series(returns).rolling(window=period).std() * np.sqrt(252)  # Annualized
            logger.info(f"✅ Volatility ({period}) calculated")
            return df
        except Exception as e:
            logger.error(f"❌ Error calculating volatility: {e}")
            return df
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all indicator columns added
        """
        logger.info("📊 Calculating all technical indicators...")
        
        # Calculate all indicators
        df = self.calculate_rsi(df)
        df = self.calculate_macd(df)
        df = self.calculate_ema(df)
        df = self.calculate_bollinger_bands(df)
        df = self.calculate_stochastic(df)
        df = self.calculate_volume_indicators(df)
        df = self.calculate_volatility(df)
        
        # Calculate additional derived features
        df = self.add_price_position_features(df)
        df = self.add_momentum_features(df)
        
        logger.info("✅ All technical indicators calculated")
        return df
    
    def add_price_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add price position features relative to indicators
        
        Args:
            df: DataFrame with indicators
            
        Returns:
            DataFrame with position features added
        """
        try:
            # Price position relative to Bollinger Bands
            if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
                close_prices = df['close'] if 'close' in df.columns else df['Close']
                df['price_bb_pos'] = (close_prices - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Price position relative to EMAs
            close_prices = df['close'] if 'close' in df.columns else df['Close']
            if 'ema_20' in df.columns:
                df['price_ema20_pos'] = close_prices / df['ema_20']
            if 'ema_50' in df.columns:
                df['price_ema50_pos'] = close_prices / df['ema_50']
            if 'ema_200' in df.columns:
                df['price_ema200_pos'] = close_prices / df['ema_200']
                
            logger.info("✅ Price position features added")
            return df
        except Exception as e:
            logger.error(f"❌ Error adding price position features: {e}")
            return df
    
    def add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add momentum-based features
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with momentum features added
        """
        try:
            close_prices = df['close'] if 'close' in df.columns else df['Close']
            
            # Price change percentage
            df['price_change_pct'] = close_prices.pct_change()
            
            # Price change over different periods
            df['price_change_5d'] = close_prices.pct_change(periods=5)
            df['price_change_10d'] = close_prices.pct_change(periods=10)
            df['price_change_20d'] = close_prices.pct_change(periods=20)
            
            # Rolling max/min
            df['rolling_max_10d'] = close_prices.rolling(window=10).max()
            df['rolling_min_10d'] = close_prices.rolling(window=10).min()
            df['rolling_max_30d'] = close_prices.rolling(window=30).max()
            df['rolling_min_30d'] = close_prices.rolling(window=30).min()
            
            logger.info("✅ Momentum features added")
            return df
        except Exception as e:
            logger.error(f"❌ Error adding momentum features: {e}")
            return df

# Example usage
def main():
    """Example usage of the Indicator Calculator"""
    # This would typically be used after getting data from the DataAgent
    print("Indicator calculator module ready for use with market data")

if __name__ == "__main__":
    main()