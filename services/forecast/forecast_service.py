"""
Forecast Service for Adaptive Institutional AI Trader
Implements probabilistic forecasting for market movements
"""

import logging
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass

from core.event_bus import Event, EventType
from models.quantile_model import QuantileModel
# Use the new centralized pipeline
from data_pipeline import get_data_pipeline

logger = logging.getLogger(__name__)

@dataclass
class ForecastEvent(Event):
    """Event for market forecasts"""
    symbol: str
    horizon: int  # forecast horizon in days
    expected_return: float
    volatility: float
    confidence: float
    p10: float  # 10th percentile
    p50: float  # 50th percentile (median)
    p90: float  # 90th percentile
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        self.event_type = EventType.FORECAST  # Assuming FORECAST is added to EventType

class ForecastService:
    """Service for generating market forecasts"""
    
    def __init__(self):
        self.model = QuantileModel()
        self.data_pipeline = get_data_pipeline()
        self.is_trained = False
        self.min_samples = 252  # 1 year of daily data
        
        logger.info("🔮 Forecast Service initialized")
    
    async def generate_forecast(self, symbol: str, horizon: int = 5) -> Optional[ForecastEvent]:
        """Generate probabilistic forecast for a symbol"""
        try:
            # Fetch features from pipeline
            features_df = await self._get_features_for_forecast(symbol, horizon)
            if features_df is None or len(features_df) < self.min_samples:
                logger.warning(f"Insufficient data for {symbol} forecast")
                return None
            
            # Get latest features
            latest_features = features_df.tail(1).dropna()
            if len(latest_features) == 0:
                logger.warning(f"No valid features for {symbol}")
                return None
            
            # Generate forecast
            forecast = self.model.predict(latest_features)
            
            if forecast is None:
                logger.warning(f"Model prediction failed for {symbol}")
                return None
            
            # Create forecast event
            forecast_event = ForecastEvent(
                symbol=symbol,
                horizon=horizon,
                expected_return=forecast['expected_return'],
                volatility=forecast['volatility'],
                confidence=forecast['confidence'],
                p10=forecast['p10'],
                p50=forecast['p50'],
                p90=forecast['p90']
            )
            
            logger.info(f"📈 Forecast for {symbol}: E[R]={forecast['expected_return']:.3f}, "
                       f"Conf={forecast['confidence']:.2f}, Vol={forecast['volatility']:.3f}")
            
            return forecast_event
            
        except Exception as e:
            logger.error(f"❌ Forecast generation failed for {symbol}: {e}")
            return None
    
    async def _get_features_for_forecast(self, symbol: str, horizon: int) -> Optional[pd.DataFrame]:
        """Get features needed for forecasting"""
        try:
            # Fetch data with microstructure features
            df = self.data_pipeline.get_data_for_analysis(
                symbol=symbol,
                data_type='crypto_ohlcv',
                create_features=True
            )
            
            if df is None or len(df) < self.min_samples:
                # Try to fetch fresh data
                df = await self.data_pipeline.fetch_crypto_data(symbol)
                if df is None:
                    return None
            
            # Add microstructure features
            df = self._add_microstructure_features(df, horizon)
            return df
            
        except Exception as e:
            logger.error(f"❌ Error getting features for {symbol}: {e}")
            return None
    
    def _add_microstructure_features(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Add market microstructure features for forecasting"""
        df_features = df.copy()
        
        # Log returns (multi-horizon)
        for h in [1, 3, 5, 10, 20]:
            df_features[f'return_{h}d'] = np.log(df_features['close'] / df_features['close'].shift(h))
        
        # Realized volatility
        df_features['rv_5d'] = df_features['return_1d'].rolling(5).std() * np.sqrt(252)
        df_features['rv_20d'] = df_features['return_1d'].rolling(20).std() * np.sqrt(252)
        
        # Volume imbalance
        df_features['volume_ma'] = df_features['volume'].rolling(20).mean()
        df_features['volume_ratio'] = df_features['volume'] / df_features['volume_ma']
        
        # Volatility clustering features
        df_features['vol_cluster'] = df_features['rv_20d'].rolling(20).std()
        
        # Rolling Sharpe ratio proxy
        df_features['rolling_sharpe'] = (
            df_features['return_5d'].rolling(20).mean() / 
            df_features['rv_20d'].replace(0, np.nan)
        )
        
        # Momentum vs mean reversion indicators
        df_features['price_position'] = (
            (df_features['close'] - df_features['close'].rolling(20).min()) /
            (df_features['close'].rolling(20).max() - df_features['close'].rolling(20).min())
        ).clip(0, 1)
        
        # Regime features
        df_features['vol_regime'] = (
            df_features['rv_20d'] / df_features['rv_20d'].rolling(252).mean()
        )
        
        return df_features
    
    async def train_model(self, symbols: list, lookback_days: int = 730) -> bool:
        """Train the forecasting model"""
        try:
            all_features = []
            all_targets = []
            
            for symbol in symbols:
                logger.info(f"📊 Preparing training data for {symbol}")
                
                # Get historical data
                df = self.data_pipeline.get_data_for_analysis(
                    symbol=symbol,
                    data_type='crypto_ohlcv',
                    start_date=(datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d'),
                    create_features=True
                )
                
                if df is None or len(df) < self.min_samples:
                    logger.warning(f"Not enough data for {symbol}")
                    continue
                
                # Add microstructure features
                df = self._add_microstructure_features(df, 5)
                
                # Create targets (future returns)
                for horizon in [1, 3, 5, 10]:
                    df[f'target_{horizon}d'] = (
                        np.log(df['close'].shift(-horizon) / df['close'])
                    )
                
                # Prepare features and targets
                feature_cols = [col for col in df.columns if not col.startswith('target_')]
                target_cols = [col for col in df.columns if col.startswith('target_')]
                
                if len(feature_cols) > 0 and len(target_cols) > 0:
                    X = df[feature_cols].dropna()
                    y_multi = df[target_cols].loc[X.index]  # Align indices
                    
                    if len(X) > 0 and len(y_multi) > 0:
                        all_features.append(X)
                        all_targets.append(y_multi)
            
            if not all_features:
                logger.error("No valid training data found")
                return False
            
            # Combine all data
            X_combined = pd.concat(all_features, axis=0)
            y_combined = pd.concat(all_targets, axis=0)
            
            # Train quantile model
            self.model.fit(X_combined, y_combined)
            self.is_trained = True
            
            logger.info(f"✅ Forecast model trained with {len(X_combined)} samples")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model training failed: {e}")
            return False