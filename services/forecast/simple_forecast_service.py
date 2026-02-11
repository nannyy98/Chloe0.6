"""
Simplified Forecast Service for Chloe AI Demo
Lightweight version without external ML dependencies
"""
import logging
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass

from core.event_bus import Event, EventType
from feature_store.feature_calculator import get_feature_calculator

logger = logging.getLogger(__name__)

@dataclass
class ForecastEvent:
    """Event for market forecasts"""
    symbol: str
    horizon: int  # forecast horizon in days
    expected_return: float
    volatility: float
    confidence: float
    p10: float  # 10th percentile
    p50: float  # 50th percentile (median)
    p90: float  # 90th percentile
    timestamp: datetime
    data: Optional[Dict] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        # self.event_type = EventType.FORECAST  # Remove this line for now

class SimpleForecastModel:
    """Simple rule-based forecast model for demo purposes"""
    
    def __init__(self):
        self.is_trained = True  # Always ready for demo
        self.feature_importance = {}
        
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """Fit model (dummy implementation)"""
        logger.info("🎯 Simple forecast model fitted")
        # Store feature importance based on correlation with target
        if len(X.columns) > 0 and len(y.columns) > 0:
            target_col = y.columns[0]
            correlations = {}
            for feature in X.columns:
                if X[feature].dtype in [np.float64, np.int64] and not X[feature].isna().all():
                    corr = X[feature].corr(y[target_col])
                    if not pd.isna(corr):
                        correlations[feature] = abs(corr)
            
            # Sort by importance
            sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
            self.feature_importance = dict(sorted_features[:20])  # Top 20 features
            
        return self
    
    def predict(self, X: pd.DataFrame) -> Optional[Dict]:
        """Generate forecast predictions"""
        if len(X) == 0:
            return None
            
        try:
            # Simple rule-based approach for demo
            latest_row = X.iloc[-1]
            
            # Use key technical indicators for prediction
            rsi = latest_row.get('rsi_14', 50)
            macd_hist = latest_row.get('macd_histogram', 0)
            bb_position = latest_row.get('bb_position', 0.5)
            volatility = latest_row.get('volatility_20', 0.02)
            
            # Simple prediction logic
            # RSI component (30% weight)
            rsi_signal = 0
            if rsi < 30:
                rsi_signal = 0.003  # Bullish oversold
            elif rsi > 70:
                rsi_signal = -0.002  # Bearish overbought
            else:
                rsi_signal = 0  # Neutral
            
            # MACD component (40% weight)
            macd_signal = 0
            if macd_hist > 0:
                macd_signal = 0.004  # Bullish momentum
            else:
                macd_signal = -0.003  # Bearish momentum
            
            # Bollinger Bands component (30% weight)
            bb_signal = 0
            if bb_position < 0.2:
                bb_signal = 0.002  # Oversold
            elif bb_position > 0.8:
                bb_signal = -0.001  # Overbought
            else:
                bb_signal = 0  # Neutral
            
            # Combine signals
            expected_return = (rsi_signal * 0.3 + macd_signal * 0.4 + bb_signal * 0.3)
            
            # Confidence based on signal strength and volatility
            signal_strength = abs(expected_return)
            confidence = min(0.95, max(0.1, signal_strength * 50 + (1 - volatility * 10)))
            
            # Calculate percentiles (simple approach)
            std_dev = volatility * np.sqrt(5)  # 5-day horizon
            p50 = expected_return
            p10 = p50 - 1.28 * std_dev  # 10th percentile
            p90 = p50 + 1.28 * std_dev  # 90th percentile
            
            return {
                'expected_return': expected_return,
                'volatility': volatility,
                'confidence': confidence,
                'p10': p10,
                'p50': p50,
                'p90': p90
            }
            
        except Exception as e:
            logger.error(f"❌ Forecast prediction failed: {e}")
            return None

class ForecastService:
    """Simplified forecast service for the unified pipeline"""
    
    def __init__(self):
        self.model = SimpleForecastModel()
        self.feature_calculator = get_feature_calculator()
        self.is_trained = True
        self.min_samples = 50  # Reduced for demo
        
        logger.info("🔮 Simplified Forecast Service initialized")
    
    async def generate_forecast(self, symbol: str, horizon: int = 5) -> Optional[ForecastEvent]:
        """Generate probabilistic forecast for a symbol"""
        try:
            # Get features from feature store
            features_df = await self._get_features_for_forecast(symbol, horizon)
            if features_df is None or len(features_df) < self.min_samples:
                logger.warning(f"Insufficient data for {symbol} forecast")
                return None
            
            # Get latest features
            latest_features = features_df.tail(1).dropna(axis=1)
            if len(latest_features.columns) < 5:  # Need minimum features
                logger.warning(f"Insufficient features for {symbol}")
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
            # For demo, we'll create synthetic data or use existing data structures
            # In production, this would fetch from the data pipeline
            from data.data_agent import DataAgent
            data_agent = DataAgent()
            
            # Try to get real data
            df = await data_agent.fetch_crypto_ohlcv(symbol, timeframe='1d', limit=200)
            if df is None:
                df = await data_agent.fetch_stock_ohlcv(symbol, period='200d', interval='1d')
            
            if df is None:
                # Create minimal synthetic data for demo
                dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
                prices = 50000 + np.cumsum(np.random.normal(0, 100, 100))
                df = pd.DataFrame({
                    'timestamp': dates,
                    'open': prices * (1 + np.random.normal(0, 0.01, 100)),
                    'high': prices * (1 + abs(np.random.normal(0, 0.02, 100))),
                    'low': prices * (1 - abs(np.random.normal(0, 0.02, 100))),
                    'close': prices,
                    'volume': np.random.uniform(1000, 10000, 100)
                })
                df.set_index('timestamp', inplace=True)
            
            # Calculate features using feature calculator
            df_with_features = self.feature_calculator.calculate_all_features(df, symbol=symbol)
            return df_with_features
            
        except Exception as e:
            logger.error(f"❌ Error getting features for {symbol}: {e}")
            return None
    
    async def train_model(self, symbols: list, lookback_days: int = 365) -> bool:
        """Train the forecasting model (dummy implementation for demo)"""
        try:
            logger.info(f"🎯 Training forecast model on {len(symbols)} symbols")
            
            # For demo, we'll just initialize the model
            # In production, this would train on historical data
            self.model.is_trained = True
            
            logger.info("✅ Forecast model ready for demo")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model training failed: {e}")
            return False
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the model"""
        return self.model.feature_importance

# Backward compatibility
QuantileModel = SimpleForecastModel