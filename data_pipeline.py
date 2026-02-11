"""
Centralized Data Pipeline for Chloe AI
Implements the canonical data flow: market_data → feature_store → forecast → allocation
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import asyncio

from market_data.gateway import MarketDataGatewayService
from feature_store.feature_calculator import get_feature_calculator
from services.forecast.simple_forecast_service import ForecastService
from core.event_bus import event_bus, EventType, Event

logger = logging.getLogger(__name__)

class DataPipeline:
    """
    Centralized data pipeline that orchestrates the flow:
    market_data → feature_store → forecast → allocation
    """
    
    def __init__(self):
        self.market_data_service = None
        self.feature_calculator = get_feature_calculator()
        self.forecast_service = ForecastService()
        self.is_initialized = False
        self.subscribed_symbols = set()
        
    async def initialize(self, symbols: List[str], api_key: Optional[str] = None, 
                        secret: Optional[str] = None) -> bool:
        """
        Initialize the complete data pipeline
        
        Args:
            symbols: List of symbols to track
            api_key: Exchange API key
            secret: Exchange secret
            
        Returns:
            bool: True if initialization successful
        """
        try:
            logger.info("🚀 Initializing centralized data pipeline...")
            
            # For demo, skip real market data connection
            # In production, uncomment the following:
            '''
            # Initialize market data service
            self.market_data_service = MarketDataGatewayService()
            for symbol in symbols:
                self.market_data_service.subscribe_to_symbol(symbol)
            
            await self.market_data_service.setup_binance_adapter(api_key, secret)
            '''
            
            # Initialize forecast service
            await self.forecast_service.train_model(symbols)
            
            self.subscribed_symbols = set(symbols)
            self.is_initialized = True
            
            logger.info(f"✅ Data pipeline initialized for {len(symbols)} symbols (demo mode)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Data pipeline initialization failed: {e}")
            return False
    
    async def fetch_and_process_data(self, symbol: str, 
                                   lookback_days: int = 365) -> Optional[pd.DataFrame]:
        """
        Fetch raw data and process through feature store
        
        Args:
            symbol: Trading symbol
            lookback_days: Number of days of historical data
            
        Returns:
            DataFrame with processed features, or None if failed
        """
        if not self.is_initialized:
            logger.error("Pipeline not initialized")
            return None
            
        try:
            logger.info(f"📥 Fetching and processing data for {symbol}")
            
            # Fetch raw market data
            raw_data = await self._fetch_raw_data(symbol, lookback_days)
            if raw_data is None or len(raw_data) < 50:
                logger.warning(f"Insufficient data for {symbol}")
                return None
            
            # Process through feature store
            processed_data = self.feature_calculator.calculate_all_features(
                raw_data, symbol=symbol
            )
            
            logger.info(f"✅ Data processing complete for {symbol}: {len(processed_data)} rows, {len(processed_data.columns)} features")
            return processed_data
            
        except Exception as e:
            logger.error(f"❌ Data processing failed for {symbol}: {e}")
            return None
    
    async def _fetch_raw_data(self, symbol: str, lookback_days: int) -> Optional[pd.DataFrame]:
        """Fetch raw market data from exchanges"""
        try:
            # For demo, create synthetic data
            # In production, use the real data agent
            '''
            from data.data_agent import DataAgent
            data_agent = DataAgent()
            
            # Try crypto first, then stocks
            data = await data_agent.fetch_crypto_ohlcv(symbol, timeframe='1d', limit=lookback_days)
            if data is None:
                data = await data_agent.fetch_stock_ohlcv(symbol, period=f'{lookback_days}d', interval='1d')
            
            return data
            '''
            
            # Create synthetic data for demo
            dates = pd.date_range(end=datetime.now(), periods=lookback_days, freq='D')
            
            # Generate realistic price series
            initial_price = 50000 if 'BTC' in symbol else 3000
            returns = np.random.normal(0.001, 0.03, lookback_days)  # 0.1% drift, 3% volatility
            prices = initial_price * np.exp(np.cumsum(returns))
            
            # Create OHLCV data
            df = pd.DataFrame({
                'timestamp': dates,
                'open': prices * (1 + np.random.normal(0, 0.005, lookback_days)),
                'high': prices * (1 + abs(np.random.normal(0, 0.01, lookback_days))),
                'low': prices * (1 - abs(np.random.normal(0, 0.01, lookback_days))),
                'close': prices,
                'volume': np.random.uniform(1000, 10000, lookback_days)
            })
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"📊 Created synthetic data for {symbol}: {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"❌ Raw data fetch failed for {symbol}: {e}")
            return None
    
    async def generate_forecast(self, symbol: str, horizon: int = 5) -> Optional[Dict]:
        """
        Generate forecast through the centralized pipeline
        
        Args:
            symbol: Trading symbol
            horizon: Forecast horizon in days
            
        Returns:
            Dictionary with forecast results
        """
        if not self.is_initialized:
            logger.error("Pipeline not initialized")
            return None
            
        try:
            logger.info(f"🔮 Generating forecast for {symbol}")
            
            # Get processed features
            features_df = await self.fetch_and_process_data(symbol, lookback_days=730)
            if features_df is None:
                return None
            
            # Generate forecast using forecast service
            forecast_event = await self.forecast_service.generate_forecast(symbol, horizon)
            if forecast_event is None:
                return None
            
            # Return structured forecast
            forecast_result = {
                'symbol': symbol,
                'horizon': horizon,
                'expected_return': forecast_event.expected_return,
                'volatility': forecast_event.volatility,
                'confidence': forecast_event.confidence,
                'percentiles': {
                    'p10': forecast_event.p10,
                    'p50': forecast_event.p50,
                    'p90': forecast_event.p90
                },
                'timestamp': forecast_event.timestamp.isoformat() if forecast_event.timestamp else datetime.now().isoformat(),
                'latest_price': features_df['close'].iloc[-1] if 'close' in features_df.columns else None,
                'feature_count': len(features_df.columns)
            }
            
            logger.info(f"✅ Forecast generated for {symbol}: E[R]={forecast_result['expected_return']:.4f}")
            return forecast_result
            
        except Exception as e:
            logger.error(f"❌ Forecast generation failed for {symbol}: {e}")
            return None
    
    async def get_trading_signal(self, symbol: str, portfolio=None) -> Optional[Dict]:
        """
        Get complete trading signal through the pipeline
        This is the main interface for strategies
        
        Args:
            symbol: Trading symbol
            portfolio: Portfolio object for context
            
        Returns:
            Dictionary with trading signal and risk parameters
        """
        if not self.is_initialized:
            logger.error("Pipeline not initialized")
            return None
            
        try:
            logger.info(f"🎯 Generating trading signal for {symbol}")
            
            # Step 1: Generate forecast (mandatory)
            forecast = await self.generate_forecast(symbol, horizon=5)
            if forecast is None:
                logger.warning(f"No forecast available for {symbol}")
                return None
            
            # Step 2: Calculate risk parameters
            features_df = await self.fetch_and_process_data(symbol, lookback_days=365)
            if features_df is None:
                return None
                
            latest_price = features_df['close'].iloc[-1]
            volatility = forecast['volatility']
            atr = self._calculate_atr(features_df)
            
            # Step 3: Generate signal based on forecast
            signal = self._generate_signal_from_forecast(forecast, latest_price, atr)
            
            # Step 4: Calculate position sizing
            position_size = self._calculate_position_size(
                forecast, latest_price, volatility, portfolio
            )
            
            # Step 5: Calculate stops
            stop_loss, take_profit = self._calculate_stops(
                latest_price, signal['direction'], atr, forecast['confidence']
            )
            
            # Final signal structure
            complete_signal = {
                'symbol': symbol,
                'signal': signal['type'],
                'direction': signal['direction'],
                'strength': signal['strength'],
                'confidence': forecast['confidence'],
                'expected_return': forecast['expected_return'],
                'volatility': forecast['volatility'],
                'position_size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'entry_price': latest_price,
                'atr': atr,
                'timestamp': datetime.now().isoformat(),
                'source': 'forecast_pipeline'  # Indicates this came through proper pipeline
            }
            
            logger.info(f"✅ Trading signal for {symbol}: {signal['type']} (conf: {forecast['confidence']:.2f})")
            return complete_signal
            
        except Exception as e:
            logger.error(f"❌ Trading signal generation failed for {symbol}: {e}")
            return None
    
    def _generate_signal_from_forecast(self, forecast: Dict, price: float, atr: float) -> Dict:
        """Generate trading signal based on forecast"""
        expected_return = forecast['expected_return']
        confidence = forecast['confidence']
        
        # Signal strength based on expected return and confidence
        strength = expected_return * confidence * 100  # Scale for practical use
        
        # Determine signal type
        if abs(strength) < 0.1:  # Minimum threshold
            signal_type = 'HOLD'
            direction = 0
        elif strength > 0:
            signal_type = 'BUY'
            direction = 1
        else:
            signal_type = 'SELL'
            direction = -1
            
        return {
            'type': signal_type,
            'direction': direction,
            'strength': strength,
            'confidence': confidence
        }
    
    def _calculate_position_size(self, forecast: Dict, price: float, 
                               volatility: float, portfolio=None) -> float:
        """Calculate position size based on forecast and risk management"""
        confidence = forecast['confidence']
        expected_return = abs(forecast['expected_return'])
        
        # Base position sizing
        base_risk = 0.02  # 2% of capital risk per trade
        account_size = 10000  # Default account size, would come from portfolio
        
        if portfolio:
            account_size = portfolio.current_capital
            
        # Adjust position size based on confidence
        confidence_adjustment = min(2.0, max(0.5, confidence / 0.6))
        
        # Adjust for expected return magnitude
        return_adjustment = min(2.0, max(0.5, 1 + expected_return * 50))
        
        # Calculate position size
        risk_amount = account_size * base_risk
        position_value = risk_amount * confidence_adjustment * return_adjustment
        position_size = position_value / price
        
        # Cap at reasonable limits
        max_position = account_size * 0.1 / price  # Max 10% of capital
        return min(position_size, max_position)
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(df) < period + 1:
            return df['close'].iloc[-1] * 0.02  # Fallback
            
        high = df['high'] if 'high' in df.columns else df['High']
        low = df['low'] if 'low' in df.columns else df['Low']
        close = df['close'] if 'close' in df.columns else df['Close']
        
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        
        return atr if not pd.isna(atr) else close.iloc[-1] * 0.02
    
    def _calculate_stops(self, price: float, direction: int, atr: float, 
                        confidence: float) -> Tuple[Optional[float], Optional[float]]:
        """Calculate stop loss and take profit levels"""
        if direction == 0 or atr <= 0:  # HOLD signal or invalid ATR
            return None, None
            
        # Adjust stop distance based on confidence
        stop_multiplier = max(1.5, min(3.0, 2.5 / confidence))
        stop_distance = atr * stop_multiplier
        
        # Risk/reward ratio based on confidence
        rr_ratio = max(1.5, min(3.0, 1.5 + confidence))
        profit_distance = stop_distance * rr_ratio
        
        if direction > 0:  # BUY
            stop_loss = price - stop_distance
            take_profit = price + profit_distance
        else:  # SELL
            stop_loss = price + stop_distance
            take_profit = price - profit_distance
            
        return stop_loss, take_profit
    
    async def batch_process_symbols(self, symbols: List[str]) -> Dict[str, Dict]:
        """Process multiple symbols in batch"""
        results = {}
        
        for symbol in symbols:
            try:
                signal = await self.get_trading_signal(symbol)
                if signal:
                    results[symbol] = signal
                else:
                    results[symbol] = {'error': 'No signal generated'}
            except Exception as e:
                results[symbol] = {'error': str(e)}
                
        return results
    
    def get_pipeline_status(self) -> Dict:
        """Get current pipeline status"""
        return {
            'initialized': self.is_initialized,
            'subscribed_symbols': list(self.subscribed_symbols),
            'feature_calculator_ready': self.feature_calculator is not None,
            'forecast_service_trained': self.forecast_service.is_trained if hasattr(self.forecast_service, 'is_trained') else False
        }

# Global pipeline instance
data_pipeline = None

def get_data_pipeline() -> DataPipeline:
    """Get singleton data pipeline instance"""
    global data_pipeline
    if data_pipeline is None:
        data_pipeline = DataPipeline()
    return data_pipeline

# Backward compatibility functions
async def get_features_for_symbol(symbol: str, lookback_days: int = 365) -> Optional[pd.DataFrame]:
    """Backward compatibility function"""
    pipeline = get_data_pipeline()
    return await pipeline.fetch_and_process_data(symbol, lookback_days)

async def get_trading_signal_for_symbol(symbol: str) -> Optional[Dict]:
    """Backward compatibility function"""
    pipeline = get_data_pipeline()
    return await pipeline.get_trading_signal(symbol)