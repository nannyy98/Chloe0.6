"""
Real-time Data Processing Pipeline for Chloe AI
Processes streaming market data and generates real-time signals
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from collections import deque
import threading
from dataclasses import dataclass

# Import our modules
from realtime.websocket_client import WebSocketClient, MarketData, OrderBook
from indicators.indicator_calculator import IndicatorCalculator
from models.enhanced_ml_core import EnhancedMLCore, SignalInterpreter
from risk.risk_engine import RiskEngine
from llm.chloe_llm import ChloeLLM

logger = logging.getLogger(__name__)

@dataclass
class RealTimeSignal:
    """Real-time trading signal"""
    symbol: str
    signal: str
    confidence: float
    price: float
    timestamp: datetime
    indicators: Dict[str, float]
    risk_level: str
    explanation: str

class RealTimeDataPipeline:
    """
    Real-time data processing pipeline
    """
    
    def __init__(self, symbols: List[str] = None):
        self.symbols = symbols or ['BTCUSDT', 'ETHUSDT']
        self.websocket_client = WebSocketClient('binance')
        self.indicator_calc = IndicatorCalculator()
        self.ml_core = EnhancedMLCore(model_type='ensemble')
        self.signal_interpreter = SignalInterpreter()
        self.risk_engine = RiskEngine()
        self.chloe_llm = ChloeLLM()
        
        # Data buffers
        self.price_buffers = {symbol: deque(maxlen=1000) for symbol in self.symbols}
        self.indicator_buffers = {symbol: {} for symbol in self.symbols}
        self.signal_buffers = {symbol: deque(maxlen=100) for symbol in self.symbols}
        
        # Processing settings
        self.indicator_update_interval = 10  # seconds
        self.ml_update_interval = 30  # seconds
        self.signal_update_interval = 5  # seconds
        
        # State tracking
        self.is_processing = False
        self.last_indicator_update = {}
        self.last_ml_update = {}
        self.last_signal_update = {}
        
        # Callbacks
        self.signal_callbacks = []
        self.alert_callbacks = []
        
        # Initialize buffers with historical data
        self._initialize_buffers()
        
        logger.info("🔄 Real-time data pipeline initialized")
    
    def _initialize_buffers(self):
        """Initialize data buffers with historical data"""
        logger.info("📊 Initializing data buffers with historical data...")
        
        # This would typically load recent historical data
        # For now, we'll create some sample data
        for symbol in self.symbols:
            # Create sample historical data for initialization
            dates = pd.date_range(end=datetime.now(), periods=200, freq='1min')
            prices = 45000 + np.cumsum(np.random.randn(200) * 10)
            
            sample_data = []
            for i, (date, price) in enumerate(zip(dates, prices)):
                market_data = MarketData(
                    symbol=symbol,
                    price=price,
                    timestamp=date,
                    volume=np.random.randint(100, 1000),
                    buy_volume=np.random.randint(50, 500),
                    sell_volume=np.random.randint(50, 500),
                    price_change=np.random.randn() * 5,
                    price_change_percent=np.random.randn() * 0.1
                )
                sample_data.append(market_data)
                self.price_buffers[symbol].append(market_data)
            
            self.last_indicator_update[symbol] = datetime.now()
            self.last_ml_update[symbol] = datetime.now()
            self.last_signal_update[symbol] = datetime.now()
            
        logger.info("✅ Data buffers initialized")
    
    def add_signal_callback(self, callback: Callable[[RealTimeSignal], None]):
        """Add callback for real-time signals"""
        self.signal_callbacks.append(callback)
    
    def add_alert_callback(self, callback: Callable[[str, str, Dict], None]):
        """Add callback for alerts"""
        self.alert_callbacks.append(callback)
    
    async def start_pipeline(self):
        """Start the real-time processing pipeline"""
        logger.info("🚀 Starting real-time data pipeline...")
        
        # Setup WebSocket callbacks
        self.websocket_client.add_price_callback(self._handle_price_update)
        self.websocket_client.add_orderbook_callback(self._handle_orderbook_update)
        self.websocket_client.add_trade_callback(self._handle_trade_update)
        
        # Connect to WebSocket
        if not await self.websocket_client.connect():
            logger.error("❌ Failed to connect to WebSocket")
            return
        
        # Subscribe to data streams
        await self.websocket_client.subscribe_to_tickers(self.symbols)
        await self.websocket_client.subscribe_to_orderbook(self.symbols, depth=20)
        await self.websocket_client.subscribe_to_trades(self.symbols)
        
        # Start processing
        self.is_processing = True
        
        # Start background processing tasks
        processing_tasks = [
            self._process_indicators(),
            self._process_ml_signals(),
            self._process_trading_signals(),
            self.websocket_client.listen()
        ]
        
        try:
            await asyncio.gather(*processing_tasks)
        except Exception as e:
            logger.error(f"❌ Pipeline error: {e}")
        finally:
            self.is_processing = False
            await self.websocket_client.disconnect()
    
    def _handle_price_update(self, market_data: MarketData):
        """Handle incoming price updates"""
        try:
            # Add to price buffer
            self.price_buffers[market_data.symbol].append(market_data)
            
            # Log significant price movements
            if abs(market_data.price_change_percent) > 1.0:  # 1% movement
                logger.info(f"🚨 Significant price movement: {market_data.symbol} {market_data.price_change_percent:+.2f}%")
                
                # Trigger alerts
                for callback in self.alert_callbacks:
                    try:
                        callback(
                            'PRICE_MOVEMENT',
                            market_data.symbol,
                            {
                                'price': market_data.price,
                                'change_percent': market_data.price_change_percent,
                                'volume': market_data.volume,
                                'timestamp': market_data.timestamp.isoformat()
                            }
                        )
                    except Exception as e:
                        logger.error(f"❌ Error in alert callback: {e}")
                        
        except Exception as e:
            logger.error(f"❌ Error handling price update: {e}")
    
    def _handle_orderbook_update(self, orderbook: OrderBook):
        """Handle order book updates"""
        try:
            # Store order book data
            # This could be used for liquidity analysis, spread monitoring, etc.
            spread = orderbook.asks[0][0] - orderbook.bids[0][0]
            mid_price = (orderbook.bids[0][0] + orderbook.asks[0][0]) / 2
            
            # Log unusual spreads
            if spread / mid_price > 0.005:  # 0.5% spread
                logger.warning(f"⚠️ Wide spread detected: {orderbook.symbol} {spread/mid_price:.4f}")
                
        except Exception as e:
            logger.error(f"❌ Error handling order book update: {e}")
    
    def _handle_trade_update(self, trade_data: Dict):
        """Handle trade updates"""
        try:
            # Analyze trade flow
            symbol = trade_data['symbol']
            is_buyer_maker = trade_data.get('is_buyer_maker', False)
            
            # Could implement volume imbalance analysis here
            # For now, just log large trades
            if trade_data['quantity'] * trade_data['price'] > 100000:  # $100k+ trades
                trade_type = "BUY" if not is_buyer_maker else "SELL"
                logger.info(f"💰 Large trade: {symbol} {trade_type} ${trade_data['quantity'] * trade_data['price']:,.0f}")
                
        except Exception as e:
            logger.error(f"❌ Error handling trade update: {e}")
    
    async def _process_indicators(self):
        """Process technical indicators in real-time"""
        while self.is_processing:
            try:
                current_time = datetime.now()
                
                for symbol in self.symbols:
                    # Check if it's time to update indicators
                    last_update = self.last_indicator_update.get(symbol, datetime.min)
                    if (current_time - last_update).seconds >= self.indicator_update_interval:
                        
                        # Create DataFrame from recent price data
                        if len(self.price_buffers[symbol]) >= 50:
                            price_data = list(self.price_buffers[symbol])[-100:]  # Last 100 points
                            
                            df = pd.DataFrame([{
                                'timestamp': data.timestamp,
                                'open': data.price,
                                'high': data.price,
                                'low': data.price,
                                'close': data.price,
                                'volume': data.volume
                            } for data in price_data])
                            df.set_index('timestamp', inplace=True)
                            
                            # Calculate indicators
                            df_with_indicators = self.indicator_calc.calculate_all_indicators(df)
                            
                            # Store latest indicator values
                            latest_indicators = {}
                            for col in df_with_indicators.columns:
                                if col not in ['open', 'high', 'low', 'close', 'volume']:
                                    val = df_with_indicators[col].iloc[-1]
                                    if pd.notna(val):
                                        latest_indicators[col] = float(val)
                            
                            self.indicator_buffers[symbol] = latest_indicators
                            self.last_indicator_update[symbol] = current_time
                            
                            logger.debug(f"📊 Indicators updated for {symbol}")
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"❌ Error in indicator processing: {e}")
                await asyncio.sleep(5)
    
    async def _process_ml_signals(self):
        """Process ML signals in real-time"""
        while self.is_processing:
            try:
                current_time = datetime.now()
                
                for symbol in self.symbols:
                    # Check if it's time to update ML signals
                    last_update = self.last_ml_update.get(symbol, datetime.min)
                    if (current_time - last_update).seconds >= self.ml_update_interval:
                        
                        # Check if we have enough data and indicators
                        if (symbol in self.indicator_buffers and 
                            len(self.indicator_buffers[symbol]) > 10 and
                            len(self.price_buffers[symbol]) >= 100):
                            
                            try:
                                # Create feature vector from current indicators
                                indicators = self.indicator_buffers[symbol]
                                feature_vector = pd.DataFrame([indicators])
                                
                                # Generate ML prediction
                                predictions, confidences = self.ml_core.predict_with_confidence(feature_vector)
                                interpreted_signals = self.signal_interpreter.interpret_predictions(
                                    predictions, confidences
                                )
                                
                                # Store signal
                                if len(interpreted_signals) > 0:
                                    latest_signal = interpreted_signals.iloc[-1]
                                    self.signal_buffers[symbol].append({
                                        'signal': latest_signal['signal'],
                                        'confidence': latest_signal['confidence'],
                                        'timestamp': current_time,
                                        'indicators': indicators.copy()
                                    })
                                    
                                    self.last_ml_update[symbol] = current_time
                                    logger.debug(f"🤖 ML signal updated for {symbol}: {latest_signal['signal']}")
                                    
                            except Exception as e:
                                logger.warning(f"⚠️ ML prediction failed for {symbol}: {e}")
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"❌ Error in ML processing: {e}")
                await asyncio.sleep(10)
    
    async def _process_trading_signals(self):
        """Generate final trading signals"""
        while self.is_processing:
            try:
                current_time = datetime.now()
                
                for symbol in self.symbols:
                    # Check if it's time to generate trading signals
                    last_update = self.last_signal_update.get(symbol, datetime.min)
                    if (current_time - last_update).seconds >= self.signal_update_interval:
                        
                        # Get latest data
                        if (len(self.price_buffers[symbol]) > 0 and 
                            len(self.signal_buffers[symbol]) > 0):
                            
                            latest_price = self.price_buffers[symbol][-1]
                            latest_ml_signal = self.signal_buffers[symbol][-1]
                            
                            # Calculate risk parameters
                            current_price = latest_price.price
                            volatility = self._calculate_volatility(symbol)
                            
                            # Risk assessment
                            if volatility > 0.05:  # High volatility
                                risk_level = "HIGH"
                            elif volatility > 0.02:  # Medium volatility
                                risk_level = "MEDIUM"
                            else:
                                risk_level = "LOW"
                            
                            # Generate final signal
                            final_signal = RealTimeSignal(
                                symbol=symbol,
                                signal=latest_ml_signal['signal'],
                                confidence=latest_ml_signal['confidence'],
                                price=current_price,
                                timestamp=current_time,
                                indicators=latest_ml_signal['indicators'],
                                risk_level=risk_level,
                                explanation=f"Real-time signal based on ML analysis. Volatility: {volatility:.4f}"
                            )
                            
                            # Trigger callbacks
                            for callback in self.signal_callbacks:
                                try:
                                    callback(final_signal)
                                except Exception as e:
                                    logger.error(f"❌ Error in signal callback: {e}")
                            
                            # Log significant signals
                            if final_signal.confidence > 0.8 and final_signal.signal in ['BUY', 'SELL', 'STRONG_BUY', 'STRONG_SELL']:
                                logger.info(f"🚨 Trading Signal: {symbol} {final_signal.signal} ({final_signal.confidence:.2f})")
                            
                            self.last_signal_update[symbol] = current_time
                
                await asyncio.sleep(2)  # Check every 2 seconds
                
            except Exception as e:
                logger.error(f"❌ Error in signal processing: {e}")
                await asyncio.sleep(5)
    
    def _calculate_volatility(self, symbol: str) -> float:
        """Calculate current volatility for a symbol"""
        try:
            if len(self.price_buffers[symbol]) >= 20:
                prices = [data.price for data in list(self.price_buffers[symbol])[-20:]]
                returns = np.diff(np.log(prices))
                return np.std(returns) * np.sqrt(252)  # Annualized
            return 0.0
        except Exception:
            return 0.0
    
    def get_current_state(self, symbol: str) -> Dict:
        """Get current pipeline state for a symbol"""
        return {
            'symbol': symbol,
            'price_buffer_size': len(self.price_buffers.get(symbol, [])),
            'latest_price': self.price_buffers.get(symbol, [None])[-1].price if self.price_buffers.get(symbol) else None,
            'latest_indicators': self.indicator_buffers.get(symbol, {}),
            'latest_signal': self.signal_buffers.get(symbol, [None])[-1] if self.signal_buffers.get(symbol) else None,
            'data_quality': self.websocket_client.get_data_quality(symbol) if self.websocket_client else 0.0
        }
    
    def get_portfolio_signals(self) -> List[RealTimeSignal]:
        """Get current signals for all symbols"""
        signals = []
        for symbol in self.symbols:
            if self.signal_buffers[symbol]:
                # Get the most recent signal
                latest_signal_data = self.signal_buffers[symbol][-1]
                latest_price = self.price_buffers[symbol][-1]
                
                signal = RealTimeSignal(
                    symbol=symbol,
                    signal=latest_signal_data['signal'],
                    confidence=latest_signal_data['confidence'],
                    price=latest_price.price,
                    timestamp=latest_signal_data['timestamp'],
                    indicators=latest_signal_data['indicators'],
                    risk_level="MEDIUM",  # Would be calculated properly
                    explanation="Portfolio signal"
                )
                signals.append(signal)
        return signals

# Example usage
async def main():
    """Example usage of real-time pipeline"""
    # Initialize pipeline
    pipeline = RealTimeDataPipeline(['BTCUSDT', 'ETHUSDT'])
    
    # Add signal callback
    def signal_callback(signal: RealTimeSignal):
        print(f"🔔 Signal: {signal.symbol} - {signal.signal} ({signal.confidence:.2f}) at ${signal.price:.2f}")
    
    def alert_callback(alert_type: str, symbol: str, data: Dict):
        print(f"🚨 Alert: {alert_type} for {symbol} - {data}")
    
    pipeline.add_signal_callback(signal_callback)
    pipeline.add_alert_callback(alert_callback)
    
    # Start pipeline
    print("Starting real-time pipeline...")
    await pipeline.start_pipeline()

if __name__ == "__main__":
    asyncio.run(main())