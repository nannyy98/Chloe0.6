"""
Market Agent for Chloe AI
Coordinates data collection, analysis, and signal generation
"""

import asyncio
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time

# Import core components
from data.data_agent import DataAgent
from indicators.indicator_calculator import IndicatorCalculator
from features.advanced_features import AdvancedFeatureEngineer
from models.enhanced_ml_core import EnhancedMLCore, SignalInterpreter
from risk.risk_engine import RiskEngine
from llm.chloe_llm import ChloeLLM

logger = logging.getLogger(__name__)

class MarketAgent:
    """
    Main market agent that orchestrates the analysis pipeline
    """
    
    def __init__(self, symbols: List[str] = None):
        self.symbols = symbols or ['BTC/USDT', 'ETH/USDT']
        self.data_agent = DataAgent()
        self.indicator_calc = IndicatorCalculator()
        self.feature_engineer = AdvancedFeatureEngineer()
        self.ml_core = EnhancedMLCore(model_type='ensemble')
        self.signal_interpreter = SignalInterpreter()
        self.risk_engine = RiskEngine()
        self.chloe_llm = ChloeLLM()
        
        self.is_trained = False
        self.last_training_time = None
        self.training_data_cache = {}
        self.analysis_cache = {}
        self.min_training_samples = 100
        
        logger.info("📈 Market Agent initialized")
    
    async def collect_and_process_data(self, symbol: str, days: int = 365) -> Optional[pd.DataFrame]:
        """
        Collect and process market data for a symbol
        
        Args:
            symbol: Trading symbol
            days: Number of days of historical data
            
        Returns:
            Processed DataFrame with all features
        """
        logger.info(f"📊 Collecting data for {symbol}...")
        
        try:
            # Fetch data
            data = await self.data_agent.fetch_crypto_ohlcv(symbol, timeframe='1d', limit=days)
            
            if data is None:
                # Try as stock
                data = await self.data_agent.fetch_stock_ohlcv(symbol, period=f'{days}d', interval='1d')
            
            if data is None:
                logger.error(f"No data available for {symbol}")
                return None
            
            # Basic indicator calculation
            logger.info(f"📈 Calculating basic indicators for {symbol}...")
            data_with_indicators = self.indicator_calc.calculate_all_indicators(data)
            
            # Advanced feature engineering
            logger.info(f"🔧 Creating advanced features for {symbol}...")
            enhanced_data = self.feature_engineer.create_all_advanced_features(data_with_indicators)
            
            logger.info(f"✅ Data processing completed for {symbol} ({len(enhanced_data)} samples)")
            return enhanced_data
            
        except Exception as e:
            logger.error(f"❌ Error processing data for {symbol}: {e}")
            return None
    
    async def train_models(self, force_retrain: bool = False) -> bool:
        """
        Train ML models on available data
        
        Args:
            force_retrain: Force retraining even if recently trained
            
        Returns:
            True if training successful, False otherwise
        """
        # Check if retraining is needed
        if not force_retrain and self.is_trained:
            if self.last_training_time and (datetime.now() - self.last_training_time).days < 1:
                logger.info("⏭️  Skipping training - models recently trained")
                return True
        
        logger.info("🧠 Starting model training...")
        
        try:
            # Collect data for all symbols
            all_data = {}
            for symbol in self.symbols:
                data = await self.collect_and_process_data(symbol, days=730)  # 2 years
                if data is not None and len(data) >= self.min_training_samples:
                    all_data[symbol] = data
                    self.training_data_cache[symbol] = data
            
            if not all_data:
                logger.error("❌ No valid training data available")
                return False
            
            # Train separate models for each symbol (could be enhanced later)
            training_results = {}
            for symbol, data in all_data.items():
                try:
                    logger.info(f"🎯 Training model for {symbol}...")
                    
                    # Prepare features and targets
                    X, y = self.ml_core.prepare_features_and_target(data, lookahead_period=5)
                    
                    if len(X) >= self.min_training_samples:
                        # Train model
                        self.ml_core.train(X, y, cv_folds=3)
                        training_results[symbol] = True
                        logger.info(f"✅ Model trained for {symbol}")
                    else:
                        logger.warning(f"⚠️  Insufficient data for {symbol} ({len(X)} samples)")
                        training_results[symbol] = False
                        
                except Exception as e:
                    logger.error(f"❌ Training failed for {symbol}: {e}")
                    training_results[symbol] = False
            
            # Check if any models were successfully trained
            successful_trainings = sum(training_results.values())
            if successful_trainings > 0:
                self.is_trained = True
                self.last_training_time = datetime.now()
                logger.info(f"✅ Training completed - {successful_trainings}/{len(self.symbols)} models trained")
                return True
            else:
                logger.error("❌ All training attempts failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Training process failed: {e}")
            return False
    
    async def analyze_symbol(self, symbol: str, force_training: bool = False) -> Optional[Dict]:
        """
        Complete analysis for a single symbol
        
        Args:
            symbol: Trading symbol to analyze
            force_training: Force model training before analysis
            
        Returns:
            Dictionary with complete analysis results
        """
        logger.info(f"🔍 Analyzing {symbol}...")
        
        try:
            # Get or collect data
            if symbol in self.training_data_cache:
                data = self.training_data_cache[symbol]
                logger.info(f"🔄 Using cached data for {symbol}")
            else:
                data = await self.collect_and_process_data(symbol, days=365)
                if data is None:
                    return None
            
            # Ensure model is trained
            if not self.is_trained or force_training:
                training_success = await self.train_models(force_retrain=force_training)
                if not training_success:
                    logger.warning("⚠️  Proceeding with analysis without model predictions")
            
            # Get current price and latest indicators
            current_price = data['close'].iloc[-1] if 'close' in data.columns else data['Close'].iloc[-1]
            
            # Calculate volatility for risk assessment
            volatility = data['close'].pct_change().rolling(20).std().iloc[-1] if 'close' in data.columns else data['Close'].pct_change().rolling(20).std().iloc[-1]
            
            # Get latest features for prediction
            latest_features = data[self.ml_core.selected_features].tail(10)  # Use last 10 samples
            
            # Generate ML predictions if model is trained
            signal_prediction = None
            confidence = 0.0
            
            if self.is_trained and len(latest_features) > 0:
                try:
                    predictions, confidences = self.ml_core.predict_with_confidence(latest_features)
                    interpreted_signals = self.signal_interpreter.interpret_predictions(predictions, confidences)
                    
                    # Get most recent signal
                    latest_signal = interpreted_signals.iloc[-1]
                    signal_prediction = latest_signal['signal']
                    confidence = latest_signal['confidence']
                    
                    logger.info(f"🤖 ML Prediction for {symbol}: {signal_prediction} ({confidence:.2f})")
                    
                except Exception as e:
                    logger.warning(f"⚠️  ML prediction failed: {e}")
                    signal_prediction = None
            
            # Calculate risk parameters
            atr = volatility * current_price * 3  # Approximate ATR
            stop_loss, take_profit = self.risk_engine.calculate_stop_loss_take_profit(
                current_price, signal_prediction or 'HOLD', atr
            )
            position_size = self.risk_engine.calculate_position_size(
                current_price, stop_loss, 10000, 0.02  # Assuming $10k account
            )
            
            # Create technical data summary for LLM
            technical_data = {}
            for col in data.columns:
                if col.startswith(('rsi', 'macd', 'ema', 'bb_', 'stoch', 'volatility')):
                    val = data[col].iloc[-1]
                    if pd.notna(val):
                        technical_data[col] = round(float(val), 4)
            
            risk_data = {
                'volatility': volatility,
                'position_size': position_size,
                'stop_loss_distance': abs(current_price - stop_loss) / current_price
            }
            
            # Generate LLM explanation
            if signal_prediction:
                analysis = self.chloe_llm.analyze_signal(
                    symbol=symbol,
                    signal=signal_prediction,
                    confidence=confidence,
                    technical_data=technical_data,
                    risk_data=risk_data
                )
                explanation = analysis.explanation
                suggested_action = analysis.suggested_action
            else:
                explanation = "Insufficient data for ML prediction. Based on technical indicators only."
                suggested_action = "Monitor market conditions and wait for clearer signals."
            
            # Compile results
            result = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'current_price': float(current_price),
                'signal': signal_prediction or 'INSUFFICIENT_DATA',
                'confidence': float(confidence),
                'stop_loss': float(stop_loss),
                'take_profit': float(take_profit),
                'position_size': float(position_size),
                'volatility': float(volatility),
                'technical_indicators': technical_data,
                'explanation': explanation,
                'suggested_action': suggested_action,
                'model_trained': self.is_trained,
                'data_points': len(data)
            }
            
            # Cache the analysis
            self.analysis_cache[symbol] = result
            
            logger.info(f"✅ Analysis completed for {symbol}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {symbol}: {e}")
            return None
    
    async def monitor_market(self, interval_minutes: int = 60) -> None:
        """
        Continuous market monitoring
        
        Args:
            interval_minutes: Monitoring interval in minutes
        """
        logger.info(f"👀 Starting market monitoring (interval: {interval_minutes} minutes)")
        
        while True:
            try:
                logger.info(f"🔄 Market monitoring cycle started at {datetime.now()}")
                
                # Analyze all symbols
                results = {}
                for symbol in self.symbols:
                    result = await self.analyze_symbol(symbol)
                    if result:
                        results[symbol] = result
                        # Log significant signals
                        if result['signal'] in ['BUY', 'SELL', 'STRONG_BUY', 'STRONG_SELL'] and result['confidence'] > 0.7:
                            logger.info(f"🚨 Significant signal: {symbol} - {result['signal']} ({result['confidence']:.2f})")
                
                # Save results
                self._save_monitoring_results(results)
                
                logger.info(f"✅ Monitoring cycle completed. Next update in {interval_minutes} minutes")
                
                # Wait for next cycle
                await asyncio.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                logger.info("🛑 Market monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Error in monitoring cycle: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    def _save_monitoring_results(self, results: Dict):
        """Save monitoring results to cache"""
        try:
            # In a real implementation, this would save to database or file
            # For now, we'll just update the cache
            for symbol, result in results.items():
                self.analysis_cache[symbol] = result
                
        except Exception as e:
            logger.error(f"❌ Error saving monitoring results: {e}")
    
    async def get_portfolio_analysis(self) -> Dict:
        """
        Get comprehensive analysis for all monitored symbols
        
        Returns:
            Dictionary with portfolio analysis
        """
        logger.info("📊 Generating portfolio analysis...")
        
        portfolio_results = {}
        signals_summary = {'BUY': 0, 'SELL': 0, 'HOLD': 0, 'STRONG_BUY': 0, 'STRONG_SELL': 0}
        
        for symbol in self.symbols:
            result = await self.analyze_symbol(symbol)
            if result:
                portfolio_results[symbol] = result
                signal = result['signal']
                if signal in signals_summary:
                    signals_summary[signal] += 1
        
        # Calculate portfolio metrics
        total_symbols = len(portfolio_results)
        if total_symbols > 0:
            avg_confidence = sum(result['confidence'] for result in portfolio_results.values()) / total_symbols
            avg_volatility = sum(result['volatility'] for result in portfolio_results.values()) / total_symbols
        else:
            avg_confidence = 0
            avg_volatility = 0
        
        portfolio_analysis = {
            'timestamp': datetime.now().isoformat(),
            'symbols_analyzed': total_symbols,
            'signals_summary': signals_summary,
            'average_confidence': avg_confidence,
            'average_volatility': avg_volatility,
            'market_sentiment': self._calculate_market_sentiment(signals_summary),
            'individual_analysis': portfolio_results
        }
        
        logger.info("✅ Portfolio analysis completed")
        return portfolio_analysis
    
    def _calculate_market_sentiment(self, signals_summary: Dict) -> str:
        """Calculate overall market sentiment"""
        buy_signals = signals_summary.get('BUY', 0) + signals_summary.get('STRONG_BUY', 0)
        sell_signals = signals_summary.get('SELL', 0) + signals_summary.get('STRONG_SELL', 0)
        hold_signals = signals_summary.get('HOLD', 0)
        
        total_signals = buy_signals + sell_signals + hold_signals
        
        if total_signals == 0:
            return "NEUTRAL"
        
        buy_ratio = buy_signals / total_signals
        sell_ratio = sell_signals / total_signals
        
        if buy_ratio > 0.6:
            return "BULLISH"
        elif sell_ratio > 0.6:
            return "BEARISH"
        elif buy_ratio > sell_ratio:
            return "MODERATELY_BULLISH"
        elif sell_ratio > buy_ratio:
            return "MODERATELY_BEARISH"
        else:
            return "NEUTRAL"

# Example usage
async def main():
    """Example usage of Market Agent"""
    # Initialize with common crypto symbols
    agent = MarketAgent(symbols=['BTC/USDT', 'ETH/USDT'])
    
    # Train models
    print("Training models...")
    await agent.train_models()
    
    # Analyze a single symbol
    print("\nAnalyzing BTC/USDT...")
    btc_analysis = await agent.analyze_symbol('BTC/USDT')
    if btc_analysis:
        print(f"Signal: {btc_analysis['signal']}")
        print(f"Confidence: {btc_analysis['confidence']:.2f}")
        print(f"Current Price: ${btc_analysis['current_price']:.2f}")
    
    # Get portfolio analysis
    print("\nPortfolio Analysis...")
    portfolio = await agent.get_portfolio_analysis()
    print(f"Market Sentiment: {portfolio['market_sentiment']}")
    print(f"Signals Summary: {portfolio['signals_summary']}")

if __name__ == "__main__":
    asyncio.run(main())