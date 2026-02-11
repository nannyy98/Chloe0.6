#!/usr/bin/env python3
"""
Chloe AI - Crypto & Stock Market Analysis Agent
Main entry point for the application
"""

import asyncio
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

# Import our modules
from data.data_agent import DataAgent
from indicators.indicator_calculator import IndicatorCalculator
from features.advanced_features import AdvancedFeatureEngineer
from models.enhanced_ml_core import EnhancedMLCore, SignalInterpreter
from risk.basic_risk_engine import RiskEngine
from llm.chloe_llm import ChloeLLM
from backtest.backtester import Backtester
from agents.market_agent import MarketAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class ChloeOrchestrator:
    """
    Main orchestrator for Chloe AI system
    Coordinates all components to provide market analysis and trading signals
    """
    
    def __init__(self, use_advanced_mode: bool = True):
        self.use_advanced_mode = use_advanced_mode
        
        if use_advanced_mode:
            # Use advanced components
            self.data_agent = DataAgent()
            self.indicator_calc = IndicatorCalculator()
            self.ml_core = EnhancedMLCore()
            self.signal_interpreter = SignalInterpreter()
            self.risk_engine = RiskEngine()
            self.chloe_llm = ChloeLLM()
            self.backtester = Backtester()
            logger.info("🤖 Chloe AI Orchestrator initialized (Advanced Mode)")
        else:
            # Use basic components
            self.data_agent = DataAgent()
            self.indicator_calc = IndicatorCalculator()
            self.ml_core = MLSignalsCore()
            self.signal_processor = SignalProcessor()
            self.risk_engine = RiskEngine()
            self.chloe_llm = ChloeLLM()
            self.backtester = Backtester()
            logger.info("🤖 Chloe AI Orchestrator initialized (Basic Mode)")
    
    async def analyze_market(self, symbol: str, days: int = 365) -> Dict:
        """
        Comprehensive market analysis for a symbol
        
        Args:
            symbol: Trading symbol to analyze
            days: Number of days of historical data to use
            
        Returns:
            Dictionary with analysis results
        """
        logger.info(f"🔍 Analyzing market for {symbol}")
        
        # Step 1: Fetch data
        logger.info("📊 Fetching market data...")
        data = await self.data_agent.fetch_crypto_ohlcv(symbol, timeframe='1d', limit=days)
        
        if data is None:
            # Try as stock
            data = await self.data_agent.fetch_stock_ohlcv(symbol, period=f'{days}d', interval='1d')
            
        if data is None:
            raise ValueError(f"No data found for symbol: {symbol}")
        
        logger.info(f"✅ Fetched {len(data)} data points for {symbol}")
        
        # Step 2: Calculate indicators
        logger.info("📈 Calculating technical indicators...")
        data_with_indicators = self.indicator_calc.calculate_all_indicators(data)
        logger.info("✅ Technical indicators calculated")
        
        # Step 3: Prepare features for ML model
        logger.info("⚙️ Preparing features for ML model...")
        X, y = self.ml_core.prepare_features_and_target(data_with_indicators, lookahead_period=5)
        
        if len(X) < 10:
            logger.warning("⚠️ Insufficient data for ML model training")
            # Return just the indicators analysis
            latest_indicators = {}
            for indicator in ['rsi_14', 'macd', 'ema_20', 'ema_50', 'volatility']:
                if indicator in data_with_indicators.columns:
                    val = data_with_indicators[indicator].iloc[-1]
                    if pd.notna(val):
                        latest_indicators[indicator] = round(float(val), 4)
            
            current_price = data['close'].iloc[-1] if 'close' in data.columns else data['Close'].iloc[-1]
            
            return {
                "symbol": symbol,
                "current_price": current_price,
                "indicators": latest_indicators,
                "signal": "INSUFFICIENT_DATA",
                "confidence": 0.0,
                "analysis_timestamp": datetime.now().isoformat(),
                "message": "Not enough data to generate ML-based signals. Showing technical indicators only."
            }
        
        # Step 4: Train ML model
        logger.info("🧠 Training ML model...")
        self.ml_core.train(X, y)
        logger.info("✅ ML model trained")
        
        # Step 5: Generate predictions
        logger.info("🔮 Generating trading signals...")
        latest_features = X.tail(1)
        predictions, probabilities = self.ml_core.predict_with_confidence(latest_features)
        
        # Process predictions based on mode
        if self.use_advanced_mode:
            signals_df = self.signal_interpreter.interpret_predictions(predictions, probabilities)
            signal = signals_df['signal'].iloc[0]
            confidence = signals_df['confidence'].iloc[0]
        else:
            signals_df = self.signal_processor.process_predictions(predictions, probabilities)
            signal = signals_df['signal'].iloc[0]
            confidence = signals_df['confidence'].iloc[0]
        
        logger.info(f"✅ Generated signal: {signal} with {confidence:.2f} confidence")
        
        # Step 6: Calculate risk metrics
        current_price = data['close'].iloc[-1] if 'close' in data.columns else data['Close'].iloc[-1]
        volatility = data['close'].pct_change().rolling(20).std().iloc[-1] if 'close' in data.columns else data['Close'].pct_change().rolling(20).std().iloc[-1]
        
        # Step 7: Calculate stop loss and take profit
        atr = (data['high'] - data['low']).rolling(14).mean().iloc[-1] if 'high' in data.columns and 'low' in data.columns else volatility * current_price * 3
        
        # Map signal to compatible format for risk engine
        if signal in ['STRONG_BUY', 'BUY']:
            risk_engine_signal = 'BUY'
        elif signal in ['STRONG_SELL', 'SELL']:
            risk_engine_signal = 'SELL'
        else:  # HOLD
            risk_engine_signal = 'HOLD'
        
        # Only calculate stop loss and take profit for BUY and SELL signals
        if risk_engine_signal in ['BUY', 'SELL']:
            stop_loss, take_profit = self.risk_engine.calculate_stop_loss_take_profit(current_price, risk_engine_signal, atr)
        else:
            # For HOLD signals, set neutral values
            stop_loss = current_price
            take_profit = current_price
        
        # Step 8: Calculate position size
        position_size = self.risk_engine.calculate_position_size(current_price, stop_loss, 10000, 0.02)  # Assuming $10k account
        
        # Step 9: Assess risk level
        if confidence > 0.8 and volatility < 0.03:
            risk_level = "LOW"
        elif confidence > 0.6 or volatility > 0.05:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
        

        
        # Step 10: Generate explanation using LLM
        technical_data = {col: data_with_indicators[col].iloc[-1] for col in data_with_indicators.columns if pd.api.types.is_numeric_dtype(data_with_indicators[col])}
        risk_data = {'volatility': volatility}
        
        analysis = self.chloe_llm.analyze_signal(
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            technical_data=technical_data,
            risk_data=risk_data
        )
        
        # Step 11: Validate trade
        is_valid, reason, validation_details = self.risk_engine.validate_trade(
            signal=self._create_trade_signal_obj(
                symbol, risk_engine_signal, confidence, current_price, stop_loss, take_profit, risk_level, volatility
            )
        )
        
        result = {
            "symbol": symbol,
            "current_price": current_price,
            "signal": signal,
            "confidence": confidence,
            "risk_level": risk_level,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "position_size": position_size,
            "valid_trade": is_valid,
            "validation_reason": reason,
            "explanation": analysis.explanation,
            "suggested_action": analysis.suggested_action,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        # Add key indicators
        result["indicators"] = {}
        for indicator in ['rsi_14', 'macd', 'ema_20', 'ema_50', 'volatility']:
            if indicator in data_with_indicators.columns:
                val = data_with_indicators[indicator].iloc[-1]
                if pd.notna(val):
                    result["indicators"][indicator] = round(float(val), 4)
        
        logger.info(f"✅ Complete analysis for {symbol} finished")
        return result
    
    def _create_trade_signal_obj(self, symbol: str, signal: str, confidence: float, entry_price: float, 
                               stop_loss: float, take_profit: float, risk_level_str: str, volatility: float):
        """Helper to create a TradeSignal-like object"""
        from risk.risk_engine import TradeSignal, RiskLevel
        risk_level = RiskLevel[risk_level_str] if risk_level_str in RiskLevel.__members__ else RiskLevel.MEDIUM
        return TradeSignal(
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=0.0,
            risk_level=risk_level,
            volatility=volatility,
            timestamp=pd.Timestamp.now()
        )
    
    async def backtest_strategy(self, symbol: str, days: int = 365) -> Dict:
        """
        Backtest the ML strategy on historical data
        
        Args:
            symbol: Trading symbol to backtest
            days: Number of days of historical data to use
            
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"🔄 Backtesting strategy for {symbol}")
        
        # Fetch data
        data = await self.data_agent.fetch_crypto_ohlcv(symbol, timeframe='1d', limit=days)
        if data is None:
            data = await self.data_agent.fetch_stock_ohlcv(symbol, period=f'{days}d', interval='1d')
        
        if data is None:
            raise ValueError(f"No data found for symbol: {symbol}")
        
        # Calculate indicators
        data_with_indicators = self.indicator_calc.calculate_all_indicators(data)
        
        # Generate simple signals based on RSI for demonstration
        # In a real scenario, you'd use your trained ML model
        signals = pd.Series(0, index=data.index)  # Default to hold
        
        # Simple RSI-based signals
        if 'rsi_14' in data_with_indicators.columns:
            rsi = data_with_indicators['rsi_14']
            signals[rsi < 30] = 1  # Buy when RSI < 30 (oversold)
            signals[rsi > 70] = -1  # Sell when RSI > 70 (overbought)
        
        # Run backtest
        results = self.backtester.run_backtest(data, signals)
        
        logger.info(f"✅ Backtest for {symbol} completed")
        return results
    
    async def run_monitoring_cycle(self, symbols: List[str]):
        """
        Run a monitoring cycle for multiple symbols
        
        Args:
            symbols: List of symbols to monitor
        """
        logger.info(f"👀 Starting monitoring cycle for {len(symbols)} symbols")
        
        results = {}
        for symbol in symbols:
            try:
                result = await self.analyze_market(symbol, days=90)  # Use shorter history for monitoring
                results[symbol] = result
                logger.info(f"✅ Completed analysis for {symbol}")
            except Exception as e:
                logger.error(f"❌ Error analyzing {symbol}: {e}")
                results[symbol] = {"error": str(e), "symbol": symbol}
        
        logger.info("✅ Monitoring cycle completed")
        return results

async def main():
    """Main entry point for Chloe AI"""
    logger.info("🚀 Starting Chloe AI - Market Analysis Agent")
    logger.info(f"Starting time: {datetime.now()}")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Chloe AI - Market Analysis Agent')
    parser.add_argument('--mode', choices=['analyze', 'monitor', 'backtest'], default='analyze',
                       help='Operation mode: analyze single symbol, monitor multiple, or backtest')
    parser.add_argument('--symbol', type=str, help='Symbol to analyze (e.g., BTC/USDT)')
    parser.add_argument('--symbols', type=str, nargs='+', help='Multiple symbols to monitor')
    parser.add_argument('--days', type=int, default=365, help='Days of historical data to use')
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    chloe = ChloeOrchestrator()
    
    try:
        if args.mode == 'analyze':
            if not args.symbol:
                print("❌ Please specify a symbol to analyze using --symbol")
                return
            
            result = await chloe.analyze_market(args.symbol, args.days)
            print(f"\n📊 Analysis Result for {result['symbol']}:")
            print(f"Current Price: ${result['current_price']:.2f}")
            print(f"Signal: {result['signal']}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Risk Level: {result['risk_level']}")
            if 'stop_loss' in result:
                print(f"Stop Loss: ${result['stop_loss']:.2f}")
            if 'take_profit' in result:
                print(f"Take Profit: ${result['take_profit']:.2f}")
            print(f"Explanation: {result['explanation'][:200]}...")
            
        elif args.mode == 'monitor':
            symbols = args.symbols or [args.symbol] if args.symbol else ['BTC/USDT', 'ETH/USDT']
            results = await chloe.run_monitoring_cycle(symbols)
            
            print(f"\n👀 Monitoring Results for {len(results)} symbols:")
            for symbol, result in results.items():
                if 'error' in result:
                    print(f"❌ {symbol}: {result['error']}")
                else:
                    print(f"✅ {symbol}: {result['signal']} ({result['confidence']:.2f})")
                    
        elif args.mode == 'backtest':
            if not args.symbol:
                print("❌ Please specify a symbol to backtest using --symbol")
                return
                
            results = await chloe.backtest_strategy(args.symbol, args.days)
            print(f"\n🔄 Backtest Results for {args.symbol}:")
            print(f"Initial Capital: ${results['initial_capital']:,.2f}")
            print(f"Final Capital: ${results['final_capital']:,.2f}")
            print(f"Total Return: {results['total_return']:.2%}")
            print(f"Annualized Return: {results['annualized_return']:.2%}")
            print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {results['max_drawdown']:.2%}")
            print(f"Win Rate: {results['win_rate']:.2%}")
    
    except KeyboardInterrupt:
        logger.info("🛑 Chloe AI stopped by user")
    except Exception as e:
        logger.error(f"❌ Chloe AI encountered an error: {e}")
        raise
    
    logger.info("👋 Chloe AI shutting down gracefully")

if __name__ == "__main__":
    asyncio.run(main())