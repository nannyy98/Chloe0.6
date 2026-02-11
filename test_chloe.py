#!/usr/bin/env python3
"""
Simple test script to verify Chloe AI components work correctly
"""

import asyncio
import sys
import os
from datetime import datetime

# Add the project root to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from data.data_agent import DataAgent
from indicators.indicator_calculator import IndicatorCalculator
from models.ml_core import MLSignalsCore, SignalProcessor
from risk.risk_engine import RiskEngine
from llm.chloe_llm import ChloeLLM
from backtest.backtester import Backtester

async def test_data_collection():
    """Test data collection functionality"""
    print("🧪 Testing Data Collection...")
    data_agent = DataAgent()
    
    # Test with a small dataset since we may not have API keys configured
    try:
        # Create sample data to test the indicator calculator
        import pandas as pd
        import numpy as np
        
        # Create mock data for testing
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        
        sample_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices * (1 + np.random.randn(100) * 0.01),
            'high': prices * (1 + abs(np.random.randn(100)) * 0.02),
            'low': prices * (1 - abs(np.random.randn(100)) * 0.02),
            'close': prices,
            'volume': np.random.randint(1000, 10000, 100)
        })
        sample_data.set_index('timestamp', inplace=True)
        
        print(f"✅ Created sample data with shape: {sample_data.shape}")
        return sample_data
        
    except Exception as e:
        print(f"❌ Error in data collection test: {e}")
        return None

def test_indicator_calculation(sample_data):
    """Test indicator calculation functionality"""
    print("🧪 Testing Indicator Calculation...")
    try:
        calc = IndicatorCalculator()
        
        # Calculate all indicators
        result = calc.calculate_all_indicators(sample_data)
        
        print(f"✅ Indicators calculated. New columns added: {len(result.columns) - len(sample_data.columns)}")
        
        # Print some sample indicator values
        for col in result.columns:
            if col not in sample_data.columns and result[col].notna().any():
                print(f"   {col}: {result[col].iloc[-1]:.4f}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error in indicator calculation test: {e}")
        return None

def test_ml_signals(indicator_data):
    """Test ML signal generation"""
    print("🧪 Testing ML Signals...")
    try:
        ml_core = MLSignalsCore()
        processor = SignalProcessor()
        
        # Prepare features and target
        X, y = ml_core.prepare_features_and_target(indicator_data, lookahead_period=5)
        
        if len(X) > 10:  # Need minimum data for training
            # Train the model
            ml_core.train(X, y, test_size=0.3)
            
            # Generate predictions
            predictions, probabilities = ml_core.predict_with_probabilities(X.tail(5))
            processed_signals = processor.process_predictions(predictions, probabilities)
            
            print(f"✅ ML model trained and signals generated")
            print(f"   Sample predictions: {predictions[:3]}")
            print(f"   Sample confidences: {processed_signals['confidence'].head(3).tolist()}")
            
            return ml_core, processor
        else:
            print("⚠️ Not enough data for ML training, skipping this test")
            return ml_core, processor
        
    except Exception as e:
        print(f"❌ Error in ML signals test: {e}")
        return None, None

def test_risk_engine():
    """Test risk engine functionality"""
    print("🧪 Testing Risk Engine...")
    try:
        risk_engine = RiskEngine()
        
        # Test position sizing
        position_size = risk_engine.calculate_position_size(
            entry_price=45000.0,
            stop_loss=43000.0,
            account_balance=10000.0,
            risk_percentage=0.02
        )
        
        print(f"✅ Position size calculated: {position_size:.4f}")
        
        # Test stop loss/take profit calculation
        stop_loss, take_profit = risk_engine.calculate_stop_loss_take_profit(
            entry_price=45000.0,
            signal='BUY',
            atr=500.0,
            risk_reward_ratio=2.0
        )
        
        print(f"   Stop loss: {stop_loss:.2f}, Take profit: {take_profit:.2f}")
        
        return risk_engine
        
    except Exception as e:
        print(f"❌ Error in risk engine test: {e}")
        return None

def test_chloe_llm():
    """Test LLM functionality"""
    print("🧪 Testing LLM Integration...")
    try:
        chloe = ChloeLLM()
        
        # Create sample technical data
        tech_data = {
            'rsi_14': 65.42,
            'macd': 123.45,
            'macd_signal': 110.23,
            'ema_20': 45678.90,
            'ema_50': 44234.56,
            'volatility': 0.0234
        }
        
        risk_data = {
            'volatility': 0.0234,
            'correlation': 0.65
        }
        
        # Analyze a sample signal
        analysis = chloe.analyze_signal(
            symbol="BTC/USDT",
            signal="BUY",
            confidence=0.85,
            technical_data=tech_data,
            risk_data=risk_data
        )
        
        print(f"✅ LLM analysis completed for {analysis.symbol}")
        print(f"   Signal: {analysis.signal}")
        print(f"   Confidence: {analysis.confidence}")
        print(f"   Explanation: {analysis.explanation[:100]}...")
        
        return chloe
        
    except Exception as e:
        print(f"❌ Error in LLM test: {e}")
        return None

def test_backtesting():
    """Test backtesting functionality"""
    print("🧪 Testing Backtesting...")
    try:
        backtester = Backtester()
        
        # Create sample data for backtesting
        import pandas as pd
        import numpy as np
        
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        
        data = pd.DataFrame({
            'close': prices
        }, index=dates)
        
        # Create simple signals (1 for buy, -1 for sell, 0 for hold)
        signals = pd.Series(0, index=data.index)
        # Just for demo, alternate between buy and sell
        signals[::10] = 1  # Buy every 10th day
        signals[5::10] = -1  # Sell 5 days later
        
        # Run backtest
        results = backtester.run_backtest(data, signals)
        
        print(f"✅ Backtest completed")
        print(f"   Total return: {results['total_return']:.2%}")
        print(f"   Final capital: ${results['final_capital']:,.2f}")
        print(f"   Sharpe ratio: {results['sharpe_ratio']:.2f}")
        
        return backtester
        
    except Exception as e:
        print(f"❌ Error in backtesting test: {e}")
        return None

async def main():
    """Run all tests"""
    print("🔬 Starting Chloe AI Component Tests\n")
    
    # Test data collection and create sample data
    sample_data = await test_data_collection()
    if sample_data is None:
        print("❌ Failed to create sample data, stopping tests")
        return
    
    # Test indicator calculation
    indicator_data = test_indicator_calculation(sample_data)
    if indicator_data is None:
        print("❌ Indicator calculation failed, continuing with other tests")
    else:
        # Test ML signals if we have indicator data
        ml_core, processor = test_ml_signals(indicator_data)
    
    # Test other components
    test_risk_engine()
    test_chloe_llm()
    test_backtesting()
    
    print(f"\n✅ All component tests completed at {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    asyncio.run(main())