#!/usr/bin/env python3
"""
Demo Script for Chloe AI
Shows the main capabilities of the system
"""

import asyncio
import sys
import os
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

async def run_demo():
    """Run a comprehensive demonstration of Chloe AI capabilities"""
    
    print("🚀 Chloe AI Demo - Market Analysis Agent")
    print("=" * 50)
    print(f"Demo started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Import required modules
    from data.data_agent import DataAgent
    from indicators.indicator_calculator import IndicatorCalculator
    from models.ml_core import MLSignalsCore, SignalProcessor
    from risk.risk_engine import RiskEngine
    from llm.chloe_llm import ChloeLLM
    from backtest.backtester import Backtester
    
    # Initialize components
    print("🤖 Initializing Chloe AI Components...")
    data_agent = DataAgent()
    indicator_calc = IndicatorCalculator()
    ml_core = MLSignalsCore()
    processor = SignalProcessor()
    risk_engine = RiskEngine()
    chloe = ChloeLLM()
    backtester = Backtester()
    
    print("✅ Components initialized successfully")
    print()
    
    # Create sample data for demonstration
    print("📊 Creating Sample Market Data...")
    import pandas as pd
    import numpy as np
    
    # Create 100 days of sample data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # Generate realistic price series with trend
    trend = np.linspace(0, 20, 100)  # Upward trend
    noise = np.cumsum(np.random.randn(100) * 2)  # Random noise
    base_price = 45000
    
    prices = base_price + trend + noise
    
    # Create DataFrame with realistic market data
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.randn(100) * 0.01),
        'high': prices * (1 + abs(np.random.randn(100)) * 0.02),
        'low': prices * (1 - abs(np.random.randn(100)) * 0.02),
        'close': prices,
        'volume': np.random.randint(1000, 10000, 100)
    })
    sample_data.set_index('timestamp', inplace=True)
    
    print(f"✅ Created sample data with {len(sample_data)} rows")
    print(f"   Price range: ${sample_data['close'].min():.2f} - ${sample_data['close'].max():.2f}")
    print(f"   Current price: ${sample_data['close'].iloc[-1]:.2f}")
    print()
    
    # Demonstrate technical indicator calculation
    print("📈 Calculating Technical Indicators...")
    indicator_data = indicator_calc.calculate_all_indicators(sample_data)
    
    print(f"✅ Indicators calculated: {len(indicator_data.columns) - len(sample_data.columns)} indicators")
    print("   Sample indicators:")
    print(f"   - RSI (14): {indicator_data['rsi_14'].iloc[-1]:.2f}")
    print(f"   - MACD: {indicator_data['macd'].iloc[-1]:.2f}")
    print(f"   - EMA (20): {indicator_data['ema_20'].iloc[-1]:.2f}")
    print(f"   - EMA (50): {indicator_data['ema_50'].iloc[-1]:.2f}")
    print(f"   - Volatility: {indicator_data['volatility'].iloc[-1]:.4f}")
    print()
    
    # Demonstrate risk calculation
    print("🛡️ Risk Management Example...")
    current_price = indicator_data['close'].iloc[-1]
    atr = indicator_data['volatility'].iloc[-1] * current_price  # Simple approximation
    
    print(f"Current Price: ${current_price:.2f}")
    print(f"Average True Range: ${atr:.2f}")
    
    # Calculate risk parameters
    stop_loss, take_profit = risk_engine.calculate_stop_loss_take_profit(
        current_price, 'BUY', atr
    )
    position_size = risk_engine.calculate_position_size(
        current_price, stop_loss, 10000, 0.02
    )
    
    print(f"Risk-adjusted BUY trade:")
    print(f"  - Stop Loss: ${stop_loss:.2f} (2ATR from entry)")
    print(f"  - Take Profit: ${take_profit:.2f} (Risk:Reward 1:2)")
    print(f"  - Position Size: {position_size:.6f} units")
    print(f"  - Investment Amount: ${position_size * current_price:.2f}")
    print()
    
    # Demonstrate backtesting with sample signals
    print("🧪 Backtesting Performance...")
    
    # Create simple buy/sell signals based on EMA crossover
    short_ema = indicator_data['ema_20']
    long_ema = indicator_data['ema_50']
    
    # Generate simple signals (1 = BUY, 0 = HOLD, -1 = SELL)
    signals = pd.Series(0, index=indicator_data.index)
    signals[short_ema > long_ema] = 1  # Buy when short EMA above long EMA
    signals[short_ema < long_ema] = -1  # Sell when short EMA below long EMA
    
    # Run backtest
    bt_results = backtester.run_backtest(sample_data, signals)
    
    print(f"📊 EMA Crossover Strategy Backtest Results:")
    print(f"  - Total Return: {bt_results['total_return']:.2%}")
    print(f"  - Annualized Return: {bt_results['annualized_return']:.2%}")
    print(f"  - Max Drawdown: {bt_results['max_drawdown']:.2%}")
    print(f"  - Sharpe Ratio: {bt_results['sharpe_ratio']:.2f}")
    print(f"  - Final Portfolio: ${bt_results['final_capital']:.2f}")
    print(f"  - Number of Trades: {bt_results['num_trades']}")
    print(f"  - Benchmark Total: {bt_results['benchmark_total_return']:.2%}")
    print()
    
    # Demonstrate LLM explanation
    print("💬 LLM Signal Explanation...")
    
    # Create sample technical data for LLM
    tech_data = {
        'rsi_14': indicator_data['rsi_14'].iloc[-1],
        'macd': indicator_data['macd'].iloc[-1],
        'macd_signal': indicator_data['macd_signal'].iloc[-1],
        'ema_20': indicator_data['ema_20'].iloc[-1],
        'ema_50': indicator_data['ema_50'].iloc[-1],
        'volatility': indicator_data['volatility'].iloc[-1]
    }
    
    risk_data = {
        'volatility': indicator_data['volatility'].iloc[-1]
    }
    
    # Generate analysis for a BUY signal
    analysis = chloe.analyze_signal(
        symbol="BTC/USDT",
        signal="BUY",
        confidence=0.75,
        technical_data=tech_data,
        risk_data=risk_data
    )
    
    print(f"Signal Analysis for BTC/USDT:")
    print(f"  - Signal: {analysis.signal}")
    print(f"  - Confidence: {analysis.confidence:.2f}")
    print(f"  - Explanation: {analysis.explanation}")
    print(f"  - Suggested Action: {analysis.suggested_action}")
    print()
    
    # Demonstrate API-like functionality
    print("🌐 API-Style Analysis Summary...")
    
    demo_result = {
        "symbol": "BTC/USDT",
        "current_price": current_price,
        "signal": "BUY",
        "confidence": 0.75,
        "risk_level": "MEDIUM",
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "position_size": position_size,
        "explanation": analysis.explanation[:100] + "...",
        "timestamp": datetime.now().isoformat()
    }
    
    print("API Response Sample:")
    for key, value in demo_result.items():
        print(f"  {key}: {value}")
    print()
    
    # Final summary
    print("🎉 Demo Completed Successfully!")
    print("=" * 50)
    print("Chloe AI demonstrated capabilities:")
    print("✅ Data Collection and Analysis")
    print("✅ Technical Indicator Calculation")
    print("✅ Risk Management and Position Sizing")
    print("✅ Backtesting and Performance Analysis")
    print("✅ LLM-Powered Signal Explanation")
    print("✅ API-Ready Architecture")
    print()
    print("Ready for real market data integration and advanced ML training!")
    print(f"Demo completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    asyncio.run(run_demo())