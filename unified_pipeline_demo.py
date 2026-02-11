"""
Demo script for the new unified Chloe AI architecture
Tests the canonical data flow: market_data → feature_store → forecast → allocation
"""
import asyncio
import logging
from datetime import datetime

from data_pipeline import get_data_pipeline
from forecast_strategies import initialize_forecast_strategy_manager, StrategySignal
from portfolio.portfolio import Portfolio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def demo_unified_pipeline():
    """Demonstrate the new unified pipeline architecture"""
    logger.info("🚀 Starting Chloe AI Unified Pipeline Demo")
    logger.info("=" * 60)
    
    # Symbols to test
    test_symbols = ['BTC/USDT', 'ETH/USDT']
    
    try:
        # Initialize the unified pipeline
        logger.info("🔧 Initializing unified data pipeline...")
        pipeline = get_data_pipeline()
        
        initialization_success = await pipeline.initialize(
            symbols=test_symbols,
            api_key=None,  # No API key needed for demo
            secret=None
        )
        
        if not initialization_success:
            logger.error("❌ Pipeline initialization failed")
            return
            
        logger.info("✅ Pipeline initialized successfully")
        logger.info(f"Pipeline status: {pipeline.get_pipeline_status()}")
        
        # Initialize forecast-based strategies
        logger.info("\n🎯 Initializing forecast-based strategies...")
        strategy_manager = initialize_forecast_strategy_manager()
        
        # Create demo portfolio
        portfolio = Portfolio(initial_capital=10000.0)
        logger.info(f"Portfolio created with ${portfolio.current_capital:,.2f}")
        
        # Test the complete pipeline for each symbol
        for symbol in test_symbols:
            logger.info(f"\n{'='*50}")
            logger.info(f"📋 Processing symbol: {symbol}")
            logger.info(f"{'='*50}")
            
            # Step 1: Show raw data fetching and feature calculation
            logger.info("📥 Step 1: Fetching and processing market data...")
            processed_data = await pipeline.fetch_and_process_data(symbol, lookback_days=180)
            if processed_data is not None:
                logger.info(f"   ✅ Processed {len(processed_data)} rows with {len(processed_data.columns)} features")
                logger.info(f"   Latest price: ${processed_data['close'].iloc[-1]:.2f}")
                # Show some key features
                key_features = ['rsi_14', 'macd_line', 'volatility_20', 'bb_position']
                available_features = [f for f in key_features if f in processed_data.columns]
                if available_features:
                    logger.info("   Sample features:")
                    for feature in available_features[:3]:
                        val = processed_data[feature].iloc[-1]
                        logger.info(f"     {feature}: {val:.4f}")
            else:
                logger.warning(f"   ⚠️  No data available for {symbol}")
                continue
            
            # Step 2: Show forecast generation
            logger.info("\n🔮 Step 2: Generating forecast...")
            forecast = await pipeline.generate_forecast(symbol, horizon=5)
            if forecast:
                logger.info(f"   ✅ Forecast generated:")
                logger.info(f"     Expected Return: {forecast['expected_return']:.4f} ({forecast['expected_return']*100:.2f}%)")
                logger.info(f"     Volatility: {forecast['volatility']:.4f} ({forecast['volatility']*100:.2f}%)")
                logger.info(f"     Confidence: {forecast['confidence']:.3f}")
                logger.info(f"     Percentiles: P10={forecast['percentiles']['p10']:.4f}, P50={forecast['percentiles']['p50']:.4f}, P90={forecast['percentiles']['p90']:.4f}")
            else:
                logger.warning(f"   ⚠️  No forecast available for {symbol}")
                continue
            
            # Step 3: Show strategy signal generation
            logger.info("\n🎯 Step 3: Generating strategy signals...")
            signals = await strategy_manager.generate_signals(symbol, portfolio)
            logger.info(f"   ✅ Generated {len(signals)} signals from different strategies")
            
            for i, signal in enumerate(signals):
                strategy = list(strategy_manager.strategies.values())[i]
                logger.info(f"     Strategy {i+1} ({strategy.name}):")
                logger.info(f"       Signal: {signal.signal}")
                logger.info(f"       Confidence: {signal.confidence:.3f}")
                logger.info(f"       Strength: {signal.strength:.3f}")
                if signal.position_size:
                    logger.info(f"       Position Size: {signal.position_size:.6f}")
                if signal.stop_loss:
                    logger.info(f"       Stop Loss: ${signal.stop_loss:.2f}")
                if signal.take_profit:
                    logger.info(f"       Take Profit: ${signal.take_profit:.2f}")
            
            # Step 4: Show combined signal
            logger.info("\n⚖️  Step 4: Combining signals...")
            combined_signal = strategy_manager.combine_signals(symbol, signals)
            if combined_signal:
                logger.info(f"   ✅ Combined signal:")
                logger.info(f"     Final Signal: {combined_signal.signal}")
                logger.info(f"     Combined Confidence: {combined_signal.confidence:.3f}")
                logger.info(f"     Combined Strength: {combined_signal.strength:.3f}")
                if combined_signal.position_size:
                    position_value = combined_signal.position_size * processed_data['close'].iloc[-1]
                    logger.info(f"     Position Size: {combined_signal.position_size:.6f} (${position_value:.2f})")
                if combined_signal.stop_loss:
                    logger.info(f"     Stop Loss: ${combined_signal.stop_loss:.2f}")
                if combined_signal.take_profit:
                    logger.info(f"     Take Profit: ${combined_signal.take_profit:.2f}")
            else:
                logger.info("   ℹ️  No combined signal (likely HOLD)")
            
            # Step 5: Show complete trading signal from pipeline
            logger.info("\n💰 Step 5: Complete pipeline trading signal...")
            complete_signal = await pipeline.get_trading_signal(symbol, portfolio)
            if complete_signal:
                logger.info(f"   ✅ Pipeline trading signal:")
                logger.info(f"     Signal: {complete_signal['signal']}")
                logger.info(f"     Confidence: {complete_signal['confidence']:.3f}")
                logger.info(f"     Expected Return: {complete_signal['expected_return']:.4f} ({complete_signal['expected_return']*100:.2f}%)")
                logger.info(f"     Volatility: {complete_signal['volatility']:.4f}")
                if complete_signal['position_size']:
                    position_value = complete_signal['position_size'] * complete_signal['entry_price']
                    logger.info(f"     Position: {complete_signal['position_size']:.6f} units (${position_value:.2f})")
                logger.info(f"     Entry Price: ${complete_signal['entry_price']:.2f}")
                logger.info(f"     ATR: ${complete_signal['atr']:.2f}")
                if complete_signal['stop_loss']:
                    risk_reward = abs(complete_signal['take_profit'] - complete_signal['entry_price']) / abs(complete_signal['entry_price'] - complete_signal['stop_loss']) if complete_signal['stop_loss'] != complete_signal['entry_price'] else 0
                    logger.info(f"     Stop Loss: ${complete_signal['stop_loss']:.2f}")
                    logger.info(f"     Take Profit: ${complete_signal['take_profit']:.2f}")
                    logger.info(f"     Risk/Reward: {risk_reward:.2f}:1")
            else:
                logger.warning(f"   ⚠️  No complete trading signal for {symbol}")
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("🏁 DEMO COMPLETED SUCCESSFULLY")
        logger.info(f"{'='*60}")
        logger.info("✅ Key achievements:")
        logger.info("   • Unified feature store consolidates all feature engineering")
        logger.info("   • Centralized data pipeline enforces proper data flow")
        logger.info("   • Forecast service acts as mandatory signal source")
        logger.info("   • Strategies consume only forecast events")
        logger.info("   • No direct market data access from strategies")
        logger.info("   • Clear separation of concerns between components")
        
        logger.info(f"\n📊 Pipeline Status: {pipeline.get_pipeline_status()}")
        
    except Exception as e:
        logger.error(f"❌ Demo failed with error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(demo_unified_pipeline())