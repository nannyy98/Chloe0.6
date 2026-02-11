#!/usr/bin/env python3
"""
Edge Probability Models Demo for Chloe 0.6
Professional edge detection using regime-aware probability modeling
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List
import numpy as np
import pandas as pd
from edge_probability_model import get_edge_model, EdgeTrainingData

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_historical_data(symbols: List[str], periods: int = 200) -> Dict[str, pd.DataFrame]:
    """Generate synthetic historical data for multiple symbols"""
    data = {}
    
    # Different market behaviors for different symbols
    symbol_behaviors = {
        'BTC/USDT': {'volatility': 0.04, 'trend': 0.002, 'regime_changes': True},
        'ETH/USDT': {'volatility': 0.03, 'trend': 0.001, 'regime_changes': True},
        'SOL/USDT': {'volatility': 0.05, 'trend': -0.0005, 'regime_changes': False},
        'ADA/USDT': {'volatility': 0.02, 'trend': 0.0005, 'regime_changes': False}
    }
    
    dates = pd.date_range(end=datetime.now(), periods=periods, freq='D')
    
    for symbol in symbols:
        behavior = symbol_behaviors.get(symbol, {'volatility': 0.03, 'trend': 0.001})
        
        # Generate base price series
        prices = [100]  # Start at 100
        
        # Simulate regime changes for some symbols
        regime_schedule = []
        if behavior['regime_changes']:
            # Alternate between regimes
            regimes = ['STABLE', 'TRENDING', 'VOLATILE', 'CRISIS']
            regime_periods = periods // len(regimes)
            for i, regime in enumerate(regimes):
                regime_schedule.extend([regime] * regime_periods)
            regime_schedule.extend([regimes[-1]] * (periods - len(regime_schedule)))
        else:
            # Stay in stable regime
            regime_schedule = ['STABLE'] * periods
        
        for i in range(1, periods):
            current_regime = regime_schedule[i]
            
            # Regime-specific parameters
            regime_params = {
                'STABLE': {'vol_mult': 1.0, 'trend_mult': 1.0, 'noise': 0.8},
                'TRENDING': {'vol_mult': 1.2, 'trend_mult': 2.0, 'noise': 0.5},
                'VOLATILE': {'vol_mult': 2.0, 'trend_mult': 0.5, 'noise': 1.5},
                'CRISIS': {'vol_mult': 2.5, 'trend_mult': -1.0, 'noise': 2.0}
            }
            
            params = regime_params[current_regime]
            
            # Calculate return
            trend_component = behavior['trend'] * params['trend_mult']
            volatility_component = np.random.normal(0, behavior['volatility'] * params['vol_mult'])
            noise_component = np.random.normal(0, behavior['volatility'] * params['noise'] * 0.1)
            
            total_return = trend_component + volatility_component + noise_component
            new_price = prices[-1] * (1 + total_return)
            prices.append(max(new_price, 1))  # Ensure positive prices
        
        # Create DataFrame
        df = pd.DataFrame({
            'close': prices,
            'Close': prices,  # Duplicate for compatibility
            'high': [p * (1 + np.random.uniform(0, 0.02)) for p in prices],
            'low': [p * (1 - np.random.uniform(0, 0.02)) for p in prices],
            'volume': [np.random.randint(1000000, 10000000) for _ in prices]
        }, index=dates)
        
        data[symbol] = df
    
    return data, regime_schedule

async def demonstrate_edge_modeling():
    """Demonstrate edge probability modeling capabilities"""
    logger.info("🎲 EDGE PROBABILITY MODELING DEMO")
    logger.info("=" * 50)
    
    try:
        # Initialize edge model
        logger.info("🔧 Initializing Edge Probability Model...")
        edge_model = get_edge_model()
        logger.info("✅ Edge Model initialized")
        
        # Generate synthetic historical data
        logger.info("📊 Generating historical training data...")
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT']
        historical_data, regime_labels = generate_historical_data(symbols, periods=150)
        
        logger.info(f"   Generated data for {len(historical_data)} symbols")
        logger.info(f"   Total periods: {len(regime_labels)}")
        logger.info(f"   Regime distribution: {pd.Series(regime_labels).value_counts().to_dict()}")
        
        # Prepare training data
        logger.info("🧮 Preparing edge training data...")
        training_data = edge_model.prepare_training_data(historical_data, regime_labels)
        
        logger.info(f"   Training samples: {len(training_data.labels)}")
        logger.info(f"   Positive labels: {sum(training_data.labels)} ({sum(training_data.labels)/len(training_data.labels)*100:.1f}%)")
        logger.info(f"   Feature dimensions: {training_data.features.shape[1]}")
        
        # Train models
        logger.info("🧠 Training regime-aware edge models...")
        edge_model.train_edge_models(training_data)
        
        # Display training results
        performance = edge_model.get_model_performance_report()
        logger.info("📈 TRAINING RESULTS:")
        for regime, metrics in performance['individual_model_performance'].items():
            logger.info(f"   {regime}:")
            logger.info(f"      Train Accuracy: {metrics['train_accuracy']:.3f}")
            logger.info(f"      Val Accuracy: {metrics['validation_accuracy']:.3f}")
            logger.info(f"      Samples: {metrics['samples']}")
        
        # Test edge evaluation
        logger.info(f"\n🎯 TESTING EDGE EVALUATION:")
        
        test_scenarios = [
            ('BTC/USDT', 'STABLE'),
            ('ETH/USDT', 'TRENDING'), 
            ('SOL/USDT', 'VOLATILE'),
            ('ADA/USDT', 'CRISIS')
        ]
        
        edge_opportunities = []
        
        for symbol, regime in test_scenarios:
            logger.info(f"\n   Evaluating {symbol} in {regime} regime:")
            
            # Get recent data for this symbol
            symbol_data = historical_data[symbol].tail(30)
            
            # Evaluate edge opportunity
            edge_opportunity = edge_model.evaluate_edge_opportunity(
                current_features=symbol_data,  # Model will extract features internally
                regime=regime,
                symbol=symbol
            )
            
            edge_opportunities.append(edge_opportunity)
            
            logger.info(f"      Edge Probability: {edge_opportunity.edge_probability:.3f}")
            logger.info(f"      Expected Return: {edge_opportunity.expected_return:.4f}")
            logger.info(f"      Confidence Interval: [{edge_opportunity.confidence_interval[0]:.3f}, {edge_opportunity.confidence_interval[1]:.3f}]")
            logger.info(f"      Time Horizon: {edge_opportunity.time_horizon}")
            logger.info(f"      Risk Metrics: {edge_opportunity.risk_metrics}")
            
            # Show top contributing features
            top_features = sorted(edge_opportunity.feature_importance.items(), 
                                key=lambda x: x[1], reverse=True)[:3]
            logger.info(f"      Top Features: {top_features}")
        
        # Analyze edge opportunities
        logger.info(f"\n📊 EDGE OPPORTUNITY ANALYSIS:")
        
        probabilities = [opp.edge_probability for opp in edge_opportunities]
        expected_returns = [opp.expected_return for opp in edge_opportunities]
        time_horizons = [opp.time_horizon for opp in edge_opportunities]
        
        logger.info(f"   Average Edge Probability: {np.mean(probabilities):.3f}")
        logger.info(f"   Best Opportunity: {max(probabilities):.3f}")
        logger.info(f"   Worst Opportunity: {min(probabilities):.3f}")
        logger.info(f"   Average Expected Return: {np.mean(expected_returns):.4f}")
        logger.info(f"   Time Horizons: {pd.Series(time_horizons).value_counts().to_dict()}")
        
        # Test regime-specific performance
        logger.info(f"\n🔍 REGIME-SPECIFIC PERFORMANCE:")
        
        regime_performance = {}
        for opportunity in edge_opportunities:
            regime = opportunity.regime_context
            if regime not in regime_performance:
                regime_performance[regime] = []
            regime_performance[regime].append(opportunity.edge_probability)
        
        for regime, probs in regime_performance.items():
            logger.info(f"   {regime}: avg={np.mean(probs):.3f}, count={len(probs)}")
        
        # Simulate real-world edge detection
        logger.info(f"\n🌍 REAL-WORLD SCENARIO SIMULATION:")
        
        # Create streaming data simulation
        logger.info("   Simulating live edge detection...")
        
        for i in range(0, len(regime_labels), 10):  # Check every 10 periods
            current_regime = regime_labels[i]
            current_date = historical_data['BTC/USDT'].index[i]
            
            # Evaluate BTC edge at this point
            btc_data = historical_data['BTC/USDT'].iloc[:i+1] if i > 0 else historical_data['BTC/USDT'].head(30)
            
            edge_opp = edge_model.evaluate_edge_opportunity(
                current_features=btc_data,
                regime=current_regime,
                symbol='BTC/USDT'
            )
            
            # Determine signal based on probability
            if edge_opp.edge_probability > 0.6:
                signal = "🟢 STRONG_BUY"
            elif edge_opp.edge_probability > 0.55:
                signal = "🔵 BUY"
            elif edge_opp.edge_probability < 0.4:
                signal = "🔴 SELL"
            elif edge_opp.edge_probability < 0.45:
                signal = "🟠 STRONG_SELL"
            else:
                signal = "⚪ HOLD"
            
            logger.info(f"      {current_date.strftime('%Y-%m-%d')}: {signal} "
                       f"(Prob: {edge_opp.edge_probability:.3f}, Regime: {current_regime})")
        
        # Validate model consistency
        logger.info(f"\n✅ MODEL CONSISTENCY CHECK:")
        
        # Test same inputs produce same outputs
        test_symbol = 'BTC/USDT'
        test_regime = 'STABLE'
        test_data = historical_data[test_symbol].tail(30)
        
        results = []
        for i in range(5):
            opp = edge_model.evaluate_edge_opportunity(test_data, test_regime, test_symbol)
            results.append(opp.edge_probability)
        
        consistency = np.std(results) < 0.01  # Should be very consistent
        logger.info(f"   Consistency Check: {'PASSED' if consistency else 'FAILED'}")
        logger.info(f"   Probability Variance: {np.std(results):.6f}")
        
        logger.info(f"\n🎯 EDGE MODELING DEMO COMPLETED SUCCESSFULLY")
        logger.info("🚀 Key Achievements:")
        logger.info("   • Implemented regime-aware edge probability modeling")
        logger.info("   • Created P(strategy profitable | regime, features) framework")
        logger.info("   • Built ensemble models for different market regimes")
        logger.info("   • Developed confidence interval estimation")
        logger.info("   • Integrated risk-adjusted metrics")
        
        logger.info(f"\n📊 PERFORMANCE SUMMARY:")
        logger.info(f"   Models Trained: {len(performance['trained_regimes'])}")
        logger.info(f"   Features Used: {len(performance['feature_names'])}")
        logger.info(f"   Average Edge Detection Accuracy: {np.mean([m['validation_accuracy'] for m in performance['individual_model_performance'].values()]):.3f}")
        logger.info(f"   Edge Opportunities Identified: {len(edge_opportunities)}")
        
        logger.info(f"\n🎯 NEXT STEPS:")
        logger.info("   1. Integrate with Risk Engine (Kelly Criterion + CVaR)")
        logger.info("   2. Connect to Portfolio Construction Engine")
        logger.info("   3. Add Walk-Forward Validation")
        logger.info("   4. Implement Real-time Edge Scoring")
        
    except Exception as e:
        logger.error(f"❌ Edge modeling demo failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def demonstrate_feature_engineering():
    """Demonstrate edge feature engineering capabilities"""
    logger.info(f"\n🧮 EDGE FEATURE ENGINEERING DEMO")
    logger.info("=" * 40)
    
    try:
        from edge_probability_model import RegimeAwareEdgeModel
        
        # Create model instance
        model = RegimeAwareEdgeModel()
        
        # Generate sample data
        dates = pd.date_range(end=datetime.now(), periods=50, freq='D')
        sample_data = pd.DataFrame({
            'close': 50000 + np.random.randn(50).cumsum() * 100,
            'Close': 50000 + np.random.randn(50).cumsum() * 100,
            'high': 50000 + np.random.randn(50).cumsum() * 100 + 200,
            'low': 50000 + np.random.randn(50).cumsum() * 100 - 200,
            'volume': np.random.randint(1000000, 10000000, 50)
        }, index=dates)
        
        # Extract features
        features = model._extract_edge_features(sample_data)
        
        logger.info("Extracted Edge Features:")
        logger.info(f"   Shape: {features.shape}")
        logger.info(f"   Columns: {list(features.columns)}")
        logger.info(f"   Sample values:")
        
        # Show sample of key features
        key_features = ['price_momentum_5', 'volatility_10', 'rsi', 'macd']
        for feature in key_features:
            if feature in features.columns:
                logger.info(f"      {feature}: {features[feature].iloc[-1]:.4f}")
        
        logger.info("✅ Feature engineering demonstrated successfully")
        
    except Exception as e:
        logger.error(f"❌ Feature engineering demo failed: {e}")

async def main():
    """Main demo function"""
    print("Chloe 0.6 - Edge Probability Models Demo")
    print("Professional regime-aware edge detection")
    print()
    
    # Run main edge modeling demo
    await demonstrate_edge_modeling()
    
    # Run feature engineering demo
    demonstrate_feature_engineering()
    
    print(f"\n🎉 EDGE PROBABILITY MODELS DEMO COMPLETED")
    print("Chloe 0.6 now has professional edge probability modeling capabilities!")

if __name__ == "__main__":
    asyncio.run(main())