"""
Enhanced Risk Engine Demo for Chloe AI 0.4
Demonstrates professional risk management with Kelly criterion, CVaR optimization, and regime-aware risk controls
"""
import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from enhanced_risk_engine import EnhancedRiskEngine, RiskParameters
from regime_detection import RegimeDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def demo_enhanced_risk_engine():
    """Demonstrate enhanced risk engine capabilities"""
    logger.info("🛡️ Chloe AI 0.4 - Enhanced Risk Engine Demo")
    logger.info("=" * 60)
    
    try:
        # Initialize enhanced risk engine
        logger.info("🔧 Initializing enhanced risk engine...")
        risk_params = RiskParameters(
            max_position_size=0.10,      # 10% max per position
            kelly_fraction=0.25,         # Quarter Kelly
            max_drawdown_limit=0.20,     # 20% max drawdown
            var_confidence=0.95,
            cvar_confidence=0.95
        )
        
        risk_engine = EnhancedRiskEngine(initial_capital=10000.0, risk_params=risk_params)
        risk_engine.initialize_portfolio_tracking()
        
        logger.info(f"✅ Risk engine initialized with ${risk_engine.current_capital:,.2f} capital")
        logger.info(f"   Kelly fraction: {risk_engine.risk_params.kelly_fraction}")
        logger.info(f"   Max position size: {risk_engine.risk_params.max_position_size*100:.1f}%")
        logger.info(f"   Max drawdown limit: {risk_engine.risk_params.max_drawdown_limit*100:.1f}%")
        
        # Test 1: Kelly Position Sizing
        logger.info(f"\n{'='*50}")
        logger.info("🎲 Test 1: Kelly Criterion Position Sizing")
        logger.info(f"{'='*50}")
        
        test_scenarios = [
            {'name': 'High Conviction Trade', 'win_rate': 0.65, 'win_loss_ratio': 2.0, 'regime': 'TRENDING'},
            {'name': 'Medium Conviction Trade', 'win_rate': 0.55, 'win_loss_ratio': 1.5, 'regime': 'STABLE'},
            {'name': 'Low Conviction Trade', 'win_rate': 0.45, 'win_loss_ratio': 1.2, 'regime': 'VOLATILE'},
            {'name': 'Mean Reversion Trade', 'win_rate': 0.70, 'win_loss_ratio': 1.8, 'regime': 'MEAN_REVERTING'}
        ]
        
        for scenario in test_scenarios:
            position_size = risk_engine.calculate_kelly_position_size(
                win_rate=scenario['win_rate'],
                win_loss_ratio=scenario['win_loss_ratio'],
                account_size=risk_engine.current_capital,
                regime=scenario['regime']
            )
            
            position_value = position_size * risk_engine.current_capital
            
            logger.info(f"   {scenario['name']}:")
            logger.info(f"     Win Rate: {scenario['win_rate']*100:.1f}%")
            logger.info(f"     W/L Ratio: {scenario['win_loss_ratio']:.1f}:1")
            logger.info(f"     Regime: {scenario['regime']}")
            logger.info(f"     Recommended Size: {position_size*100:.2f}% (${position_value:,.2f})")
        
        # Test 2: Position Risk Assessment
        logger.info(f"\n{'='*50}")
        logger.info("⚖️ Test 2: Position Risk Assessment")
        logger.info(f"{'='*50}")
        
        position_tests = [
            {
                'symbol': 'BTC/USDT',
                'entry_price': 50000,
                'position_size': 0.5,  # 0.5 BTC
                'stop_loss': 48000,
                'take_profit': 55000,
                'volatility': 0.03,
                'regime': 'TRENDING'
            },
            {
                'symbol': 'ETH/USDT',
                'entry_price': 3000,
                'position_size': 2.0,  # 2 ETH
                'stop_loss': 2800,
                'take_profit': 3300,
                'volatility': 0.04,
                'regime': 'VOLATILE'
            }
        ]
        
        for test in position_tests:
            risk_assessment = risk_engine.assess_position_risk(
                symbol=test['symbol'],
                entry_price=test['entry_price'],
                position_size=test['position_size'],
                stop_loss=test['stop_loss'],
                take_profit=test['take_profit'],
                volatility=test['volatility'],
                regime=test['regime']
            )
            
            metrics = risk_assessment['risk_metrics']
            approved = risk_assessment['approved']
            
            logger.info(f"   {test['symbol']} Position Assessment:")
            logger.info(f"     Entry: ${test['entry_price']:,.2f}")
            logger.info(f"     Size: {test['position_size']} units (${metrics['position_value']:,.2f})")
            logger.info(f"     Risk: ${metrics['max_loss']:,.2f} ({metrics['position_percentage']*100:.2f}% of capital)")
            logger.info(f"     Reward: ${metrics['max_gain']:,.2f}")
            logger.info(f"     R/R Ratio: {metrics['risk_reward_ratio']:.2f}:1")
            logger.info(f"     Volatility: {test['volatility']*100:.1f}%")
            logger.info(f"     Regime: {test['regime']}")
            logger.info(f"     Kelly Recommendation: {metrics['kelly_recommended_size']*100:.2f}%")
            logger.info(f"     STATUS: {'✅ APPROVED' if approved else '❌ REJECTED'}")
            
            if not approved:
                recommendations = risk_assessment.get('recommendations', [])
                if recommendations:
                    logger.info(f"     Recommendations: {', '.join(recommendations)}")
        
        # Test 3: CVaR Optimization
        logger.info(f"\n{'='*50}")
        logger.info("📊 Test 3: CVaR Portfolio Optimization")
        logger.info(f"{'='*50}")
        
        # Generate synthetic returns data
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT']
        dates = pd.date_range(end=datetime.now(), periods=252, freq='D')  # 1 year of daily data
        
        # Create correlated return series
        np.random.seed(42)  # For reproducible results
        base_returns = np.random.normal(0.0005, 0.02, 252)  # Base market returns
        
        returns_data = pd.DataFrame(index=dates)
        correlations = [1.0, 0.8, 0.6, 0.4]  # Different correlations to base
        
        for i, symbol in enumerate(symbols):
            # Create correlated returns
            symbol_returns = base_returns * correlations[i] + np.random.normal(0, 0.015, 252)
            returns_data[symbol] = symbol_returns
        
        logger.info(f"   Generated returns data for {len(symbols)} symbols")
        logger.info(f"   Data period: {len(returns_data)} days")
        logger.info("   Correlation structure:")
        for i, symbol in enumerate(symbols):
            logger.info(f"     {symbol}: {correlations[i]:.2f} with base market")
        
        # Optimize portfolio allocation
        optimal_allocation = risk_engine.calculate_cvar_optimal_allocation(
            returns_data=returns_data,
            symbols=symbols,
            confidence_level=0.95
        )
        
        logger.info("   CVaR-Optimal Portfolio Allocation:")
        total_allocated = 0
        for symbol, weight in optimal_allocation.items():
            allocation_value = weight * risk_engine.current_capital
            logger.info(f"     {symbol}: {weight*100:.2f}% (${allocation_value:,.2f})")
            total_allocated += allocation_value
        
        logger.info(f"   Total allocated: ${total_allocated:,.2f} ({total_allocated/risk_engine.current_capital*100:.1f}% of capital)")
        
        # Test 4: Portfolio Risk Metrics
        logger.info(f"\n{'='*50}")
        logger.info("📈 Test 4: Portfolio Risk Monitoring")
        logger.info(f"{'='*50}")
        
        # Simulate portfolio updates
        portfolio_scenarios = [
            {'value': 10500, 'positions': {'BTC/USDT': 1.0, 'ETH/USDT': 2.0}},
            {'value': 10200, 'positions': {'BTC/USDT': 0.8, 'ETH/USDT': 1.5, 'SOL/USDT': 10.0}},
            {'value': 9800, 'positions': {'BTC/USDT': 0.5, 'ETH/USDT': 1.0}},  # Drawdown scenario
            {'value': 11000, 'positions': {'BTC/USDT': 1.2, 'ETH/USDT': 2.5, 'SOL/USDT': 15.0}}
        ]
        
        for i, scenario in enumerate(portfolio_scenarios):
            logger.info(f"   Scenario {i+1}: Portfolio value ${scenario['value']:,.2f}")
            
            # Update portfolio state
            update_success = risk_engine.update_portfolio_state(
                portfolio_value=scenario['value'],
                positions=scenario['positions']
            )
            
            # Get risk report
            risk_report = risk_engine.get_risk_report()
            metrics = risk_report['portfolio_metrics']
            
            logger.info(f"     Current Value: ${metrics['current_value']:,.2f}")
            logger.info(f"     Peak Value: ${metrics['peak_value']:,.2f}")
            logger.info(f"     Current Drawdown: {metrics['current_drawdown']*100:.2f}%")
            logger.info(f"     Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
            logger.info(f"     Portfolio Volatility: {metrics['volatility']*100:.2f}%")
            logger.info(f"     Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            logger.info(f"     Positions: {metrics['number_of_positions']}")
            logger.info(f"     Status: {risk_report['status']}")
            
            if not update_success:
                logger.critical("     ⚠️ DRAWDOWN LIMIT EXCEEDED - Risk mitigation triggered!")
        
        # Test 5: Regime-Aware Risk Adjustments
        logger.info(f"\n{'='*50}")
        logger.info("🎭 Test 5: Regime-Aware Risk Adjustments")
        logger.info(f"{'='*50}")
        
        regime_multipliers = risk_engine.risk_params.regime_risk_multiplier
        logger.info("   Risk multipliers by market regime:")
        for regime, multiplier in regime_multipliers.items():
            logger.info(f"     {regime}: {multiplier:.2f}x base risk")
        
        # Demonstrate how same trade parameters yield different risk assessments
        base_trade = {
            'symbol': 'TEST/USDT',
            'entry_price': 1000,
            'position_size': 1.0,
            'stop_loss': 950,
            'take_profit': 1100,
            'volatility': 0.025
        }
        
        logger.info(f"\n   Same trade ({base_trade['symbol']}) assessed under different regimes:")
        for regime in ['STABLE', 'TRENDING', 'MEAN_REVERTING', 'VOLATILE']:
            risk_assessment = risk_engine.assess_position_risk(
                symbol=base_trade['symbol'],
                entry_price=base_trade['entry_price'],
                position_size=base_trade['position_size'],
                stop_loss=base_trade['stop_loss'],
                take_profit=base_trade['take_profit'],
                volatility=base_trade['volatility'],
                regime=regime
            )
            
            metrics = risk_assessment['risk_metrics']
            approved = risk_assessment['approved']
            multiplier = regime_multipliers[regime]
            
            logger.info(f"     {regime}: {'✅' if approved else '❌'} "
                       f"(Multiplier: {multiplier:.2f}x, "
                       f"Adjusted volatility: {metrics['adjusted_volatility']*100:.2f}%, "
                       f"Risk: ${metrics['max_loss']:,.2f})")
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("🎯 ENHANCED RISK ENGINE DEMO COMPLETED")
        logger.info(f"{'='*60}")
        logger.info("✅ Key achievements:")
        logger.info("   • Implemented Kelly criterion position sizing")
        logger.info("   • Added CVaR portfolio optimization")
        logger.info("   • Created comprehensive risk assessment framework")
        logger.info("   • Built regime-aware risk adjustments")
        logger.info("   • Added portfolio monitoring and drawdown protection")
        logger.info("   • Integrated with existing Chloe architecture")
        
        logger.info(f"\n🚀 Chloe 0.4 Progress:")
        logger.info("   ✅ Phase 1: Market Intelligence Layer (70% complete)")
        logger.info("   ✅ Phase 2: Risk Engine Core Enhancement (now complete)")
        logger.info("   ⬜ Phase 3: Edge Classification Model")
        logger.info("   ⬜ Phase 4: Portfolio Construction Logic")
        logger.info("   ⬜ Phase 5: Simulation Lab")
        
        # Final risk report
        final_report = risk_engine.get_risk_report()
        logger.info(f"\n📊 Final Risk Status: {final_report['status']}")
        logger.info(f"   Current Portfolio Value: ${final_report['portfolio_metrics']['current_value']:,.2f}")
        logger.info(f"   Maximum Drawdown: {final_report['portfolio_metrics']['max_drawdown']*100:.2f}%")
        logger.info(f"   Portfolio Volatility: {final_report['portfolio_metrics']['volatility']*100:.2f}%")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(demo_enhanced_risk_engine())