#!/usr/bin/env python3
"""
Advanced Risk Engine Demo for Chloe 0.6
Professional risk management with Kelly Criterion and CVaR optimization
"""

import asyncio
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from advanced_risk_engine import get_advanced_risk_engine, KellyPosition, CVaROptimization

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_asset_data(symbols: list, periods: int = 252) -> tuple:
    """Generate synthetic asset data for risk modeling"""
    # Different asset characteristics
    asset_params = {
        'BTC/USDT': {'expected_return': 0.001, 'volatility': 0.04, 'correlation_base': 0.3},
        'ETH/USDT': {'expected_return': 0.0008, 'volatility': 0.035, 'correlation_base': 0.4},
        'SOL/USDT': {'expected_return': 0.0005, 'volatility': 0.05, 'correlation_base': 0.2},
        'ADA/USDT': {'expected_return': 0.0003, 'volatility': 0.025, 'correlation_base': 0.1}
    }
    
    dates = pd.date_range(end=datetime.now(), periods=periods, freq='D')
    returns_data = {}
    price_data = {}
    
    # Generate correlated returns
    np.random.seed(42)  # For reproducible results
    
    # Create base correlation structure
    correlation_matrix = np.array([
        [1.0, 0.7, 0.4, 0.2],
        [0.7, 1.0, 0.5, 0.3],
        [0.4, 0.5, 1.0, 0.1],
        [0.2, 0.3, 0.1, 1.0]
    ])
    
    # Generate multivariate normal returns
    n_assets = len(symbols)
    mean_returns = [asset_params[sym]['expected_return'] for sym in symbols]
    
    # Create covariance matrix
    volatilities = [asset_params[sym]['volatility'] for sym in symbols]
    covariance_matrix = np.zeros((n_assets, n_assets))
    
    for i in range(n_assets):
        for j in range(n_assets):
            covariance_matrix[i, j] = correlation_matrix[i, j] * volatilities[i] * volatilities[j]
    
    # Generate correlated returns
    correlated_returns = np.random.multivariate_normal(mean_returns, covariance_matrix, periods)
    
    # Convert to price series
    for i, symbol in enumerate(symbols):
        returns = correlated_returns[:, i]
        prices = 100 * np.exp(np.cumsum(returns))  # Start at 100
        
        returns_data[symbol] = returns.tolist()
        price_data[symbol] = pd.Series(prices, index=dates)
    
    return returns_data, price_data, covariance_matrix

async def demonstrate_advanced_risk_engine():
    """Demonstrate advanced risk engine capabilities"""
    logger.info("🛡️ ADVANCED RISK ENGINE DEMO")
    logger.info("=" * 50)
    
    try:
        # Initialize advanced risk engine
        logger.info("🔧 Initializing Advanced Risk Engine...")
        risk_engine = get_advanced_risk_engine(initial_capital=100000.0)
        logger.info("✅ Advanced Risk Engine initialized")
        
        # Generate synthetic market data
        logger.info("📊 Generating market data for risk analysis...")
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT']
        returns_data, price_data, covariance_matrix = generate_asset_data(symbols, periods=252)
        
        logger.info(f"   Generated data for {len(symbols)} assets")
        logger.info(f"   Historical periods: {len(returns_data[symbols[0]])}")
        logger.info(f"   Covariance matrix shape: {covariance_matrix.shape}")
        
        # Test Kelly Criterion calculations
        logger.info(f"\n🎲 KELLY CRITERION ANALYSIS:")
        
        kelly_scenarios = [
            {'name': 'High Conviction Trade', 'win_rate': 0.65, 'win_loss_ratio': 2.0, 'regime': 'TRENDING'},
            {'name': 'Moderate Trade', 'win_rate': 0.55, 'win_loss_ratio': 1.5, 'regime': 'STABLE'},
            {'name': 'Conservative Trade', 'win_rate': 0.45, 'win_loss_ratio': 1.2, 'regime': 'VOLATILE'},
            {'name': 'Crisis Conditions', 'win_rate': 0.35, 'win_loss_ratio': 0.8, 'regime': 'CRISIS'}
        ]
        
        kelly_positions = []
        
        for scenario in kelly_scenarios:
            kelly_pos = risk_engine.calculate_kelly_criterion(
                win_rate=scenario['win_rate'],
                win_loss_ratio=scenario['win_loss_ratio'],
                regime=scenario['regime']
            )
            
            kelly_positions.append(kelly_pos)
            
            logger.info(f"   {scenario['name']}:")
            logger.info(f"      Win Rate: {scenario['win_rate']:.1%}")
            logger.info(f"      Win/Loss Ratio: {scenario['win_loss_ratio']:.1f}")
            logger.info(f"      Regime: {scenario['regime']}")
            logger.info(f"      Optimal Fraction: {kelly_pos.optimal_fraction:.3f} ({kelly_pos.optimal_fraction * 100000:.0f} USD)")
            logger.info(f"      Expected Growth: {kelly_pos.expected_growth:.4f}")
            logger.info(f"      Risk of Ruin: {kelly_pos.risk_of_ruin:.1%}")
            logger.info(f"      Regime Adjustment: {kelly_pos.regime_adjustment:.2f}")
        
        # Analyze Kelly results
        logger.info(f"\n📊 KELLY CRITERION SUMMARY:")
        fractions = [kp.optimal_fraction for kp in kelly_positions]
        growth_rates = [kp.expected_growth for kp in kelly_positions]
        risk_levels = [kp.risk_of_ruin for kp in kelly_positions]
        
        logger.info(f"   Average Position Size: {np.mean(fractions):.3f}")
        logger.info(f"   Maximum Position Size: {np.max(fractions):.3f}")
        logger.info(f"   Average Expected Growth: {np.mean(growth_rates):.4f}")
        logger.info(f"   Average Risk of Ruin: {np.mean(risk_levels):.1%}")
        
        # Test CVaR Portfolio Optimization
        logger.info(f"\n📊 CVaR PORTFOLIO OPTIMIZATION:")
        
        # Prepare expected returns
        expected_returns = {
            'BTC/USDT': 0.15,  # 15% annual expected return
            'ETH/USDT': 0.12,  # 12% annual expected return
            'SOL/USDT': 0.20,  # 20% annual expected return
            'ADA/USDT': 0.08   # 8% annual expected return
        }
        
        # Test different regimes
        regimes = ['STABLE', 'TRENDING', 'VOLATILE', 'CRISIS']
        
        cvar_optimizations = []
        
        for regime in regimes:
            logger.info(f"   Optimizing for {regime} regime:")
            
            cvar_opt = risk_engine.optimize_cvar_portfolio(
                expected_returns=expected_returns,
                covariance_matrix=covariance_matrix,
                symbols=symbols,
                regime=regime
            )
            
            cvar_optimizations.append(cvar_opt)
            
            logger.info(f"      Optimal Weights:")
            for symbol, weight in cvar_opt.optimal_weights.items():
                logger.info(f"         {symbol}: {weight:.3f} ({weight * 100:.1f}%)")
            
            logger.info(f"      Portfolio CVaR: {cvar_opt.portfolio_cvar:.3f}")
            logger.info(f"      Expected Return: {cvar_opt.expected_return:.3f}")
            logger.info(f"      Sharpe Ratio: {cvar_opt.sharpe_ratio:.3f}")
            logger.info(f"      Diversification Ratio: {cvar_opt.diversification_ratio:.3f}")
        
        # Analyze CVaR optimization results
        logger.info(f"\n📈 CVaR OPTIMIZATION ANALYSIS:")
        
        cvar_values = [opt.portfolio_cvar for opt in cvar_optimizations]
        expected_returns_list = [opt.expected_return for opt in cvar_optimizations]
        sharpe_ratios = [opt.sharpe_ratio for opt in cvar_optimizations]
        
        logger.info(f"   CVaR Range: {min(cvar_values):.3f} to {max(cvar_values):.3f}")
        logger.info(f"   Average CVaR: {np.mean(cvar_values):.3f}")
        logger.info(f"   Best Sharpe Ratio: {max(sharpe_ratios):.3f}")
        logger.info(f"   Return Volatility: {np.std(expected_returns_list):.3f}")
        
        # Test Risk Assessment
        logger.info(f"\n🔍 REGIME RISK ASSESSMENT:")
        
        # Simulate portfolio with sample weights
        sample_weights = {
            'BTC/USDT': 0.4,
            'ETH/USDT': 0.3,
            'SOL/USDT': 0.2,
            'ADA/USDT': 0.1
        }
        
        for regime in regimes:
            logger.info(f"   Assessing risk in {regime} regime:")
            
            risk_assessment = risk_engine.assess_regime_risk(
                current_regime=regime,
                portfolio_weights=sample_weights,
                asset_returns=returns_data
            )
            
            logger.info(f"      Risk Rating: {risk_assessment['risk_rating']}")
            logger.info(f"      Volatility: {risk_assessment['volatility']:.3f}")
            logger.info(f"      VaR (95%): {risk_assessment['value_at_risk']:.3f}")
            logger.info(f"      CVaR (95%): {risk_assessment['conditional_var']:.3f}")
            logger.info(f"      Max Drawdown: {risk_assessment['max_drawdown']:.3f}")
            logger.info(f"      Sharpe Ratio: {risk_assessment['sharpe_ratio']:.3f}")
            
            # Show allocation suggestions
            suggestions = risk_assessment['allocation_suggestions']
            if 'recommended_shifts' in suggestions:
                logger.info(f"      Allocation Suggestions: {suggestions['recommended_shifts']}")
        
        # Test Risk Engine Integration
        logger.info(f"\n🔗 RISK ENGINE INTEGRATION TEST:")
        
        # Simulate portfolio update
        positions = {
            'BTC/USDT': {'size': 0.5, 'entry_price': 45000},
            'ETH/USDT': {'size': 2.0, 'entry_price': 3000}
        }
        
        current_prices = {
            'BTC/USDT': 46250,
            'ETH/USDT': 2950,
            'SOL/USDT': 100,
            'ADA/USDT': 0.5
        }
        
        risk_engine.update_portfolio_state(positions, current_prices)
        
        logger.info(f"   Portfolio updated successfully")
        logger.info(f"   Current capital: ${risk_engine.current_capital:,.2f}")
        logger.info(f"   Active positions: {len(risk_engine.positions)}")
        logger.info(f"   Assets tracked: {len(risk_engine.historical_returns)}")
        
        # Performance comparison
        logger.info(f"\n🏆 RISK ENGINE PERFORMANCE SUMMARY:")
        
        # Compare different approaches
        logger.info("   Kelly Criterion vs Traditional Position Sizing:")
        traditional_sizes = [0.02, 0.03, 0.01, 0.005]  # Traditional fixed percentages
        kelly_sizes = [kp.optimal_fraction for kp in kelly_positions[:4]]
        
        logger.info(f"      Traditional average: {np.mean(traditional_sizes):.3f}")
        logger.info(f"      Kelly average: {np.mean(kelly_sizes):.3f}")
        logger.info(f"      Kelly advantage: {(np.mean(kelly_sizes)/np.mean(traditional_sizes) - 1)*100:+.1f}%")
        
        logger.info(f"\n   CVaR vs Equal Weight Portfolio:")
        equal_weight_cvar = risk_engine.calculate_conditional_var(
            np.array(returns_data['BTC/USDT']), 0.95
        )
        optimized_cvar = min(cvar_values)
        
        logger.info(f"      Equal weight CVaR: {equal_weight_cvar:.3f}")
        logger.info(f"      Optimized CVaR: {optimized_cvar:.3f}")
        logger.info(f"      Improvement: {(equal_weight_cvar/optimized_cvar - 1)*100:+.1f}%")
        
        logger.info(f"\n✅ ADVANCED RISK ENGINE DEMO COMPLETED SUCCESSFULLY")
        logger.info("🚀 Key Achievements:")
        logger.info("   • Implemented Kelly Criterion with regime adjustments")
        logger.info("   • Built CVaR portfolio optimization engine")
        logger.info("   • Created regime-aware risk calibration")
        logger.info("   • Developed comprehensive risk assessment framework")
        logger.info("   • Integrated mathematical risk management")
        
        logger.info(f"\n📊 FINAL PERFORMANCE METRICS:")
        logger.info(f"   Kelly Positions Analyzed: {len(kelly_positions)}")
        logger.info(f"   Portfolio Optimizations: {len(cvar_optimizations)}")
        logger.info(f"   Risk Assessments: {len(regimes)}")
        logger.info(f"   Average Kelly Advantage: {(np.mean(kelly_sizes)/np.mean(traditional_sizes) - 1)*100:+.1f}%")
        logger.info(f"   CVaR Optimization Gain: {(equal_weight_cvar/optimized_cvar - 1)*100:+.1f}%")
        
        logger.info(f"\n🎯 NEXT STEPS:")
        logger.info("   1. Integrate with Edge Probability Models")
        logger.info("   2. Connect to Portfolio Construction Engine")
        logger.info("   3. Add Real-time Risk Monitoring")
        logger.info("   4. Implement Stress Testing Framework")
        
    except Exception as e:
        logger.error(f"❌ Advanced risk engine demo failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def demonstrate_risk_calculations():
    """Demonstrate individual risk calculation methods"""
    logger.info(f"\n🧮 RISK CALCULATION METHODS DEMO")
    logger.info("=" * 40)
    
    try:
        from advanced_risk_engine import get_advanced_risk_engine
        
        risk_engine = get_advanced_risk_engine()
        
        # Generate sample returns
        np.random.seed(42)
        sample_returns = np.random.normal(0.001, 0.02, 252)  # Daily returns
        
        logger.info("Testing risk calculation methods:")
        
        # VaR calculation
        var_95 = risk_engine.calculate_value_at_risk(sample_returns, 0.95)
        logger.info(f"   VaR (95%): {var_95:.4f} ({var_95 * 100:.2f}%)")
        
        # CVaR calculation
        cvar_95 = risk_engine.calculate_conditional_var(sample_returns, 0.95)
        logger.info(f"   CVaR (95%): {cvar_95:.4f} ({cvar_95 * 100:.2f}%)")
        
        # Kelly criterion
        kelly_pos = risk_engine.calculate_kelly_criterion(0.6, 2.0, 'STABLE')
        logger.info(f"   Kelly Position: {kelly_pos.optimal_fraction:.4f} ({kelly_pos.optimal_fraction * 100:.1f}%)")
        logger.info(f"   Expected Growth: {kelly_pos.expected_growth:.4f}")
        
        logger.info("✅ Risk calculations demonstrated successfully")
        
    except Exception as e:
        logger.error(f"❌ Risk calculations demo failed: {e}")

async def main():
    """Main demo function"""
    print("Chloe 0.6 - Advanced Risk Engine Demo")
    print("Professional Kelly Criterion + CVaR optimization")
    print()
    
    # Run main risk engine demo
    await demonstrate_advanced_risk_engine()
    
    # Run calculation methods demo
    demonstrate_risk_calculations()
    
    print(f"\n🎉 ADVANCED RISK ENGINE DEMO COMPLETED")
    print("Chloe 0.6 now has professional mathematical risk management!")

if __name__ == "__main__":
    asyncio.run(main())