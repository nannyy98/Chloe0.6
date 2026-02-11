#!/usr/bin/env python3
"""
Top-Down Portfolio Construction Demo for Chloe 0.6
Professional portfolio construction from objectives down to individual trades
"""

import asyncio
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from top_down_portfolio import get_portfolio_constructor, PortfolioObjective, PortfolioConstraint

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_market_data(n_assets: int = 4) -> tuple:
    """Generate synthetic market data for portfolio construction"""
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT'][:n_assets]
    
    # Asset characteristics
    asset_data = {
        'BTC/USDT': {'expected_return': 0.15, 'volatility': 0.40, 'edge_prob': 0.55},
        'ETH/USDT': {'expected_return': 0.12, 'volatility': 0.35, 'edge_prob': 0.52},
        'SOL/USDT': {'expected_return': 0.20, 'volatility': 0.50, 'edge_prob': 0.48},
        'ADA/USDT': {'expected_return': 0.08, 'volatility': 0.25, 'edge_prob': 0.45}
    }
    
    # Generate correlation matrix
    correlation_matrix = np.array([
        [1.0, 0.7, 0.4, 0.2],
        [0.7, 1.0, 0.5, 0.3],
        [0.4, 0.5, 1.0, 0.1],
        [0.2, 0.3, 0.1, 1.0]
    ])
    
    # Create covariance matrix
    volatilities = [asset_data[sym]['volatility'] for sym in symbols]
    covariance_matrix = np.zeros((n_assets, n_assets))
    
    for i in range(n_assets):
        for j in range(n_assets):
            covariance_matrix[i, j] = correlation_matrix[i, j] * volatilities[i] * volatilities[j]
    
    # Prepare data dictionaries
    expected_returns = {sym: asset_data[sym]['expected_return'] for sym in symbols}
    market_view = {sym: asset_data[sym]['edge_prob'] for sym in symbols}
    
    return symbols, expected_returns, market_view, covariance_matrix

async def demonstrate_top_down_portfolio():
    """Demonstrate top-down portfolio construction capabilities"""
    logger.info("📈 TOP-DOWN PORTFOLIO CONSTRUCTION DEMO")
    logger.info("=" * 55)
    
    try:
        # Initialize portfolio constructor
        logger.info("🔧 Initializing Top-Down Portfolio Constructor...")
        portfolio_mgr = get_portfolio_constructor(initial_capital=100000.0)
        logger.info("✅ Portfolio Constructor initialized")
        
        # Generate market data
        logger.info("📊 Generating market data for portfolio construction...")
        symbols, expected_returns, market_view, covariance_matrix = generate_market_data(4)
        
        logger.info(f"   Assets: {symbols}")
        logger.info(f"   Expected Returns: {[f'{r:.1%}' for r in expected_returns.values()]}")
        logger.info(f"   Edge Probabilities: {[f'{p:.2f}' for p in market_view.values()]}")
        logger.info(f"   Covariance Matrix Shape: {covariance_matrix.shape}")
        
        # Test different portfolio objectives
        logger.info(f"\n🎯 PORTFOLIO OBJECTIVE OPTIMIZATION:")
        
        objectives = [
            PortfolioObjective.MAXIMIZE_SHARPE,
            PortfolioObjective.RISK_PARITY,
            PortfolioObjective.MINIMIZE_RISK,
            PortfolioObjective.TARGET_VOLATILITY,
            PortfolioObjective.MAXIMIZE_RETURN
        ]
        
        portfolio_plans = []
        
        for objective in objectives:
            logger.info(f"\n   Optimizing for {objective.value}:")
            
            # Custom constraints for different objectives
            if objective == PortfolioObjective.TARGET_VOLATILITY:
                constraints = PortfolioConstraint(
                    max_position_size=0.30,
                    min_position_size=0.05,
                    target_volatility=0.20
                )
            else:
                constraints = PortfolioConstraint(
                    max_position_size=0.25,
                    min_position_size=0.02
                )
            
            # Construct portfolio
            plan = portfolio_mgr.construct_portfolio(
                market_view=market_view,
                expected_returns=expected_returns,
                covariance_matrix=covariance_matrix,
                symbols=symbols,
                current_regime='STABLE',
                objective=objective,
                constraints=constraints
            )
            
            portfolio_plans.append(plan)
            
            # Display results
            logger.info(f"      Portfolio Metrics:")
            metrics = plan.portfolio_metrics
            logger.info(f"         Expected Return: {metrics['expected_return']:.2%}")
            logger.info(f"         Volatility: {metrics['volatility']:.2%}")
            logger.info(f"         Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
            logger.info(f"         Diversification: {metrics['diversification_ratio']:.2f}")
            
            logger.info(f"      Asset Allocations:")
            for alloc in plan.allocations[:3]:  # Show top 3 allocations
                logger.info(f"         {alloc.symbol}: {alloc.target_weight:.1%} (${alloc.target_dollar_amount:,.0f})")
            
            if len(plan.allocations) > 3:
                logger.info(f"         ... and {len(plan.allocations) - 3} more assets")
            
            logger.info(f"      Risk Budget Allocation:")
            top_risk_assets = sorted(plan.risk_budget_allocation.items(), 
                                   key=lambda x: x[1], reverse=True)[:2]
            for symbol, budget in top_risk_assets:
                logger.info(f"         {symbol}: {budget:.1%} of portfolio risk")
            
            logger.info(f"      Execution Priority: {plan.execution_priority[:3]}")
            logger.info(f"      Rebalance Required: {plan.rebalance_required}")
        
        # Compare portfolio approaches
        logger.info(f"\n📊 PORTFOLIO APPROACH COMPARISON:")
        
        comparison_metrics = ['expected_return', 'volatility', 'sharpe_ratio']
        
        for metric in comparison_metrics:
            values = [plan.portfolio_metrics[metric] for plan in portfolio_plans]
            logger.info(f"   {metric.replace('_', ' ').title()}:")
            logger.info(f"      Range: {min(values):.3f} to {max(values):.3f}")
            logger.info(f"      Average: {np.mean(values):.3f}")
            logger.info(f"      Best Approach: {objectives[np.argmax(values)].value}")
        
        # Test regime-aware construction
        logger.info(f"\n🔍 REGIME-AWARE PORTFOLIO CONSTRUCTION:")
        
        regimes = ['STABLE', 'TRENDING', 'VOLATILE', 'CRISIS']
        
        regime_results = {}
        
        for regime in regimes:
            logger.info(f"   Constructing portfolio for {regime} regime:")
            
            plan = portfolio_mgr.construct_portfolio(
                market_view=market_view,
                expected_returns=expected_returns,
                covariance_matrix=covariance_matrix,
                symbols=symbols,
                current_regime=regime,
                objective=PortfolioObjective.MAXIMIZE_SHARPE
            )
            
            regime_results[regime] = plan
            
            logger.info(f"      Aggressive Assets Allocation:")
            aggressive_symbols = ['BTC/USDT', 'SOL/USDT']
            aggressive_weight = sum(alloc.target_weight for alloc in plan.allocations 
                                  if alloc.symbol in aggressive_symbols)
            logger.info(f"         High-volatility assets: {aggressive_weight:.1%}")
            
            logger.info(f"      Conservative Assets Allocation:")
            conservative_symbols = ['ADA/USDT']
            conservative_weight = sum(alloc.target_weight for alloc in plan.allocations 
                                    if alloc.symbol in conservative_symbols)
            logger.info(f"         Low-volatility assets: {conservative_weight:.1%}")
            
            logger.info(f"      Portfolio Volatility: {plan.portfolio_metrics['volatility']:.2%}")
        
        # Analyze regime adaptation
        logger.info(f"\n📊 REGIME ADAPTATION ANALYSIS:")
        
        volatilities = [result.portfolio_metrics['volatility'] for result in regime_results.values()]
        returns = [result.portfolio_metrics['expected_return'] for result in regime_results.values()]
        
        logger.info(f"   Volatility Range: {min(volatilities):.2%} to {max(volatilities):.2%}")
        logger.info(f"   Return Range: {min(returns):.2%} to {max(returns):.2%}")
        logger.info(f"   Risk-Adjusted Returns Vary by Regime: {len(set([f'{r:.3f}' for r in returns])) > 1}")
        
        # Test portfolio rebalancing
        logger.info(f"\n🔄 PORTFOLIO REBALANCING SCENARIOS:")
        
        # Simulate current positions different from target
        current_positions = {
            'BTC/USDT': {'weight': 0.10, 'size': 0.1},
            'ETH/USDT': {'weight': 0.20, 'size': 0.5},
            'SOL/USDT': {'weight': 0.40, 'size': 1.0},
            'ADA/USDT': {'weight': 0.30, 'size': 1000}
        }
        
        portfolio_mgr.update_portfolio_state(current_positions, 
                                           {sym: 50000 for sym in symbols})  # Mock prices
        
        # Check rebalancing requirement
        plan = portfolio_mgr.construct_portfolio(
            market_view=market_view,
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
            symbols=symbols,
            current_regime='STABLE',
            objective=PortfolioObjective.MAXIMIZE_SHARPE
        )
        
        logger.info(f"   Current vs Target Analysis:")
        for alloc in plan.allocations:
            current_weight = current_positions.get(alloc.symbol, {}).get('weight', 0.0)
            deviation = abs(alloc.target_weight - current_weight)
            logger.info(f"      {alloc.symbol}: Current {current_weight:.1%} → Target {alloc.target_weight:.1%} (Deviation: {deviation:.1%})")
        
        logger.info(f"   Rebalancing Required: {plan.rebalance_required}")
        logger.info(f"   Execution Priority: {plan.execution_priority}")
        
        # Performance optimization analysis
        logger.info(f"\n🏆 PORTFOLIO OPTIMIZATION SUMMARY:")
        
        # Calculate improvement metrics
        baseline_sharpe = portfolio_plans[0].portfolio_metrics['sharpe_ratio']  # First approach as baseline
        best_sharpe = max(plan.portfolio_metrics['sharpe_ratio'] for plan in portfolio_plans)
        best_volatility = min(plan.portfolio_metrics['volatility'] for plan in portfolio_plans)
        
        logger.info(f"   Sharpe Ratio Improvement: {((best_sharpe/baseline_sharpe - 1)*100):+.1f}%")
        logger.info(f"   Risk Reduction: {((baseline_sharpe/best_volatility - 1)*100):+.1f}%")
        logger.info(f"   Diversification Gains: Up to {(max(plan.portfolio_metrics['diversification_ratio'] for plan in portfolio_plans) - 1)*100:.1f}%")
        
        # Asset allocation efficiency
        all_weights = []
        for plan in portfolio_plans:
            weights = [alloc.target_weight for alloc in plan.allocations]
            all_weights.extend(weights)
        
        logger.info(f"   Weight Concentration: {np.std(all_weights):.3f} (lower = more balanced)")
        logger.info(f"   Position Size Range: {min(all_weights):.1%} to {max(all_weights):.1%}")
        
        logger.info(f"\n✅ TOP-DOWN PORTFOLIO CONSTRUCTION DEMO COMPLETED SUCCESSFULLY")
        logger.info("🚀 Key Achievements:")
        logger.info("   • Implemented professional top-down portfolio construction")
        logger.info("   • Created multiple optimization objectives (Sharpe, Risk Parity, etc.)")
        logger.info("   • Built regime-aware portfolio adaptation")
        logger.info("   • Developed automated rebalancing triggers")
        logger.info("   • Integrated risk budget allocation")
        
        logger.info(f"\n📊 FINAL PERFORMANCE METRICS:")
        logger.info(f"   Portfolio Objectives Tested: {len(objectives)}")
        logger.info(f"   Regimes Analyzed: {len(regimes)}")
        logger.info(f"   Best Sharpe Ratio Achieved: {best_sharpe:.3f}")
        logger.info(f"   Optimal Volatility Level: {best_volatility:.2%}")
        logger.info(f"   Portfolio Efficiency Gain: {((best_sharpe/baseline_sharpe - 1)*100):+.1f}%")
        
        logger.info(f"\n🎯 NEXT STEPS:")
        logger.info("   1. Integrate with all Chloe 0.6 components")
        logger.info("   2. Add real-time portfolio monitoring")
        logger.info("   3. Implement execution algorithms")
        logger.info("   4. Add stress testing and scenario analysis")
        
    except Exception as e:
        logger.error(f"❌ Top-down portfolio demo failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def demonstrate_portfolio_methods():
    """Demonstrate individual portfolio construction methods"""
    logger.info(f"\n🧮 PORTFOLIO CONSTRUCTION METHODS DEMO")
    logger.info("=" * 45)
    
    try:
        from top_down_portfolio import get_portfolio_constructor
        
        portfolio_mgr = get_portfolio_constructor(50000.0)
        
        # Generate sample data
        symbols, expected_returns, market_view, cov_matrix = generate_market_data(3)
        
        logger.info("Testing portfolio construction methods:")
        
        # Risk Parity
        risk_parity_plan = portfolio_mgr.construct_portfolio(
            market_view, expected_returns, cov_matrix, symbols,
            'STABLE', PortfolioObjective.RISK_PARITY
        )
        logger.info(f"   Risk Parity - Sharpe: {risk_parity_plan.portfolio_metrics['sharpe_ratio']:.3f}")
        
        # Maximum Sharpe
        sharpe_plan = portfolio_mgr.construct_portfolio(
            market_view, expected_returns, cov_matrix, symbols,
            'STABLE', PortfolioObjective.MAXIMIZE_SHARPE
        )
        logger.info(f"   Max Sharpe - Sharpe: {sharpe_plan.portfolio_metrics['sharpe_ratio']:.3f}")
        
        # Minimum Risk
        min_risk_plan = portfolio_mgr.construct_portfolio(
            market_view, expected_returns, cov_matrix, symbols,
            'STABLE', PortfolioObjective.MINIMIZE_RISK
        )
        logger.info(f"   Min Risk - Volatility: {min_risk_plan.portfolio_metrics['volatility']:.2%}")
        
        logger.info("✅ Portfolio methods demonstrated successfully")
        
    except Exception as e:
        logger.error(f"❌ Portfolio methods demo failed: {e}")

async def main():
    """Main demo function"""
    print("Chloe 0.6 - Top-Down Portfolio Construction Demo")
    print("Professional portfolio construction from objectives to trades")
    print()
    
    # Run main portfolio construction demo
    await demonstrate_top_down_portfolio()
    
    # Run methods demonstration
    demonstrate_portfolio_methods()
    
    print(f"\n🎉 TOP-DOWN PORTFOLIO CONSTRUCTION DEMO COMPLETED")
    print("Chloe 0.6 now has professional portfolio construction capabilities!")

if __name__ == "__main__":
    asyncio.run(main())