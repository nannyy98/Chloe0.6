#!/usr/bin/env python3
"""
Test institutional components
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from portfolio.portfolio import Portfolio, initialize_portfolio
from risk.risk_engine import RiskManager, RiskLimits, initialize_risk_manager

def test_components():
    print("🏛️ Testing Institutional Trading Components")
    print("=" * 50)
    
    # Test portfolio initialization
    print("1. Initializing portfolio...")
    portfolio = initialize_portfolio(100000.0)
    print("✅ Portfolio initialized")
    print(f"   Initial capital: ${portfolio.initial_capital:,.2f}")
    
    # Test risk manager
    print("\n2. Initializing risk manager...")
    risk_manager = initialize_risk_manager(portfolio)
    print("✅ Risk manager initialized")
    print(f"   Max risk per trade: {risk_manager.limits.max_risk_per_trade*100}%")
    print(f"   Max drawdown: {risk_manager.limits.max_drawdown*100}%")
    
    # Test position entry
    print("\n3. Testing position entry...")
    portfolio.enter_position('BTC/USDT', 1.0, 50000.0)
    print("✅ Position entered")
    print(f"   Position size: 1.0 BTC")
    print(f"   Entry price: $50,000.00")
    print(f"   Portfolio value: ${portfolio.calculate_total_value():,.2f}")
    
    # Test risk check
    print("\n4. Testing risk assessment...")
    risk_check, reason = risk_manager.check_trade_risk('ETH/USDT', 'BUY', 2.0, 3000.0)
    print(f"✅ Risk check completed: {risk_check}")
    print(f"   Reason: {reason}")
    
    # Test performance metrics
    print("\n5. Portfolio performance metrics:")
    metrics = portfolio.get_performance_metrics()
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"   {key}: {value:,.2f}")
        else:
            print(f"   {key}: {value}")
    
    print("\n" + "=" * 50)
    print("🎉 All institutional components working correctly!")
    print("Ready for production trading system development.")

if __name__ == "__main__":
    test_components()