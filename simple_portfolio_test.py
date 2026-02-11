"""
Simple Portfolio Construction Test
Minimal test to verify core portfolio construction functionality
"""
import pandas as pd
import numpy as np
from portfolio_constructor import PortfolioConstructor

def simple_portfolio_test():
    print("📊 Simple Portfolio Construction Test")
    print("=" * 40)
    
    # Create minimal test data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    
    # Create simple price series for 3 assets
    market_data_dict = {}
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    
    for symbol in symbols:
        prices = 50000 * (1 + np.random.randn(200) * 0.02).cumprod()
        market_data = pd.DataFrame({
            'close': prices,
            'high': prices * (1 + abs(np.random.randn(200) * 0.01)),
            'low': prices * (1 - abs(np.random.randn(200) * 0.01)),
            'volume': np.random.uniform(1000, 10000, 200) * prices
        }, index=dates)
        market_data_dict[symbol] = market_data
    
    print(f"✅ Created market data for {len(market_data_dict)} assets")
    
    # Initialize portfolio constructor
    portfolio_mgr = PortfolioConstructor(initial_capital=10000.0)
    print("✅ Initialized portfolio constructor")
    
    # Test portfolio construction
    try:
        allocations = portfolio_mgr.construct_optimal_portfolio(
            market_data_dict=market_data_dict
        )
        print(f"✅ Portfolio construction completed")
        print(f"   Positions allocated: {len(allocations)}")
        
        if allocations:
            print("   Allocation details:")
            for alloc in allocations:
                print(f"     {alloc.symbol}: {alloc.weight*100:.1f}% "
                      f"(Edge: {alloc.edge_probability:.3f})")
        
        # Test portfolio summary
        summary = portfolio_mgr.get_portfolio_summary()
        print(f"   Portfolio status: {summary['status']}")
        print(f"   Total value: ${summary['total_value']:,.2f}")
        print(f"   Cash: ${summary['cash']:,.2f}")
        print(f"   Leverage: {summary['leverage']*100:.1f}%")
        
    except Exception as e:
        print(f"❌ Portfolio construction failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False
    
    print("\n🎉 Simple portfolio test completed successfully!")
    return True

if __name__ == "__main__":
    simple_portfolio_test()