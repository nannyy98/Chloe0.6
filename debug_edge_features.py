"""
Debug Edge Feature Preparation
"""
import pandas as pd
import numpy as np
from edge_classifier import EdgeClassifier

def debug_features():
    print("🔍 Debug Edge Feature Preparation")
    print("=" * 40)
    
    # Create test data - more samples to handle indicator lookbacks
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=300, freq='D')  # More data
    prices = 50000 * (1 + np.random.randn(300) * 0.02).cumprod()
    
    market_data = pd.DataFrame({
        'close': prices,
        'high': prices * (1 + abs(np.random.randn(300) * 0.01)),
        'low': prices * (1 - abs(np.random.randn(300) * 0.01)),
        'volume': np.random.uniform(1000, 10000, 300) * prices
    }, index=dates)
    
    print(f"Market data shape: {market_data.shape}")
    print(f"Columns: {list(market_data.columns)}")
    print(f"Index type: {type(market_data.index)}")
    
    # Initialize classifier
    clf = EdgeClassifier('random_forest')
    
    # Debug feature preparation step by step
    print("\n--- Debugging feature preparation ---")
    
    close_prices = market_data['close']
    print(f"Close prices: {len(close_prices)} samples")
    
    # Test individual feature calculations
    returns = close_prices.pct_change()
    print(f"Returns: {len(returns.dropna())} valid samples")
    
    vol_20 = returns.rolling(20).std()
    print(f"20-period volatility: {len(vol_20.dropna())} valid samples")
    
    # Try preparing features
    features = clf.prepare_edge_features(market_data)
    print(f"\nFinal features shape: {features.shape}")
    print(f"Feature columns: {list(features.columns) if len(features.columns) > 0 else 'NONE'}")
    
    if len(features) == 0:
        print("❌ No features generated - checking data requirements...")
        print(f"Data length: {len(market_data)}")
        print(f"Minimum required for 20-period indicators: 20")
        print(f"Minimum required for 60-period indicators: 60")

if __name__ == "__main__":
    debug_features()