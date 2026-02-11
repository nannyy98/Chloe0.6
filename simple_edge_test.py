"""
Simple Edge Classification Test
Minimal test to verify core functionality works
"""
import pandas as pd
import numpy as np
from edge_classifier import EdgeClassifier

def simple_test():
    print("🎯 Simple Edge Classification Test")
    print("=" * 40)
    
    # Create minimal test data - more samples for indicators
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=300, freq='D')  # More data
    prices = 50000 * (1 + np.random.randn(300) * 0.02).cumprod()
    
    market_data = pd.DataFrame({
        'close': prices,
        'high': prices * (1 + abs(np.random.randn(300) * 0.01)),
        'low': prices * (1 - abs(np.random.randn(300) * 0.01)),
        'volume': np.random.uniform(1000, 10000, 300) * prices
    }, index=dates)
    
    print(f"✅ Created market data: {len(market_data)} samples")
    
    # Initialize edge classifier
    clf = EdgeClassifier('random_forest')
    print("✅ Initialized edge classifier")
    
    # Prepare features
    features = clf.prepare_edge_features(market_data)
    print(f"✅ Prepared features: {features.shape}")
    
    # Create simple labels (random for testing)
    labels = pd.Series(np.random.choice([0, 1], len(features)), index=features.index)
    print(f"✅ Created labels: {labels.sum()}/{len(labels)} positive")
    
    # Train model
    try:
        results = clf.train(features, labels)
        print("✅ Training completed successfully")
        print(f"   CV Score: {list(results.values())[0]['mean_cv_score']:.3f}")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False
    
    # Make predictions
    try:
        predictions = clf.predict_edge(features.tail(10))
        print("✅ Predictions generated successfully")
        print(f"   Recent predictions: {predictions['ensemble_pred'].tolist()}")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return False
    
    print("\n🎉 Simple test completed successfully!")
    return True

if __name__ == "__main__":
    simple_test()