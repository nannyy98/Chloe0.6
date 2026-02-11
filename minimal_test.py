import numpy as np
import pandas as pd
from datetime import datetime
from regime_detection import RegimeDetector

# Minimal test
print("Starting minimal test...")

# Create minimal data
dates = pd.date_range(end=datetime.now(), periods=50, freq='D')
prices = 50000 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 50)))

df = pd.DataFrame({
    'close': prices,
    'volume': np.random.uniform(1000, 10000, 50)
})
df.index = dates

print(f"Data created: {len(df)} rows")

# Test regime detector
detector = RegimeDetector()
print("Detector initialized")

success = detector.train_hmm(df)
print(f"Training result: {success}")

regime = detector.detect_current_regime(df)
print(f"Detected: {regime.name} ({regime.probability:.3f})")

print("Test completed!")