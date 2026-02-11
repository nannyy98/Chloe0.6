"""
Minimal Edge Classifier Implementation
Focus on getting core functionality working first
"""
import numpy as np
import pandas as pd
from typing import Dict
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class MinimalEdgeClassifier:
    """Minimal working edge classifier"""
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
    
    def prepare_simple_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare minimal set of features that definitely work"""
        close_prices = market_data['close']
        
        # Very simple features that should work
        features = pd.DataFrame(index=market_data.index)
        
        # Simple momentum (5-day)
        features['momentum_5'] = close_prices.pct_change(5)
        
        # Simple volatility (10-day std)
        features['volatility_10'] = close_prices.pct_change().rolling(10).std()
        
        # Simple volume ratio
        if 'volume' in market_data.columns:
            features['volume_ratio'] = market_data['volume'] / market_data['volume'].rolling(10).mean()
        
        # Simple price position (vs 20-day MA)
        ma_20 = close_prices.rolling(20).mean()
        features['price_position'] = (close_prices - ma_20) / ma_20
        
        # Remove initial NaN values
        features = features.dropna()
        
        logger.info(f"✅ Prepared {len(features.columns)} simple features for {len(features)} samples")
        self.feature_names = list(features.columns)
        
        return features
    
    def create_simple_labels(self, market_data: pd.DataFrame, threshold: float = 0.02) -> pd.Series:
        """Create simple binary labels based on future returns"""
        close_prices = market_data['close']
        future_returns = close_prices.shift(-5) / close_prices - 1  # 5-day future return
        labels = (future_returns > threshold).astype(int)
        return labels.dropna()
    
    def train(self, features: pd.DataFrame, labels: pd.Series) -> Dict:
        """Train the classifier"""
        try:
            # Align indices
            common_index = features.index.intersection(labels.index)
            X = features.loc[common_index]
            y = labels.loc[common_index]
            
            logger.info(f"Training on {len(X)} samples")
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            cv_scores = cross_val_score(self.model, X_scaled, y, cv=tscv, scoring='roc_auc')
            
            # Train final model
            self.model.fit(X_scaled, y)
            self.is_trained = True
            
            results = {
                'mean_cv_score': cv_scores.mean(),
                'std_cv_score': cv_scores.std()
            }
            
            logger.info(f"✅ Model trained (CV AUC: {results['mean_cv_score']:.3f})")
            return results
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
    
    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Make predictions"""
        if not self.is_trained:
            return pd.DataFrame()
        
        try:
            X_scaled = self.scaler.transform(features)
            proba = self.model.predict_proba(X_scaled)[:, 1]
            pred = self.model.predict(X_scaled)
            
            results = pd.DataFrame({
                'probability': proba,
                'prediction': pred
            }, index=features.index)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            return pd.DataFrame()

def test_minimal_classifier():
    """Test the minimal implementation"""
    print("🎯 Testing Minimal Edge Classifier")
    print("=" * 40)
    
    # Create test data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    prices = 50000 * (1 + np.random.randn(200) * 0.02).cumprod()
    
    market_data = pd.DataFrame({
        'close': prices,
        'volume': np.random.uniform(1000, 10000, 200) * prices
    }, index=dates)
    
    print(f"✅ Created market data: {len(market_data)} samples")
    
    # Initialize classifier
    clf = MinimalEdgeClassifier()
    print("✅ Initialized classifier")
    
    # Prepare features
    features = clf.prepare_simple_features(market_data)
    print(f"✅ Prepared features: {features.shape}")
    
    # Create labels
    labels = clf.create_simple_labels(market_data)
    print(f"✅ Created labels: {labels.sum()}/{len(labels)} positive ({labels.mean()*100:.1f}%)")
    
    # Train model
    results = clf.train(features, labels)
    print(f"✅ Training completed: AUC = {results['mean_cv_score']:.3f}")
    
    # Make predictions
    predictions = clf.predict(features.tail(10))
    print(f"✅ Predictions: {predictions['prediction'].tolist()}")
    print(f"   Probabilities: {[f'{p:.3f}' for p in predictions['probability']]}")
    
    print("\n🎉 Minimal classifier test successful!")
    return True

if __name__ == "__main__":
    test_minimal_classifier()