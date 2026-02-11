"""
ML Core Module for Chloe AI
Generates trading signals using machine learning models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import joblib
import logging
from typing import Tuple, Dict, Optional
import os

logger = logging.getLogger(__name__)

class MLSignalsCore:
    """
    Machine Learning Core for generating trading signals
    """
    
    def __init__(self, model_type: str = 'xgboost'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_trained = False
        
    def prepare_features_and_target(self, df: pd.DataFrame, lookahead_period: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for training
        
        Args:
            df: DataFrame with indicators
            lookahead_period: Number of periods to look ahead for target
            
        Returns:
            Features DataFrame and target Series
        """
        # Select feature columns (exclude price data and target)
        feature_cols = [col for col in df.columns if col not in [
            'close', 'Close', 'open', 'Open', 'high', 'High', 'low', 'Low', 
            'volume', 'Volume', 'adj_close', 'Adj Close', 'target'
        ] and not col.startswith('target')]
        
        # Filter to only numeric columns that exist
        feature_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(df[col])]
        self.feature_columns = feature_cols
        
        X = df[self.feature_columns].copy()
        
        # Create target variable: 0=Sell, 1=Hold, 2=Buy
        # Based on future price movement
        close_prices = df['close'] if 'close' in df.columns else df['Close']
        future_prices = close_prices.shift(-lookahead_period)  # Shift forward to get future prices
        
        # Calculate price change percentage
        price_changes = (future_prices - close_prices) / close_prices
        
        # Define thresholds for buy/sell signals
        buy_threshold = 0.02  # 2% increase
        sell_threshold = -0.02  # 2% decrease
        
        y = pd.Series(1, index=df.index)  # Default to Hold (1)
        y[price_changes > buy_threshold] = 2  # Buy (2)
        y[price_changes < sell_threshold] = 0  # Sell (0)
        
        # Drop rows with NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        logger.info(f"✅ Prepared {len(X)} samples with {len(self.feature_columns)} features")
        return X, y
    
    def create_model(self):
        """Create ML model based on specified type"""
        if self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        logger.info(f"✅ Created {self.model_type} model")
    
    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
        """
        Train the ML model
        
        Args:
            X: Features DataFrame
            y: Target Series
            test_size: Proportion of data to use for testing
        """
        logger.info(f"🚀 Training {self.model_type} model...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create and train model
        self.create_model()
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        logger.info(f"✅ Model trained - Train accuracy: {train_score:.4f}, Test accuracy: {test_score:.4f}")
        
        # Detailed evaluation
        y_pred = self.model.predict(X_test_scaled)
        logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
        
        self.is_trained = True
    
    def predict_signals(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict trading signals for given features
        
        Args:
            X: Features DataFrame
            
        Returns:
            Array of predictions (0=Sell, 1=Hold, 2=Buy)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Ensure we have the right features
        X_filtered = X[self.feature_columns].copy()
        
        # Handle missing values
        X_filtered = X_filtered.fillna(method='ffill').fillna(method='bfill')
        
        # Scale features
        X_scaled = self.scaler.transform(X_filtered)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        # Get prediction probabilities
        probas = self.model.predict_proba(X_scaled)
        
        logger.info(f"✅ Generated {len(predictions)} trading signals")
        return predictions, probas
    
    def predict_with_probabilities(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict trading signals with probability scores
        
        Args:
            X: Features DataFrame
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Ensure we have the right features
        X_filtered = X[self.feature_columns].copy()
        
        # Handle missing values
        X_filtered = X_filtered.fillna(method='ffill').fillna(method='bfill')
        
        # Scale features
        X_scaled = self.scaler.transform(X_filtered)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        # Get prediction probabilities
        probas = self.model.predict_proba(X_scaled)
        
        logger.info(f"✅ Generated {len(predictions)} trading signals with probabilities")
        return predictions, probas
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the trained model
        
        Returns:
            DataFrame with feature importance
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        importance = self.model.feature_importances_
        feature_imp = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        logger.info("✅ Feature importance calculated")
        return feature_imp
    
    def save_model(self, filepath: str):
        """
        Save the trained model to disk
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model from disk
        
        Args:
            filepath: Path to load the model from
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"✅ Model loaded from {filepath}")

class SignalProcessor:
    """
    Processes raw model predictions into actionable trading signals
    """
    
    def __init__(self):
        self.signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
        self.reverse_signal_map = {'SELL': 0, 'HOLD': 1, 'BUY': 2}
    
    def process_predictions(self, predictions: np.ndarray, probabilities: np.ndarray, 
                          confidence_threshold: float = 0.6) -> pd.DataFrame:
        """
        Process model predictions into trading signals with confidence
        
        Args:
            predictions: Model predictions (0=Sell, 1=Hold, 2=Buy)
            probabilities: Prediction probabilities
            confidence_threshold: Minimum confidence to act on signal
            
        Returns:
            DataFrame with signals and confidence scores
        """
        signals = []
        
        for i, pred in enumerate(predictions):
            signal = self.signal_map[pred]
            
            # Get confidence score (probability of predicted class)
            confidence = probabilities[i][pred]
            
            # Lower confidence if it's close to another class
            max_other_prob = max([prob for j, prob in enumerate(probabilities[i]) if j != pred])
            adjusted_confidence = confidence - max_other_prob
            
            # Determine final signal based on confidence threshold
            if adjusted_confidence < (1 - confidence_threshold):
                final_signal = 'HOLD'  # Uncertain, hold position
                final_confidence = adjusted_confidence
            else:
                final_signal = signal
                final_confidence = confidence
            
            signals.append({
                'signal': final_signal,
                'confidence': final_confidence,
                'raw_prediction': signal,
                'raw_confidence': confidence
            })
        
        signals_df = pd.DataFrame(signals)
        logger.info(f"✅ Processed {len(signals_df)} predictions into trading signals")
        return signals_df

# Example usage
def main():
    """Example usage of the ML Signals Core"""
    print("ML Core module ready for training and prediction")

if __name__ == "__main__":
    main()