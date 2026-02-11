"""
Enhanced ML Core for Chloe AI
Advanced machine learning with ensemble methods and feature selection
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
import xgboost as xgb
import joblib
import logging
from typing import Tuple, Dict, Optional, List
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class EnhancedMLCore:
    """
    Enhanced Machine Learning Core with advanced features
    """
    
    def __init__(self, model_type: str = 'ensemble'):
        self.model_type = model_type
        self.models = {}
        self.scaler = RobustScaler()  # More robust to outliers
        self.feature_selector = None
        self.selected_features = []
        self.feature_importance = {}
        self.is_trained = False
        self.validation_scores = {}
        
        # Initialize models based on type
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize machine learning models"""
        if self.model_type == 'ensemble':
            self.models = {
                'rf': RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1
                ),
                'xgb': xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    objective='multi:softmax',
                    num_class=5
                ),
                'gbm': GradientBoostingClassifier(
                    n_estimators=150,
                    max_depth=10,
                    learning_rate=0.05,
                    subsample=0.8,
                    random_state=42
                )
            }
        elif self.model_type == 'xgboost':
            self.models = {
                'main': xgb.XGBClassifier(
                    n_estimators=300,
                    max_depth=10,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    objective='multi:softmax',
                    num_class=5
                )
            }
        elif self.model_type == 'random_forest':
            self.models = {
                'main': RandomForestClassifier(
                    n_estimators=300,
                    max_depth=20,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
            }
        
        logger.info(f"✅ Initialized {self.model_type} models: {list(self.models.keys())}")
    
    def prepare_features_and_target(self, df: pd.DataFrame, lookahead_period: int = 5,
                                  max_features: int = 100) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Enhanced feature preparation with automatic selection
        
        Args:
            df: DataFrame with all features
            lookahead_period: Number of periods to look ahead for target
            max_features: Maximum number of features to select
            
        Returns:
            Features DataFrame and target Series
        """
        logger.info("🔧 Preparing enhanced features and targets...")
        
        # Select potential feature columns
        feature_cols = [col for col in df.columns if col not in [
            'close', 'Close', 'open', 'Open', 'high', 'High', 'low', 'Low', 
            'volume', 'Volume', 'adj_close', 'Adj Close', 'target'
        ] and not col.startswith('target') and df[col].dtype in ['float64', 'int64']]
        
        # Filter out columns with too many missing values
        feature_cols = [col for col in feature_cols if df[col].notna().sum() > len(df) * 0.7]
        
        X = df[feature_cols].copy()
        
        # Handle missing values
        X = X.ffill().bfill().fillna(0)
        
        # Create target variable: 0=Strong Sell, 1=Sell, 2=Hold, 3=Buy, 4=Strong Buy
        close_prices = df['close'] if 'close' in df.columns else df['Close']
        future_prices = close_prices.shift(-lookahead_period)
        
        # Calculate price change percentage
        price_changes = (future_prices - close_prices) / close_prices
        
        # Define thresholds for multi-class signals
        strong_buy_threshold = 0.05    # 5% increase
        buy_threshold = 0.02           # 2% increase
        sell_threshold = -0.02         # 2% decrease
        strong_sell_threshold = -0.05  # 5% decrease
        
        y = pd.Series(2, index=df.index)  # Default to Hold (2)
        y[price_changes > strong_buy_threshold] = 4    # Strong Buy (4)
        y[price_changes > buy_threshold] = 3           # Buy (3)
        y[price_changes < strong_sell_threshold] = 0   # Strong Sell (0)
        y[price_changes < sell_threshold] = 1          # Sell (1)
        
        # Drop rows with NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        # Adjust target values to ensure they start from 0 and are consecutive
        # This is required for XGBoost compatibility
        unique_vals = sorted(y.unique())
        label_map = {val: idx for idx, val in enumerate(unique_vals)}
        y = y.map(label_map).astype(int)
        
        # Feature selection
        if len(X.columns) > max_features and len(X) > 50:
            logger.info(f"🔍 Selecting top {max_features} features from {len(X.columns)} candidates...")
            
            # Use SelectKBest for initial filtering
            selector = SelectKBest(score_func=f_classif, k=min(max_features, len(X.columns)))
            X_selected = selector.fit_transform(X, y)
            
            # Get selected feature names
            selected_indices = selector.get_support(indices=True)
            self.selected_features = [X.columns[i] for i in selected_indices]
            X = pd.DataFrame(X_selected, index=X.index, columns=self.selected_features)
            
            # Store feature scores
            self.feature_importance = dict(zip(self.selected_features, selector.scores_))
            
            logger.info(f"✅ Selected {len(self.selected_features)} most relevant features")
        else:
            self.selected_features = list(X.columns)
            # Calculate basic feature importance
            self.feature_importance = {col: 1.0 for col in self.selected_features}
        
        logger.info(f"✅ Prepared {len(X)} samples with {len(self.selected_features)} features")
        logger.info(f"📊 Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2,
              cv_folds: int = 5, validate: bool = True):
        """
        Enhanced training with cross-validation and validation
        
        Args:
            X: Features DataFrame
            y: Target Series
            test_size: Proportion of data to use for testing
            cv_folds: Number of cross-validation folds
            validate: Whether to perform validation
        """
        logger.info(f"🚀 Starting enhanced training with {self.model_type} approach...")
        
        if len(X) < 50:
            raise ValueError("Need at least 50 samples for training")
        
        # Time series split for financial data
        if validate and len(X) > 100:
            tscv = TimeSeriesSplit(n_splits=min(cv_folds, 5))
            
            # Cross-validation
            cv_scores = {}
            for name, model in self.models.items():
                try:
                    scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
                    cv_scores[name] = {
                        'mean': scores.mean(),
                        'std': scores.std(),
                        'scores': scores.tolist()
                    }
                    logger.info(f"📊 {name.upper()} CV Score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
                except Exception as e:
                    logger.warning(f"❌ CV failed for {name}: {e}")
                    cv_scores[name] = {'mean': 0, 'std': 0, 'scores': []}
            
            self.validation_scores = cv_scores
        
        # Final training on all data
        logger.info("🎯 Training final models...")
        
        for name, model in self.models.items():
            try:
                # Re-initialize XGBoost models with correct number of classes
                if 'xgb' in name.lower():
                    n_classes = len(np.unique(y))
                    if n_classes > 0:
                        # Recreate the XGBoost classifier with correct num_class
                        if self.model_type == 'xgboost':
                            self.models[name] = xgb.XGBClassifier(
                                n_estimators=300,
                                max_depth=10,
                                learning_rate=0.03,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                random_state=42,
                                n_jobs=-1,
                                objective='multi:softmax',
                                num_class=n_classes
                            )
                        else:  # ensemble mode
                            self.models[name] = xgb.XGBClassifier(
                                n_estimators=200,
                                max_depth=8,
                                learning_rate=0.05,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                random_state=42,
                                n_jobs=-1,
                                objective='multi:softmax',
                                num_class=n_classes
                            )
                        # Update the model reference
                        model = self.models[name]
                
                model.fit(X, y)
                logger.info(f"✅ {name.upper()} model trained successfully")
            except Exception as e:
                logger.error(f"❌ Error training {name}: {e}")
        
        # For ensemble, create voting classifier
        if self.model_type == 'ensemble' and len(self.models) > 1:
            try:
                estimators = [(name, model) for name, model in self.models.items()]
                self.ensemble_model = VotingClassifier(
                    estimators=estimators,
                    voting='soft',  # Use probability averaging
                    weights=[1.0] * len(estimators)  # Equal weights for now
                )
                self.ensemble_model.fit(X, y)
                logger.info("✅ Ensemble model created")
            except Exception as e:
                logger.warning(f"❌ Ensemble creation failed: {e}")
        
        # Calculate final feature importance
        self._calculate_feature_importance(X, y)
        
        self.is_trained = True
        logger.info("✅ Enhanced training completed")
    
    def _calculate_feature_importance(self, X: pd.DataFrame, y: pd.Series):
        """Calculate comprehensive feature importance"""
        importance_scores = {}
        
        # Get importance from tree-based models
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                model_importance = model.feature_importances_
                for i, feature in enumerate(X.columns):
                    if feature not in importance_scores:
                        importance_scores[feature] = []
                    importance_scores[feature].append(model_importance[i])
        
        # Average importance scores
        for feature in importance_scores:
            importance_scores[feature] = np.mean(importance_scores[feature])
        
        # Normalize to 0-1 range
        if importance_scores:
            max_imp = max(importance_scores.values())
            if max_imp > 0:
                self.feature_importance = {k: v/max_imp for k, v in importance_scores.items()}
            else:
                self.feature_importance = importance_scores
        
        # Sort by importance
        self.feature_importance = dict(sorted(self.feature_importance.items(), 
                                            key=lambda x: x[1], reverse=True))
    
    def predict_with_confidence(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Enhanced prediction with confidence scores
        
        Args:
            X: Features DataFrame
            
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Ensure we have the right features
        if not all(col in X.columns for col in self.selected_features):
            missing_cols = set(self.selected_features) - set(X.columns)
            logger.warning(f"Missing features: {missing_cols}")
            # Use available features
            available_features = [col for col in self.selected_features if col in X.columns]
            X_filtered = X[available_features].copy()
        else:
            X_filtered = X[self.selected_features].copy()
        
        # Handle missing values
        X_filtered = X_filtered.ffill().bfill().fillna(0)
        
        predictions_list = []
        probabilities_list = []
        
        # Get predictions from all models
        for name, model in self.models.items():
            try:
                pred = model.predict(X_filtered)
                probas = model.predict_proba(X_filtered)
                predictions_list.append(pred)
                probabilities_list.append(probas)
            except Exception as e:
                logger.warning(f"Error predicting with {name}: {e}")
        
        # For ensemble prediction
        if hasattr(self, 'ensemble_model'):
            try:
                ensemble_pred = self.ensemble_model.predict(X_filtered)
                ensemble_probas = self.ensemble_model.predict_proba(X_filtered)
                predictions_list.append(ensemble_pred)
                probabilities_list.append(ensemble_probas)
            except Exception as e:
                logger.warning(f"Error with ensemble prediction: {e}")
        
        if not predictions_list:
            raise ValueError("No valid predictions generated")
        
        # Ensemble predictions (simple averaging)
        final_predictions = np.mean(predictions_list, axis=0).round().astype(int)
        
        # Ensemble probabilities (average)
        if probabilities_list:
            avg_probabilities = np.mean(probabilities_list, axis=0)
            # Confidence as max probability
            confidence_scores = np.max(avg_probabilities, axis=1)
        else:
            # Fallback confidence based on prediction consistency
            pred_std = np.std(predictions_list, axis=0)
            confidence_scores = 1 - (pred_std / 2.0)  # Normalize to 0-1
        
        logger.info(f"✅ Generated {len(final_predictions)} enhanced predictions")
        return final_predictions, confidence_scores
    
    def get_model_performance(self) -> Dict:
        """
        Get comprehensive model performance metrics
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.is_trained:
            return {"error": "Model not trained"}
        
        performance = {
            "model_type": self.model_type,
            "selected_features": len(self.selected_features),
            "validation_scores": self.validation_scores,
            "feature_importance_top_10": dict(list(self.feature_importance.items())[:10])
        }
        
        return performance
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'models': self.models,
            'ensemble_model': getattr(self, 'ensemble_model', None),
            'scaler': self.scaler,
            'selected_features': self.selected_features,
            'feature_importance': self.feature_importance,
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'validation_scores': self.validation_scores
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        if model_data['ensemble_model'] is not None:
            self.ensemble_model = model_data['ensemble_model']
        self.scaler = model_data['scaler']
        self.selected_features = model_data['selected_features']
        self.feature_importance = model_data['feature_importance']
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']
        self.validation_scores = model_data['validation_scores']
        
        logger.info(f"✅ Model loaded from {filepath}")

class SignalInterpreter:
    """
    Interprets ML predictions into actionable trading signals
    """
    
    def __init__(self):
        self.signal_mapping = {
            0: 'STRONG_SELL',
            1: 'SELL', 
            2: 'HOLD',
            3: 'BUY',
            4: 'STRONG_BUY'
        }
        self.reverse_mapping = {v: k for k, v in self.signal_mapping.items()}
    
    def interpret_predictions(self, predictions: np.ndarray, 
                           confidence_scores: np.ndarray,
                           confidence_threshold: float = 0.6) -> pd.DataFrame:
        """
        Convert raw predictions to trading signals
        
        Args:
            predictions: Raw model predictions (0-4)
            confidence_scores: Confidence scores (0-1)
            confidence_threshold: Minimum confidence for action signals
            
        Returns:
            DataFrame with interpreted signals
        """
        signals = []
        
        for pred, conf in zip(predictions, confidence_scores):
            raw_signal = self.signal_mapping.get(int(pred), 'HOLD')
            
            # Apply confidence filtering
            if conf < confidence_threshold and raw_signal != 'HOLD':
                final_signal = 'HOLD'
                final_confidence = conf
            else:
                final_signal = raw_signal
                final_confidence = conf
            
            # Risk adjustment based on confidence
            if final_confidence > 0.8:
                risk_level = 'LOW'
            elif final_confidence > 0.6:
                risk_level = 'MEDIUM'
            else:
                risk_level = 'HIGH'
            
            signals.append({
                'signal': final_signal,
                'raw_signal': raw_signal,
                'confidence': final_confidence,
                'risk_level': risk_level,
                'strength': pred  # 0-4 scale
            })
        
        return pd.DataFrame(signals)

# Example usage
def main():
    """Example usage of Enhanced ML Core"""
    print("Enhanced ML Core module ready for advanced training")

if __name__ == "__main__":
    main()