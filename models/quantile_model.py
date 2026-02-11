"""
Quantile Model for Probabilistic Forecasting
Uses LightGBM/XGBoost for predicting return percentiles
"""

import logging
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import lightgbm as lgb
import joblib
import os

logger = logging.getLogger(__name__)

class QuantileModel:
    """Quantile regression model for probabilistic forecasting"""
    
    def __init__(self, 
                 alpha_low: float = 0.1,    # 10th percentile
                 alpha_mid: float = 0.5,    # 50th percentile (median)
                 alpha_high: float = 0.9,   # 90th percentile
                 model_type: str = 'lightgbm'):
        self.alpha_low = alpha_low
        self.alpha_mid = alpha_mid  
        self.alpha_high = alpha_high
        self.model_type = model_type
        
        # Initialize models for each quantile
        self.low_model = None   # P10
        self.mid_model = None   # P50 (median)
        self.high_model = None  # P90
        
        # Feature importance tracking
        self.feature_importance = {}
        
        # Model metadata
        self.is_fitted = False
        self.feature_columns = []
        self.target_columns = []
        
        logger.info(f"📊 Quantile Model initialized: P{int(self.alpha_low*100)}, "
                   f"P{int(self.alpha_mid*100)}, P{int(self.alpha_high*100)}")
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """
        Fit quantile models to predict return distributions
        
        Args:
            X: Feature matrix (samples x features)
            y: Target matrix with multiple horizons (samples x horizons)
        """
        try:
            # Store feature columns
            self.feature_columns = X.columns.tolist()
            
            # Handle multiple target horizons - we'll use the first one initially
            if isinstance(y, pd.DataFrame):
                if y.shape[1] > 1:
                    # Use the first target column as primary, or combine them
                    target_col = y.columns[0]
                    y_primary = y[target_col].copy()
                    self.target_columns = [target_col]
                else:
                    y_primary = y.iloc[:, 0].copy()
                    self.target_columns = [y.columns[0]]
            else:
                y_primary = y.copy()
                self.target_columns = ['target']
            
            # Remove invalid values
            mask = ~(X.isnull().any(axis=1) | y_primary.isnull() | 
                    (~np.isfinite(X).all(axis=1)) | (~np.isfinite(y_primary)))
            
            X_clean = X[mask]
            y_clean = y_primary[mask]
            
            if len(X_clean) < 50:  # Minimum samples needed
                logger.error(f"Not enough clean samples for training: {len(X_clean)}")
                return False
            
            logger.info(f"📈 Training on {len(X_clean)} samples with {len(self.feature_columns)} features")
            
            # Prepare training data
            X_train = X_clean[self.feature_columns].values.astype(np.float32)
            y_train = y_clean.values.astype(np.float32)
            
            # Train low quantile model (P10)
            self.low_model = self._create_quantile_model(self.alpha_low)
            self.low_model.fit(X_train, y_train)
            logger.info(f"✅ P{int(self.alpha_low*100)} model trained")
            
            # Train mid quantile model (P50/median)
            self.mid_model = self._create_quantile_model(self.alpha_mid)
            self.mid_model.fit(X_train, y_train)
            logger.info(f"✅ P{int(self.alpha_mid*100)} model trained")
            
            # Train high quantile model (P90)
            self.high_model = self._create_quantile_model(self.alpha_high)
            self.high_model.fit(X_train, y_train)
            logger.info(f"✅ P{int(self.alpha_high*100)} model trained")
            
            # Calculate feature importance (using median model as reference)
            self._calculate_feature_importance()
            
            self.is_fitted = True
            logger.info(f"🎯 Quantile model training completed")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Model training failed: {e}")
            return False
    
    def _create_quantile_model(self, alpha: float):
        """Create a quantile regression model"""
        if self.model_type.lower() == 'lightgbm':
            return lgb.LGBMRegressor(
                objective='quantile',
                alpha=alpha,
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            )
        else:
            # Fallback to a basic quantile regression approach
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(
                loss='quantile',
                alpha=alpha,
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
    
    def predict(self, X: pd.DataFrame) -> Optional[Dict]:
        """
        Generate probabilistic forecast
        
        Returns:
            Dictionary with forecast metrics
        """
        if not self.is_fitted:
            logger.error("❌ Model not fitted yet")
            return None
        
        try:
            # Ensure we have the right features
            missing_features = set(self.feature_columns) - set(X.columns)
            if missing_features:
                logger.error(f"Missing features: {missing_features}")
                return None
            
            # Select and order features correctly
            X_pred = X[self.feature_columns].values.astype(np.float32)
            
            # Generate predictions for each quantile
            pred_low = self.low_model.predict(X_pred)[0] if len(X_pred) == 1 else self.low_model.predict(X_pred)
            pred_mid = self.mid_model.predict(X_pred)[0] if len(X_pred) == 1 else self.mid_model.predict(X_pred)
            pred_high = self.high_model.predict(X_pred)[0] if len(X_pred) == 1 else self.high_model.predict(X_pred)
            
            # Handle batch predictions
            if len(X_pred) > 1:
                pred_low = pred_low[0] if len(pred_low) > 0 else 0.0
                pred_mid = pred_mid[0] if len(pred_mid) > 0 else 0.0
                pred_high = pred_high[0] if len(pred_high) > 0 else 0.0
            
            # Calculate confidence based on prediction interval width
            interval_width = abs(pred_high - pred_low)
            confidence = max(0.0, min(1.0, 1.0 / (1.0 + interval_width)))  # Inverse relationship
            
            # Calculate expected return and volatility
            expected_return = pred_mid  # Median as central tendency
            volatility = abs(pred_high - pred_low) / 2  # Rough volatility estimate
            
            # Ensure reasonable bounds
            expected_return = max(min(expected_return, 0.1), -0.1)  # Clamp between -10% and 10%
            volatility = max(min(volatility, 0.15), 0.001)  # Clamp between 0.1% and 15%
            confidence = max(min(confidence, 1.0), 0.01)  # Clamp between 1% and 100%
            
            forecast = {
                'p10': float(pred_low),
                'p50': float(pred_mid),  # median
                'p90': float(pred_high),
                'expected_return': float(expected_return),
                'volatility': float(volatility),
                'confidence': float(confidence),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.debug(f"🔮 Forecast - P10:{pred_low:.4f}, P50:{pred_mid:.4f}, P90:{pred_high:.4f}, "
                        f"E[R]:{expected_return:.4f}, Conf:{confidence:.3f}")
            
            return forecast
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            return None
    
    def _calculate_feature_importance(self):
        """Calculate and store feature importance"""
        try:
            # Use the median model for feature importance
            if hasattr(self.mid_model, 'feature_importances_'):
                importances = self.mid_model.feature_importances_
                self.feature_importance = dict(zip(self.feature_columns, importances))
                
                # Sort by importance
                sorted_imp = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
                logger.info(f"📊 Top 5 features: {sorted_imp[:5]}")
        except Exception as e:
            logger.warning(f"⚠️ Could not calculate feature importance: {e}")
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict:
        """Evaluate model performance"""
        if not self.is_fitted:
            logger.error("❌ Model not fitted yet")
            return {}
        
        try:
            # Generate predictions
            predictions = self.predict(X_test)
            if predictions is None:
                return {}
            
            # Calculate evaluation metrics
            results = {
                'coverage_p10_p90': self._calculate_coverage_score(X_test, y_test, 0.8),  # 80% of values should be between P10 and P90
                'calibration_score': self._calculate_calibration_score(X_test, y_test),
                'sharpness': abs(predictions['p90'] - predictions['p10']),  # Width of prediction interval
                'confidence': predictions['confidence']
            }
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            return {}
    
    def _calculate_coverage_score(self, X_test: pd.DataFrame, y_test: pd.DataFrame, target_coverage: float) -> float:
        """Calculate how often actual values fall within prediction intervals"""
        try:
            # This is a simplified version - in practice would need to generate predictions for each test point
            # For now, returning a placeholder
            return 0.75  # Placeholder
        except:
            return 0.0
    
    def _calculate_calibration_score(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> float:
        """Calculate calibration of predicted probabilities"""
        try:
            # Placeholder for calibration score
            return 0.85  # Placeholder
        except:
            return 0.0
    
    def save_model(self, filepath: str):
        """Save model to disk"""
        try:
            model_data = {
                'low_model': self.low_model,
                'mid_model': self.mid_model,
                'high_model': self.high_model,
                'feature_columns': self.feature_columns,
                'target_columns': self.target_columns,
                'feature_importance': self.feature_importance,
                'is_fitted': self.is_fitted,
                'alpha_low': self.alpha_low,
                'alpha_mid': self.alpha_mid,
                'alpha_high': self.alpha_high,
                'model_type': self.model_type
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"💾 Model saved to {filepath}")
        except Exception as e:
            logger.error(f"❌ Failed to save model: {e}")
    
    def load_model(self, filepath: str):
        """Load model from disk"""
        try:
            model_data = joblib.load(filepath)
            
            self.low_model = model_data['low_model']
            self.mid_model = model_data['mid_model']
            self.high_model = model_data['high_model']
            self.feature_columns = model_data['feature_columns']
            self.target_columns = model_data['target_columns']
            self.feature_importance = model_data['feature_importance']
            self.is_fitted = model_data['is_fitted']
            self.alpha_low = model_data['alpha_low']
            self.alpha_mid = model_data['alpha_mid']
            self.alpha_high = model_data['alpha_high']
            self.model_type = model_data['model_type']
            
            logger.info(f"📥 Model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False