"""
Edge Probability Models for Chloe 0.6
Professional edge detection using regime-aware probability modeling
Instead of predicting price direction, we model P(strategy profitable | regime, features)
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class EdgeOpportunity:
    """Represents an edge opportunity with probability and metrics"""
    symbol: str
    edge_probability: float          # P(profitable trade | regime, features)
    expected_return: float           # Expected return given edge
    confidence_interval: Tuple[float, float]  # [lower, upper] bounds
    regime_context: str              # Current market regime
    feature_importance: Dict[str, float]     # Feature contribution
    time_horizon: str               # SHORT/MEDIUM/LONG term
    risk_metrics: Dict[str, float]  # Risk-adjusted metrics

@dataclass
class EdgeTrainingData:
    """Structure for edge model training data"""
    features: np.ndarray
    labels: np.ndarray              # 1 for profitable, 0 for unprofitable
    regimes: List[str]             # Corresponding regime for each sample
    returns: np.ndarray            # Actual returns for validation
    timestamps: List[datetime]     # Sample timestamps

class RegimeAwareEdgeModel:
    """Main edge probability modeling engine"""
    
    def __init__(self):
        # Ensemble of models for different regimes
        self.models = {
            'STABLE': self._create_regime_model(),
            'TRENDING': self._create_regime_model(), 
            'VOLATILE': self._create_regime_model(),
            'CRISIS': self._create_regime_model()
        }
        
        # Meta-model for combining regime predictions
        self.meta_model = LogisticRegression(random_state=42)
        
        # Feature importance tracking
        self.feature_names = []
        self.training_history = []
        
        # Performance metrics
        self.model_performance = {}
        
        logger.info("Regime-Aware Edge Model initialized")

    def _create_regime_model(self):
        """Create model optimized for specific regime"""
        return GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            subsample=0.8
        )

    def prepare_training_data(self, historical_data: Dict[str, pd.DataFrame], 
                            regime_labels: List[str],
                            min_return_threshold: float = 0.01) -> EdgeTrainingData:
        """Prepare training data for edge probability modeling"""
        try:
            logger.info("Preparing edge training data...")
            
            all_features = []
            all_labels = []
            all_returns = []
            all_regimes = []
            all_timestamps = []
            
            # Process each symbol's data
            for symbol, df in historical_data.items():
                if len(df) < 50:  # Need minimum data
                    continue
                
                # Extract features
                features_df = self._extract_edge_features(df)
                self.feature_names = features_df.columns.tolist()
                
                # Calculate forward returns (prediction target)
                forward_returns = self._calculate_forward_returns(df['close'], periods=5)
                
                # Create labels: 1 if return > threshold, 0 otherwise
                labels = (forward_returns > min_return_threshold).astype(int)
                
                # Align data
                min_len = min(len(features_df), len(labels), len(regime_labels), len(forward_returns))
                
                # Store aligned data
                all_features.append(features_df.iloc[:min_len].values)
                all_labels.append(labels.iloc[:min_len].values)
                all_returns.append(forward_returns.iloc[:min_len].values)
                all_regimes.extend(regime_labels[:min_len])
                all_timestamps.extend(df.index[:min_len].tolist())
            
            # Combine all data
            if all_features:
                features_combined = np.vstack(all_features)
                labels_combined = np.concatenate(all_labels)
                returns_combined = np.concatenate(all_returns)
                
                training_data = EdgeTrainingData(
                    features=features_combined,
                    labels=labels_combined,
                    regimes=all_regimes[:len(labels_combined)],
                    returns=returns_combined,
                    timestamps=all_timestamps[:len(labels_combined)]
                )
                
                logger.info(f"Prepared {len(training_data.labels)} training samples")
                logger.info(f"Positive samples: {sum(training_data.labels)} ({sum(training_data.labels)/len(training_data.labels)*100:.1f}%)")
                
                return training_data
            else:
                raise ValueError("No valid training data prepared")
                
        except Exception as e:
            logger.error(f"Training data preparation failed: {e}")
            raise

    def _extract_edge_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract regime-aware edge features"""
        features = pd.DataFrame(index=df.index)
        
        # Price-based features
        features['price_momentum_5'] = df['close'].pct_change(5)
        features['price_momentum_20'] = df['close'].pct_change(20)
        features['price_zscore'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
        
        # Volatility features
        returns = df['close'].pct_change()
        features['volatility_10'] = returns.rolling(10).std()
        features['volatility_30'] = returns.rolling(30).std()
        features['volatility_ratio'] = features['volatility_10'] / (features['volatility_30'] + 1e-8)
        
        # Volume features
        if 'volume' in df.columns:
            features['volume_momentum'] = df['volume'].pct_change(5)
            features['volume_zscore'] = (df['volume'] - df['volume'].rolling(20).mean()) / df['volume'].rolling(20).std()
            features['price_volume_divergence'] = features['price_momentum_5'] * features['volume_momentum']
        
        # Technical indicators
        features['rsi'] = self._calculate_rsi(df['close'])
        features['macd'] = self._calculate_macd(df['close'])
        features['bollinger_position'] = self._calculate_bollinger_position(df['close'])
        
        # Regime-relative features
        features['regime_volatility_rank'] = self._calculate_regime_rank(features['volatility_10'])
        features['regime_momentum_rank'] = self._calculate_regime_rank(features['price_momentum_5'])
        
        # Market microstructure features
        if 'high' in df.columns and 'low' in df.columns:
            features['high_low_ratio'] = (df['high'] - df['low']) / df['close']
            features['close_location'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
        
        # Lagged features for temporal patterns
        for col in ['price_momentum_5', 'volatility_10', 'rsi']:
            if col in features.columns:
                for lag in [1, 2, 3]:
                    features[f'{col}_lag_{lag}'] = features[col].shift(lag)
        
        # Drop rows with NaN values
        features = features.dropna()
        
        return features

    def _calculate_forward_returns(self, price_series: pd.Series, periods: int = 5) -> pd.Series:
        """Calculate forward returns for edge labeling"""
        return price_series.pct_change(periods).shift(-periods)

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd

    def _calculate_bollinger_position(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Calculate position within Bollinger Bands"""
        sma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper_band = sma + (2 * std)
        lower_band = sma - (2 * std)
        position = (prices - lower_band) / (upper_band - lower_band + 1e-8)
        return position.fillna(0.5)

    def _calculate_regime_rank(self, series: pd.Series) -> pd.Series:
        """Calculate rank relative to recent regime history"""
        rolling_window = 50
        rank = series.rolling(rolling_window).rank(pct=True)
        return rank.fillna(0.5)

    def train_edge_models(self, training_data: EdgeTrainingData, test_size: float = 0.2):
        """Train regime-aware edge models"""
        try:
            logger.info("Training regime-aware edge models...")
            
            # Split data by regime
            regime_data = {}
            for regime in ['STABLE', 'TRENDING', 'VOLATILE', 'CRISIS']:
                mask = np.array(training_data.regimes) == regime
                if np.sum(mask) > 50:  # Need minimum samples
                    regime_data[regime] = {
                        'X': training_data.features[mask],
                        'y': training_data.labels[mask],
                        'returns': training_data.returns[mask]
                    }
            
            # Train individual regime models
            for regime, data in regime_data.items():
                if len(data['X']) > 0:
                    logger.info(f"   Training {regime} regime model ({len(data['X'])} samples)...")
                    
                    # Split for validation
                    X_train, X_val, y_train, y_val = train_test_split(
                        data['X'], data['y'], test_size=0.2, random_state=42
                    )
                    
                    # Train model
                    self.models[regime].fit(X_train, y_train)
                    
                    # Evaluate
                    train_score = self.models[regime].score(X_train, y_train)
                    val_score = self.models[regime].score(X_val, y_val)
                    
                    self.model_performance[regime] = {
                        'train_accuracy': train_score,
                        'validation_accuracy': val_score,
                        'samples': len(data['X'])
                    }
                    
                    logger.info(f"      Train Accuracy: {train_score:.3f}, Val Accuracy: {val_score:.3f}")
            
            # Train meta-model (if we have multiple trained models)
            trained_regimes = [r for r in regime_data.keys() if r in self.model_performance]
            if len(trained_regimes) > 1:
                self._train_meta_model(regime_data, training_data)
            
            logger.info("✅ Edge models training completed")
            
        except Exception as e:
            logger.error(f"Edge model training failed: {e}")
            raise

    def _train_meta_model(self, regime_data: Dict, training_data: EdgeTrainingData):
        """Train meta-model to combine regime predictions"""
        try:
            # Get predictions from all trained models
            meta_features = []
            meta_labels = []
            
            for regime in regime_data:
                if regime in self.model_performance:
                    data = regime_data[regime]
                    predictions = self.models[regime].predict_proba(data['X'])[:, 1]  # Probability of positive class
                    meta_features.extend(predictions.reshape(-1, 1))
                    meta_labels.extend(data['y'])
            
            if len(meta_features) > 0:
                meta_X = np.array(meta_features).reshape(-1, 1)
                meta_y = np.array(meta_labels)
                
                # Train meta-model
                self.meta_model.fit(meta_X, meta_y)
                logger.info("   Meta-model trained for ensemble prediction")
                
        except Exception as e:
            logger.warning(f"Meta-model training failed: {e}")

    def evaluate_edge_opportunity(self, current_features: pd.DataFrame, 
                                regime: str,
                                symbol: str) -> EdgeOpportunity:
        """Evaluate edge probability for current market conditions"""
        try:
            # Ensure features match training data
            feature_vector = self._align_features(current_features)
            
            if feature_vector is None:
                return self._get_default_edge_opportunity(symbol, regime)
            
            # Get regime-specific prediction
            if regime in self.models and regime in self.model_performance:
                model = self.models[regime]
                edge_prob = model.predict_proba(feature_vector)[0][1]  # Probability of profitable trade
                
                # Get feature importance
                feature_importance = self._get_feature_importance(model, regime)
                
                # Calculate expected return and confidence interval
                expected_return = self._estimate_expected_return(feature_vector, regime)
                confidence_interval = self._calculate_confidence_interval(feature_vector, regime)
                
                # Determine time horizon based on regime
                time_horizon = self._determine_time_horizon(regime)
                
                # Calculate risk metrics
                risk_metrics = self._calculate_risk_metrics(edge_prob, regime)
                
                return EdgeOpportunity(
                    symbol=symbol,
                    edge_probability=edge_prob,
                    expected_return=expected_return,
                    confidence_interval=confidence_interval,
                    regime_context=regime,
                    feature_importance=feature_importance,
                    time_horizon=time_horizon,
                    risk_metrics=risk_metrics
                )
            else:
                # Fallback to default evaluation
                return self._get_default_edge_opportunity(symbol, regime)
                
        except Exception as e:
            logger.error(f"Edge evaluation failed for {symbol}: {e}")
            return self._get_default_edge_opportunity(symbol, regime)

    def _align_features(self, current_features: pd.DataFrame) -> Optional[np.ndarray]:
        """Align current features with training feature set"""
        try:
            # Get the last row of features
            if len(current_features) == 0:
                return None
                
            latest_features = current_features.iloc[-1:]
            
            # Ensure all required features are present
            aligned_features = pd.DataFrame(index=latest_features.index)
            
            for feature_name in self.feature_names:
                if feature_name in latest_features.columns:
                    aligned_features[feature_name] = latest_features[feature_name]
                else:
                    aligned_features[feature_name] = 0.0  # Default value
            
            # Handle missing values
            aligned_features = aligned_features.fillna(0.0)
            
            return aligned_features.values
            
        except Exception:
            return None

    def _get_feature_importance(self, model, regime: str) -> Dict[str, float]:
        """Extract feature importance from trained model"""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                return dict(zip(self.feature_names, importances))
            else:
                # For models without built-in feature importance
                return {name: 1.0/len(self.feature_names) for name in self.feature_names}
        except Exception:
            return {name: 0.0 for name in self.feature_names}

    def _estimate_expected_return(self, feature_vector: np.ndarray, regime: str) -> float:
        """Estimate expected return based on features and regime"""
        # Simplified estimation - would use more sophisticated methods in production
        base_return = {
            'STABLE': 0.005,      # 0.5% expected in stable markets
            'TRENDING': 0.015,    # 1.5% expected in trending markets
            'VOLATILE': 0.008,    # 0.8% expected in volatile markets
            'CRISIS': -0.005      # -0.5% expected in crisis markets
        }
        
        regime_return = base_return.get(regime, 0.005)
        
        # Adjust based on edge probability
        # This is a simplified approach - real implementation would be more sophisticated
        return regime_return * 1.5  # Boost return estimate

    def _calculate_confidence_interval(self, feature_vector: np.ndarray, regime: str) -> Tuple[float, float]:
        """Calculate confidence interval for edge probability"""
        # Simplified confidence calculation
        edge_prob = self.models[regime].predict_proba(feature_vector)[0][1]
        
        # Wider intervals for less confident regimes
        regime_uncertainty = {
            'STABLE': 0.1,
            'TRENDING': 0.15,
            'VOLATILE': 0.25,
            'CRISIS': 0.3
        }
        
        uncertainty = regime_uncertainty.get(regime, 0.2)
        
        lower_bound = max(0.0, edge_prob - uncertainty)
        upper_bound = min(1.0, edge_prob + uncertainty)
        
        return (lower_bound, upper_bound)

    def _determine_time_horizon(self, regime: str) -> str:
        """Determine optimal time horizon based on regime"""
        horizon_mapping = {
            'STABLE': 'MEDIUM',      # Medium-term holding
            'TRENDING': 'LONG',      # Long-term trend following
            'VOLATILE': 'SHORT',     # Short-term trading
            'CRISIS': 'SHORT'        # Quick reactions needed
        }
        return horizon_mapping.get(regime, 'MEDIUM')

    def _calculate_risk_metrics(self, edge_prob: float, regime: str) -> Dict[str, float]:
        """Calculate risk-adjusted metrics"""
        # Risk multipliers by regime
        regime_risk = {
            'STABLE': 1.0,
            'TRENDING': 1.2,
            'VOLATILE': 1.8,
            'CRISIS': 2.5
        }
        
        risk_multiplier = regime_risk.get(regime, 1.0)
        
        return {
            'risk_adjusted_probability': edge_prob / risk_multiplier,
            'risk_premium': edge_prob * (1 - risk_multiplier * 0.1),
            'volatility_adjustment': 1.0 / risk_multiplier,
            'regime_risk_factor': risk_multiplier
        }

    def _get_default_edge_opportunity(self, symbol: str, regime: str) -> EdgeOpportunity:
        """Return default edge opportunity when evaluation fails"""
        return EdgeOpportunity(
            symbol=symbol,
            edge_probability=0.5,  # Neutral probability
            expected_return=0.0,
            confidence_interval=(0.3, 0.7),
            regime_context=regime,
            feature_importance={name: 0.0 for name in self.feature_names},
            time_horizon='MEDIUM',
            risk_metrics={'risk_adjusted_probability': 0.5, 'risk_premium': 0.0}
        )

    def get_model_performance_report(self) -> Dict:
        """Get comprehensive model performance report"""
        return {
            'individual_model_performance': self.model_performance,
            'feature_names': self.feature_names,
            'training_samples': len(self.training_history),
            'trained_regimes': [r for r in self.model_performance.keys()]
        }

# Global instance
_edge_model = None

def get_edge_model() -> RegimeAwareEdgeModel:
    """Get singleton edge model instance"""
    global _edge_model
    if _edge_model is None:
        _edge_model = RegimeAwareEdgeModel()
    return _edge_model

def main():
    """Example usage"""
    print("Edge Probability Models ready")
    print("Professional regime-aware edge detection")

if __name__ == "__main__":
    main()