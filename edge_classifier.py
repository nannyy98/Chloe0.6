"""
Edge Classification Model for Chloe AI 0.4
Implements edge detection focused on identifying genuine trading opportunities
Rather than predicting prices, this identifies probability of having a statistical edge
Based on Lopez de Prado's meta-labeling approach and industry best practices
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
import lightgbm as lgb

logger = logging.getLogger(__name__)

@dataclass
class EdgeFeatures:
    """Features used for edge classification"""
    # Primary edge indicators
    regime_edge_score: float           # Market regime favorability for this strategy
    volatility_edge: float             # Volatility-based edge opportunity
    momentum_alignment: float          # Alignment of multiple momentum indicators
    mean_reversion_strength: float     # Strength of mean-reversion signals
    volume_confirmation: float         # Volume supporting the edge
    
    # Risk-adjusted metrics
    risk_reward_ratio: float           # Risk/reward of the opportunity
    position_sizing_score: float       # Quality of position sizing
    drawdown_impact: float             # Potential drawdown impact
    
    # Market context
    liquidity_score: float             # Market liquidity conditions
    correlation_risk: float            # Correlation with other positions
    market_stress_indicator: float     # Overall market stress level
    
    # Temporal features
    time_decay_factor: float           # Edge decay over time
    seasonality_adjustment: float      # Seasonal edge patterns
    regime_duration: float             # How long current regime has persisted

@dataclass 
class EdgeLabel:
    """Meta-label for edge classification"""
    has_edge: bool                     # Binary edge classification
    edge_strength: float               # Continuous edge strength (0-1)
    edge_probability: float            # Model's confidence in edge existence
    expected_return: float             # Expected return conditional on edge
    holding_period: int                # Optimal holding period in days
    stop_loss_level: float             # Recommended stop loss level
    take_profit_level: float           # Recommended take profit level

class EdgeClassifier:
    """
    Edge classification model that identifies genuine trading opportunities
    Focuses on detecting statistical edges rather than price direction
    """
    
    def __init__(self, model_type: str = 'ensemble'):
        self.model_type = model_type
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        self.cross_validation_scores = {}
        self.feature_importance = {}
        
        # Initialize models
        self._initialize_models()
        
        logger.info(f"🎯 Edge Classifier initialized ({model_type} approach)")
    
    def _initialize_models(self):
        """Initialize classification models"""
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
                'gbm': GradientBoostingClassifier(
                    n_estimators=150,
                    max_depth=10,
                    learning_rate=0.05,
                    subsample=0.8,
                    random_state=42
                ),
                'lgbm': lgb.LGBMClassifier(
                    n_estimators=200,
                    max_depth=12,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1
                )
            }
        elif self.model_type == 'lightgbm':
            self.models = {
                'main': lgb.LGBMClassifier(
                    n_estimators=300,
                    max_depth=15,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1
                )
            }
        else:  # Random Forest default
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
    
    def prepare_edge_features(self, market_data: pd.DataFrame, 
                            regime_info: Dict = None,
                            forecast_data: Dict = None) -> pd.DataFrame:
        """
        Prepare features specifically designed for edge detection
        These are different from general trading features - focused on edge identification
        """
        if market_data.empty:
            return pd.DataFrame()
        
        close_prices = market_data['close'] if 'close' in market_data.columns else market_data['Close']
        high_prices = market_data['high'] if 'high' in market_data.columns else market_data['High']
        low_prices = market_data['low'] if 'low' in market_data.columns else market_data['Low']
        volume = market_data['volume'] if 'volume' in market_data.columns else market_data['Volume']
        
        edge_features = pd.DataFrame(index=market_data.index)
        
        # 1. Primary Edge Indicators
        logger.debug("Calculating primary edge indicators...")
        
        # Regime edge score (how favorable current regime is for trading)
        if regime_info:
            regime_favorability = self._calculate_regime_edge_score(regime_info)
            edge_features['regime_edge_score'] = regime_favorability
        else:
            edge_features['regime_edge_score'] = 0.5  # Neutral when no regime info
        
        # Volatility edge opportunities
        returns = close_prices.pct_change()
        edge_features['volatility_edge'] = self._calculate_volatility_edge(returns)
        
        # Momentum alignment across multiple timeframes
        edge_features['momentum_alignment'] = self._calculate_momentum_alignment(close_prices)
        
        # Mean reversion strength
        edge_features['mean_reversion_strength'] = self._calculate_mean_reversion_strength(close_prices)
        
        # Volume confirmation of signals
        edge_features['volume_confirmation'] = self._calculate_volume_confirmation(volume, close_prices)
        
        # 2. Risk-Adjusted Metrics
        logger.debug("Calculating risk-adjusted metrics...")
        
        # Risk/reward ratio opportunities
        edge_features['risk_reward_ratio'] = self._calculate_risk_reward_opportunities(high_prices, low_prices, close_prices)
        
        # Position sizing quality score
        edge_features['position_sizing_score'] = self._calculate_position_sizing_quality(returns)
        
        # Drawdown impact assessment
        edge_features['drawdown_impact'] = self._calculate_drawdown_impact(returns)
        
        # 3. Market Context Features
        logger.debug("Calculating market context features...")
        
        # Liquidity scoring
        edge_features['liquidity_score'] = self._calculate_liquidity_score(volume, close_prices)
        
        # Correlation risk assessment
        edge_features['correlation_risk'] = self._calculate_correlation_risk(close_prices)
        
        # Market stress indicators
        edge_features['market_stress_indicator'] = self._calculate_market_stress(returns, volume)
        
        # 4. Temporal Features
        logger.debug("Calculating temporal features...")
        
        # Time decay of edges
        edge_features['time_decay_factor'] = self._calculate_time_decay(len(market_data))
        
        # Seasonality adjustments
        edge_features['seasonality_adjustment'] = self._calculate_seasonality_adjustment(market_data.index)
        
        # Regime duration effects
        edge_features['regime_duration'] = self._calculate_regime_duration(regime_info)
        
        # Drop rows with insufficient data
        edge_features = edge_features.dropna()
        
        # If too much data was dropped, try filling NaN values
        if len(edge_features) < len(market_data) * 0.5:  # Less than 50% remaining
            logger.warning(f"⚠️ Too much data dropped ({len(edge_features)}/{len(market_data)}), filling NaN values")
            edge_features = edge_features.ffill().bfill().fillna(0)
        
        logger.info(f"✅ Prepared {len(edge_features.columns)} edge features for {len(edge_features)} samples")
        self.feature_names = list(edge_features.columns)
        
        return edge_features
    
    def _calculate_regime_edge_score(self, regime_info: Dict) -> float:
        """Calculate how favorable the current regime is for edge exploitation"""
        regime_name = regime_info.get('name', 'STABLE')
        regime_confidence = regime_info.get('probability', 0.5)
        
        # Regime favorability mapping
        regime_scores = {
            'TRENDING': 0.8,      # Best for momentum strategies
            'MEAN_REVERTING': 0.7, # Good for mean-reversion strategies  
            'STABLE': 0.4,        # Moderate opportunities
            'VOLATILE': 0.6       # Good for volatility strategies
        }
        
        base_score = regime_scores.get(regime_name, 0.5)
        # Adjust by regime confidence
        return base_score * regime_confidence
    
    def _calculate_volatility_edge(self, returns: pd.Series) -> pd.Series:
        """Identify volatility-based edge opportunities"""
        # Look for volatility clustering and regime changes
        vol_20 = returns.rolling(20).std()
        vol_60 = returns.rolling(60).std()
        
        # Volatility expansion opportunities
        vol_expansion = vol_20 / vol_60
        vol_zscore = (vol_20 - vol_20.rolling(252).mean()) / vol_20.rolling(252).std()
        
        # Combine into edge score (-1 to 1)
        edge_score = (vol_expansion - 1) * 0.5 + vol_zscore * 0.3
        return np.clip(edge_score, -1, 1)
    
    def _calculate_momentum_alignment(self, prices: pd.Series) -> pd.Series:
        """Calculate alignment of momentum across multiple timeframes"""
        # Multiple timeframe momentum
        mom_5 = prices.pct_change(5)
        mom_10 = prices.pct_change(10)
        mom_20 = prices.pct_change(20)
        
        # Alignment score (how many agree on direction)
        alignment = ((mom_5 > 0) & (mom_10 > 0) & (mom_20 > 0)).astype(int) * 2 - 1
        return alignment.astype(float)
    
    def _calculate_mean_reversion_strength(self, prices: pd.Series) -> pd.Series:
        """Calculate strength of mean-reversion opportunities"""
        # Z-score based approach
        ma_20 = prices.rolling(20).mean()
        std_20 = prices.rolling(20).std()
        z_score = (prices - ma_20) / std_20
        
        # Mean reversion strength (higher when |z_score| > 2)
        mr_strength = np.where(abs(z_score) > 2, abs(z_score) - 2, 0)
        return np.clip(mr_strength / 2, 0, 1)  # Normalize to 0-1
    
    def _calculate_volume_confirmation(self, volume: pd.Series, prices: pd.Series) -> pd.Series:
        """Calculate volume confirmation of price moves"""
        price_direction = prices.diff().apply(lambda x: 1 if x > 0 else -1)
        volume_ma = volume.rolling(20).mean()
        volume_ratio = volume / volume_ma
        
        # Confirmation when volume supports price direction
        confirmation = np.where(
            (price_direction > 0) & (volume_ratio > 1.2), 1.0,
            np.where((price_direction < 0) & (volume_ratio > 1.2), 1.0, 0.0)
        )
        return confirmation
    
    def _calculate_risk_reward_opportunities(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate risk/reward ratio opportunities"""
        # Support/resistance based risk/reward
        support = low.rolling(20).min()
        resistance = high.rolling(20).max()
        
        # Risk as distance to support, reward as distance to resistance
        risk = close - support
        reward = resistance - close
        
        # Avoid division by zero
        rr_ratio = np.where(risk > 0, reward / risk, 1.0)
        return np.clip(rr_ratio / 5, 0, 1)  # Normalize (cap at 5:1 ratio)
    
    def _calculate_position_sizing_quality(self, returns: pd.Series) -> pd.Series:
        """Score quality of potential position sizing"""
        # Based on recent volatility and trend strength
        vol_20 = returns.rolling(20).std()
        trend_strength = abs(returns.rolling(20).mean() / vol_20)
        
        # Higher scores for moderate volatility with strong trends
        quality_score = np.where(
            (vol_20 > 0.01) & (vol_20 < 0.05) & (trend_strength > 0.5),
            1.0, 0.3
        )
        return quality_score
    
    def _calculate_drawdown_impact(self, returns: pd.Series) -> pd.Series:
        """Estimate potential drawdown impact"""
        # Rolling maximum drawdown over different periods
        rolling_dd = returns.rolling(20).apply(lambda x: (x + 1).cumprod().min() - 1)
        return abs(rolling_dd)  # Return as positive values
    
    def _calculate_liquidity_score(self, volume: pd.Series, prices: pd.Series) -> pd.Series:
        """Score market liquidity conditions"""
        dollar_volume = volume * prices
        vol_percentile = dollar_volume.rolling(252).rank(pct=True)
        
        # Higher scores for better liquidity
        return vol_percentile
    
    def _calculate_correlation_risk(self, prices: pd.Series) -> pd.Series:
        """Assess correlation risk with broader market"""
        # Simplified: correlation with price changes
        market_returns = prices.pct_change()
        correlation = market_returns.rolling(20).corr(market_returns.shift(1)).fillna(0)
        return abs(correlation)  # Higher correlation = higher risk
    
    def _calculate_market_stress(self, returns: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate overall market stress indicators"""
        # Combination of volatility, volume spikes, and return extremes
        vol_stress = returns.rolling(20).std() / returns.rolling(200).std()
        volume_stress = volume / volume.rolling(50).mean()
        return_stress = abs(returns) / returns.rolling(20).std()
        
        stress_score = (vol_stress + volume_stress + return_stress) / 3
        return np.clip(stress_score, 0, 1)
    
    def _calculate_time_decay(self, data_length: int) -> pd.Series:
        """Calculate time decay factor for edges"""
        # Edges become less reliable over time
        decay_factor = np.exp(-np.arange(data_length) / 100)  # 100-day half-life
        return pd.Series(decay_factor, index=range(data_length))
    
    def _calculate_seasonality_adjustment(self, timestamps) -> pd.Series:
        """Calculate seasonality adjustments"""
        if hasattr(timestamps, 'dayofweek'):
            # Weekly seasonality
            day_effect = np.sin(2 * np.pi * timestamps.dayofweek / 7)
            return pd.Series(day_effect, index=timestamps)
        else:
            return pd.Series(0.5, index=range(len(timestamps)))  # Neutral
    
    def _calculate_regime_duration(self, regime_info: Dict) -> float:
        """Calculate how long current regime has persisted"""
        # Simplified implementation
        return regime_info.get('duration', 1.0) if regime_info else 1.0
    
    def create_meta_labels(self, market_data: pd.DataFrame, 
                          holding_period: int = 5) -> pd.Series:
        """
        Create meta-labels for edge classification
        This is the core of Lopez de Prado's approach - labeling based on future performance
        """
        close_prices = market_data['close'] if 'close' in market_data.columns else market_data['Close']
        
        # Calculate future returns over holding period
        future_returns = close_prices.shift(-holding_period) / close_prices - 1
        
        # Define edge threshold (e.g., 2% return over holding period)
        edge_threshold = 0.02
        
        # Binary labels: 1 if future return exceeds threshold, 0 otherwise
        meta_labels = (future_returns > edge_threshold).astype(int)
        
        # Remove last holding_period rows (no future data)
        meta_labels = meta_labels.iloc[:-holding_period]
        
        logger.info(f"✅ Created meta-labels: {meta_labels.sum()} positive edges out of {len(meta_labels)} samples")
        return meta_labels
    
    def train(self, features: pd.DataFrame, labels: pd.Series) -> Dict:
        """
        Train edge classification models
        """
        try:
            logger.info("🚀 Starting edge classifier training...")
            
            # Prepare data - align indices properly
            common_index = features.index.intersection(labels.index)
            if len(common_index) == 0:
                raise ValueError("No common indices between features and labels")
            
            X = features.loc[common_index]
            y = labels.loc[common_index]
            
            logger.info(f"Training on {len(X)} samples with {len(X.columns)} features")
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            cv_results = {}
            
            for model_name, model in self.models.items():
                logger.info(f"Training {model_name} model...")
                
                # Cross-validation scores
                cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='roc_auc')
                cv_results[model_name] = {
                    'mean_cv_score': cv_scores.mean(),
                    'std_cv_score': cv_scores.std(),
                    'individual_scores': cv_scores.tolist()
                }
                
                # Train final model on all data
                model.fit(X_scaled, y)
                logger.info(f"✅ {model_name} model trained (CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f})")
            
            self.cross_validation_scores = cv_results
            self.is_trained = True
            
            # Feature importance (from best model)
            best_model_name = max(cv_results.keys(), key=lambda k: cv_results[k]['mean_cv_score'])
            if hasattr(self.models[best_model_name], 'feature_importances_'):
                self.feature_importance = dict(zip(self.feature_names, self.models[best_model_name].feature_importances_))
            
            logger.info("✅ Edge classifier training completed")
            return cv_results
            
        except Exception as e:
            logger.error(f"❌ Edge classifier training failed: {e}")
            raise
    
    def predict_edge(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Predict edge probabilities for given features
        """
        if not self.is_trained:
            logger.warning("Edge classifier not trained yet")
            return pd.DataFrame()
        
        try:
            # Scale features
            X_scaled = self.scaler.transform(features)
            
            # Get predictions from all models
            predictions = pd.DataFrame(index=features.index)
            
            for model_name, model in self.models.items():
                # Probability predictions
                proba = model.predict_proba(X_scaled)[:, 1]  # Probability of positive class
                predictions[f'{model_name}_prob'] = proba
                
                # Binary predictions (using 0.5 threshold)
                predictions[f'{model_name}_pred'] = (proba > 0.5).astype(int)
            
            # Ensemble prediction (average probabilities)
            prob_columns = [col for col in predictions.columns if col.endswith('_prob')]
            predictions['ensemble_prob'] = predictions[prob_columns].mean(axis=1)
            predictions['ensemble_pred'] = (predictions['ensemble_prob'] > 0.5).astype(int)
            
            logger.info(f"✅ Generated edge predictions for {len(predictions)} samples")
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Edge prediction failed: {e}")
            return pd.DataFrame()
    
    def evaluate_model(self, features: pd.DataFrame, true_labels: pd.Series) -> Dict:
        """
        Evaluate model performance
        """
        if not self.is_trained:
            return {}
        
        try:
            predictions = self.predict_edge(features)
            if predictions.empty:
                return {}
            
            # Evaluation metrics
            evaluation = {}
            
            # For ensemble predictions
            y_pred = predictions['ensemble_pred']
            y_proba = predictions['ensemble_prob']
            
            # Basic metrics
            evaluation['auc_roc'] = roc_auc_score(true_labels, y_proba)
            evaluation['classification_report'] = classification_report(true_labels, y_pred, output_dict=True)
            
            # Precision-recall curve
            precisions, recalls, thresholds = precision_recall_curve(true_labels, y_proba)
            evaluation['pr_curve'] = {
                'precisions': precisions.tolist(),
                'recalls': recalls.tolist(),
                'thresholds': thresholds.tolist()
            }
            
            logger.info(f"✅ Model evaluation completed (AUC-ROC: {evaluation['auc_roc']:.3f})")
            return evaluation
            
        except Exception as e:
            logger.error(f"❌ Model evaluation failed: {e}")
            return {}

# Global edge classifier instance
edge_classifier = None

def get_edge_classifier(model_type: str = 'ensemble') -> EdgeClassifier:
    """Get singleton edge classifier instance"""
    global edge_classifier
    if edge_classifier is None:
        edge_classifier = EdgeClassifier(model_type)
    return edge_classifier