"""
Market Regime Detection for Chloe AI 0.4
Implements HMM-based regime switching to detect market states
Based on Aziz Salimov's industry recommendations
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
from dataclasses import dataclass

# For HMM implementation
try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logging.warning("hmmlearn not available, using simplified regime detection")

logger = logging.getLogger(__name__)

@dataclass
class MarketRegime:
    """Represents a detected market regime"""
    regime_id: int
    name: str  # 'TRENDING', 'MEAN_REVERTING', 'VOLATILE', 'STABLE'
    probability: float
    characteristics: Dict[str, float]
    timestamp: datetime

class RegimeDetector:
    """
    Detects market regimes using Hidden Markov Model approach
    Industry-standard method for identifying market states
    """
    
    def __init__(self, n_regimes: int = 4):
        self.n_regimes = n_regimes
        self.hmm_model = None
        self.is_trained = False
        self.regime_labels = {
            0: 'STABLE',
            1: 'TRENDING', 
            2: 'MEAN_REVERTING',
            3: 'VOLATILE'
        }
        self.transition_matrix = None
        self.emission_probs = None
        
    def prepare_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features specifically for regime detection
        These are different from general trading features
        """
        if df.empty:
            return df
            
        close_prices = df['close'] if 'close' in df.columns else df['Close']
        returns = close_prices.pct_change().dropna()
        
        # Use smaller windows for limited data
        window_size = min(10, len(returns) // 2)  # At least 2 data points per window
        if window_size < 3:
            window_size = 3  # Minimum viable window
            
        regime_features = pd.DataFrame(index=df.index)
        
        # 1. Return statistics (core regime indicators)
        regime_features['return_mean'] = returns.rolling(window_size).mean()
        regime_features['return_std'] = returns.rolling(window_size).std()
        regime_features['return_skew'] = returns.rolling(window_size).skew()
        regime_features['return_kurt'] = returns.rolling(window_size).kurt()
        
        # 2. Volatility clustering indicators
        small_window = max(3, window_size // 2)
        regime_features['volatility_small'] = returns.rolling(small_window).std()
        regime_features['volatility_medium'] = returns.rolling(window_size).std() 
        regime_features['volatility_large'] = returns.rolling(min(window_size * 2, len(returns))).std()
        regime_features['vol_ratio_short'] = regime_features['volatility_small'] / regime_features['volatility_medium']
        regime_features['vol_ratio_long'] = regime_features['volatility_medium'] / regime_features['volatility_large']
        
        # 3. Trend strength indicators
        trend_window = min(10, len(close_prices) // 3)
        regime_features['trend_strength'] = abs(close_prices.diff(trend_window)) / close_prices
        regime_features['price_position'] = (close_prices - close_prices.rolling(window_size).min()) / (close_prices.rolling(window_size).max() - close_prices.rolling(window_size).min())
        
        # 4. Autocorrelation (mean reversion tendency) - optimized
        regime_features['autocorr_1'] = returns.rolling(window_size).corr(returns.shift(1)).fillna(0)
        regime_features['autocorr_2'] = returns.rolling(window_size).corr(returns.shift(2)).fillna(0)
        
        # 5. Volume-based regime indicators (if available)
        if 'volume' in df.columns or 'Volume' in df.columns:
            volume = df['volume'] if 'volume' in df.columns else df['Volume']
            regime_features['volume_volatility'] = volume.rolling(window_size).std() / volume.rolling(window_size).mean()
            regime_features['volume_trend'] = volume.diff(small_window) / volume.rolling(small_window).mean()
        
        # Drop NaN values and normalize
        regime_features = regime_features.dropna()
        
        # Normalize features for HMM
        for col in regime_features.columns:
            if regime_features[col].std() > 0:
                regime_features[col] = (regime_features[col] - regime_features[col].mean()) / regime_features[col].std()
            else:
                regime_features[col] = 0
                
        return regime_features
    
    def train_hmm(self, df: pd.DataFrame) -> bool:
        """
        Train HMM model for regime detection
        """
        if not HMM_AVAILABLE:
            logger.warning("HMM not available, using rule-based detection")
            return self._train_rule_based(df)
            
        try:
            # Prepare features
            features = self.prepare_regime_features(df)
            if len(features) < 50:  # Need minimum data
                logger.warning("Insufficient data for HMM training")
                return False
            
            # Prepare observation sequences
            X = features.values
            lengths = [len(X)]  # Single sequence
            
            # Initialize and train HMM
            self.hmm_model = hmm.GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="full",
                n_iter=100,
                random_state=42
            )
            
            self.hmm_model.fit(X, lengths)
            self.is_trained = True
            
            # Store transition matrix
            self.transition_matrix = self.hmm_model.transmat_
            
            logger.info(f"✅ HMM regime detector trained with {self.n_regimes} regimes")
            logger.info(f"Transition matrix shape: {self.transition_matrix.shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ HMM training failed: {e}")
            return self._train_rule_based(df)
    
    def _train_rule_based(self, df: pd.DataFrame) -> bool:
        """
        Fallback rule-based regime detection for when HMM is not available
        """
        try:
            features = self.prepare_regime_features(df)
            if len(features) < 3:  # Even lower minimum requirement
                return False
                
            # Simple clustering approach with flexible column access
            vol_col = 'volatility_medium' if 'volatility_medium' in features.columns else 'return_std'
            trend_col = 'trend_strength' if 'trend_strength' in features.columns else 'return_mean'
            auto_col = 'autocorr_1' if 'autocorr_1' in features.columns else 'return_skew'
            
            volatility_measure = features[vol_col].iloc[-min(10, len(features)):].mean()
            trend_measure = abs(features[trend_col].iloc[-min(10, len(features)):].mean())
            autocorr_measure = features[auto_col].iloc[-min(10, len(features)):].mean()
            
            # Store baseline statistics for online detection
            self.baseline_volatility = volatility_measure
            self.baseline_trend = trend_measure
            self.baseline_autocorr = autocorr_measure
            
            self.is_trained = True
            logger.info("✅ Rule-based regime detector initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Rule-based training failed: {e}")
            return False
    
    def detect_current_regime(self, df: pd.DataFrame) -> MarketRegime:
        """
        Detect current market regime
        """
        # Handle untrained case gracefully
        if not self.is_trained:
            # Try to train quickly
            if not self._train_rule_based(df):
                # Return default regime
                return MarketRegime(
                    regime_id=0,
                    name='STABLE',
                    probability=0.5,
                    characteristics={'fallback': True},
                    timestamp=datetime.now()
                )
        
        try:
            # Get latest features
            features = self.prepare_regime_features(df)
            if len(features) == 0:
                return MarketRegime(
                    regime_id=0,
                    name='STABLE',
                    probability=0.5,
                    characteristics={'insufficient_data': True},
                    timestamp=datetime.now()
                )
            
            latest_features = features.iloc[-1:].values
            
            if HMM_AVAILABLE and self.hmm_model:
                # HMM-based detection
                regime_id = self.hmm_model.predict(latest_features)[0]
                regime_prob = self.hmm_model.predict_proba(latest_features)[0]
                max_prob = np.max(regime_prob)
            else:
                # Rule-based detection
                regime_id, max_prob = self._detect_rule_based(latest_features[0], features)
            
            regime_name = self.regime_labels.get(regime_id, f'REGIME_{regime_id}')
            
            # Extract regime characteristics with flexible column access
            vol_col = 'volatility_medium' if 'volatility_medium' in features.columns else 'return_std'
            trend_col = 'trend_strength' if 'trend_strength' in features.columns else 'return_mean'
            auto_col = 'autocorr_1' if 'autocorr_1' in features.columns else 'return_skew'
            
            characteristics = {
                'volatility': features[vol_col].iloc[-1] if vol_col in features.columns else 0,
                'trend_strength': features[trend_col].iloc[-1] if trend_col in features.columns else 0,
                'mean_reversion_tendency': features[auto_col].iloc[-1] if auto_col in features.columns else 0,
                'current_return': df['close'].pct_change().iloc[-1] if 'close' in df.columns else 0
            }
            
            return MarketRegime(
                regime_id=regime_id,
                name=regime_name,
                probability=max_prob,
                characteristics=characteristics,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"❌ Regime detection failed: {e}")
            return MarketRegime(
                regime_id=0,
                name='STABLE',
                probability=0.5,
                characteristics={'error': str(e)},
                timestamp=datetime.now()
            )
    
    def _detect_rule_based(self, features: np.ndarray, all_features: pd.DataFrame) -> Tuple[int, float]:
        """
        Rule-based regime detection logic
        """
        # Handle dynamic column names
        vol_ratio_col = 'vol_ratio_short'
        trend_col = 'trend_strength' 
        autocorr_col = 'autocorr_1'
        
        # Fallback column names
        if vol_ratio_col not in all_features.columns:
            vol_ratio_col = 'vol_ratio_long' if 'vol_ratio_long' in all_features.columns else 'volatility_small'
        if trend_col not in all_features.columns:
            trend_col = 'trend_strength_20' if 'trend_strength_20' in all_features.columns else 'return_mean'
        if autocorr_col not in all_features.columns:
            autocorr_col = 'autocorr_2' if 'autocorr_2' in all_features.columns else 'return_skew'
            
        vol_ratio_idx = all_features.columns.get_loc(vol_ratio_col)
        trend_idx = all_features.columns.get_loc(trend_col) 
        autocorr_idx = all_features.columns.get_loc(autocorr_col)
        
        vol_ratio = features[vol_ratio_idx]
        trend_strength = features[trend_idx]
        autocorr = features[autocorr_idx]
        
        # Simple heuristic-based classification
        if abs(trend_strength) > 0.01 and vol_ratio > 1.1:  # Lowered thresholds
            regime_id = 1  # TRENDING
            confidence = min(0.9, abs(trend_strength) * 10)  # Reduced multiplier
        elif autocorr < -0.05:  # Lowered threshold for mean reversion
            regime_id = 2  # MEAN_REVERTING
            confidence = min(0.8, abs(autocorr) * 3)  # Reduced multiplier
        elif vol_ratio > 1.3:  # Lowered volatility threshold
            regime_id = 3  # VOLATILE
            confidence = min(0.85, vol_ratio / 1.5)  # Adjusted ratio
        else:
            regime_id = 0  # STABLE
            confidence = 0.6  # Lower base confidence
            
        return regime_id, confidence
    
    def get_regime_history(self, df: pd.DataFrame, window: int = 100) -> List[MarketRegime]:
        """
        Get regime history for analysis
        """
        if not self.is_trained:
            return []
            
        try:
            features = self.prepare_regime_features(df)
            if len(features) < window:
                window = len(features)
                
            history = []
            for i in range(-window, 0):
                slice_df = df.iloc[:len(df)+i+1] if i < -1 else df
                regime = self.detect_current_regime(slice_df)
                history.append(regime)
                
            return history
            
        except Exception as e:
            logger.error(f"❌ Regime history failed: {e}")
            return []

class RegimeAwareFeatureEngineer:
    """
    Feature engineering that adapts to detected market regimes
    Different features are relevant in different market conditions
    """
    
    def __init__(self):
        self.regime_detector = RegimeDetector()
        
    def adapt_features_to_regime(self, df: pd.DataFrame, regime: MarketRegime) -> pd.DataFrame:
        """
        Adapt feature engineering based on current market regime
        """
        # Base features (always calculated)
        adapted_df = df.copy()
        
        if regime.name == 'TRENDING':
            # Emphasize momentum features
            adapted_df = self._add_momentum_features(adapted_df)
            
        elif regime.name == 'MEAN_REVERTING':
            # Emphasize mean-reversion features
            adapted_df = self._add_mean_reversion_features(adapted_df)
            
        elif regime.name == 'VOLATILE':
            # Emphasize volatility and risk features
            adapted_df = self._add_volatility_features(adapted_df)
            
        elif regime.name == 'STABLE':
            # Balanced feature set
            adapted_df = self._add_balanced_features(adapted_df)
            
        return adapted_df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features optimized for trending markets"""
        close_prices = df['close'] if 'close' in df.columns else df['Close']
        
        # Strong momentum indicators
        df['momentum_50'] = close_prices.pct_change(periods=50)
        df['price_trend_20'] = close_prices.diff(20) / close_prices
        df['trend_continuation'] = (close_prices > close_prices.rolling(20).mean()).astype(int)
        
        return df
    
    def _add_mean_reversion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features optimized for mean-reverting markets"""
        close_prices = df['close'] if 'close' in df.columns else df['Close']
        
        # Mean reversion indicators
        df['z_score_20'] = (close_prices - close_prices.rolling(20).mean()) / close_prices.rolling(20).std()
        df['distance_to_mean'] = abs(close_prices - close_prices.rolling(20).mean()) / close_prices
        df['reversion_signal'] = (df['z_score_20'] < -1.5).astype(int) - (df['z_score_20'] > 1.5).astype(int)
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features for volatile market conditions"""
        close_prices = df['close'] if 'close' in df.columns else df['Close']
        returns = close_prices.pct_change()
        
        # Volatility regime features
        df['volatility_regime'] = returns.rolling(20).std() / returns.rolling(200).std()
        df['volatility_spike'] = (returns.rolling(5).std() > returns.rolling(20).std() * 2).astype(int)
        df['risk_adjusted_return'] = returns / (returns.rolling(20).std() + 1e-8)
        
        return df
    
    def _add_balanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add balanced feature set for stable markets"""
        # Use existing feature calculator for balanced approach
        from feature_store.feature_calculator import get_feature_calculator
        calculator = get_feature_calculator()
        return calculator.calculate_all_features(df)

# Global regime detector instance
regime_detector = None

def get_regime_detector() -> RegimeDetector:
    """Get singleton regime detector instance"""
    global regime_detector
    if regime_detector is None:
        regime_detector = RegimeDetector()
    return regime_detector

# Integration with existing pipeline
def enhance_pipeline_with_regime_detection(pipeline_data: pd.DataFrame) -> Tuple[pd.DataFrame, MarketRegime]:
    """
    Enhance existing pipeline data with regime awareness
    """
    detector = get_regime_detector()
    
    # Train if not already trained
    if not detector.is_trained:
        detector.train_hmm(pipeline_data)
    
    # Detect current regime
    current_regime = detector.detect_current_regime(pipeline_data)
    
    # Adapt features based on regime
    feature_engineer = RegimeAwareFeatureEngineer()
    adapted_data = feature_engineer.adapt_features_to_regime(pipeline_data, current_regime)
    
    return adapted_data, current_regime