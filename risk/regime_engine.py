"""
Regime Detection Engine for Adaptive Institutional AI Trader
Implements Hidden Markov Model for market regime detection
"""

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import warnings

# Suppress warnings for HMM
warnings.filterwarnings("ignore", category=DeprecationWarning)

logger = logging.getLogger(__name__)

class RegimeState(Enum):
    """Market regime states"""
    TREND = "TREND"
    MEAN_REVERT = "MEAN_REVERT"
    CRISIS = "CRISIS"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_LIQUIDITY = "LOW_LIQUIDITY"
    NORMAL = "NORMAL"

@dataclass
class RegimeEvent:
    """Event for regime detection"""
    symbol: str
    regime: RegimeState
    probability: float
    confidence: float
    features: Dict
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class RegimeDetectionEngine:
    """Engine for detecting market regimes using HMM and statistical methods"""
    
    def __init__(self):
        self.regimes = {
            'TREND': 0,
            'MEAN_REVERT': 1, 
            'CRISIS': 2,
            'LOW_VOL': 3,
            'HIGH_VOL': 4,
            'LOW_LIQ': 5,
            'NORMAL': 6
        }
        
        # HMM parameters (will be initialized when needed)
        self.hmm_model = None
        self.n_states = 4  # 4 regime states
        self.n_features = 5  # number of features to observe
        
        # Regime characteristics
        self.regime_thresholds = {
            'volatility': {
                'low': 0.01,    # 1% daily vol
                'high': 0.03    # 3% daily vol
            },
            'correlation': {
                'low': 0.2,
                'high': 0.7
            },
            'trend_strength': {
                'weak': 0.1,
                'strong': 0.3
            },
            'liquidity': {
                'low': 0.1,     # Low volume relative to MA
                'normal': 0.5
            }
        }
        
        # State transition probabilities (placeholder, would be learned)
        self.transition_matrix = np.array([
            [0.8, 0.1, 0.05, 0.05],  # From NORMAL
            [0.1, 0.7, 0.1, 0.1],   # From TREND
            [0.2, 0.1, 0.6, 0.1],   # From MEAN_REVERT
            [0.05, 0.1, 0.1, 0.75]  # From CRISIS/VOLATILE
        ])
        
        # Emission probabilities (placeholder)
        self.emission_probs = np.random.rand(4, 5)  # states x features
        self.emission_probs = self.emission_probs / self.emission_probs.sum(axis=1, keepdims=True)
        
        # Regime state tracking
        self.current_regime = RegimeState.NORMAL
        self.regime_history = []
        self.confidence_history = []
        
        logger.info("🔄 Regime Detection Engine initialized")
    
    def detect_regime(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> Tuple[RegimeState, float, Dict]:
        """
        Detect current market regime based on features
        
        Args:
            df: DataFrame with market data
            symbol: Trading symbol
            
        Returns:
            Tuple of (regime, confidence, features)
        """
        try:
            # Extract features for regime detection
            features = self._extract_regime_features(df)
            
            if not features:
                logger.warning(f"No valid features for {symbol} regime detection")
                return RegimeState.NORMAL, 0.5, {}
            
            # Apply statistical regime detection
            regime, confidence = self._statistical_regime_detection(features)
            
            # Update internal state
            self.current_regime = regime
            self.regime_history.append((datetime.now(), regime, confidence))
            
            logger.debug(f"🎯 Regime detected for {symbol}: {regime.value} (conf: {confidence:.2f})")
            
            return regime, confidence, features
            
        except Exception as e:
            logger.error(f"❌ Regime detection failed for {symbol}: {e}")
            return RegimeState.NORMAL, 0.5, {}
    
    def _extract_regime_features(self, df: pd.DataFrame) -> Dict:
        """Extract features for regime detection"""
        if len(df) < 50:  # Need sufficient data
            return {}
        
        features = {}
        
        # Volatility features
        returns = df['close'].pct_change().dropna()
        features['volatility'] = returns.std() * np.sqrt(252)  # Annualized vol
        features['volatility_regime'] = (
            returns.rolling(20).std().iloc[-1] / 
            returns.rolling(252).std().mean()
        ) if len(returns) > 252 else 1.0
        
        # Trend features
        features['trend_strength'] = abs(
            df['close'].iloc[-1] / df['close'].rolling(20).mean().iloc[-1] - 1
        )
        
        # Momentum vs mean reversion indicators
        features['momentum_indicator'] = (
            df['close'].iloc[-1] - df['close'].rolling(20).mean().iloc[-1]
        ) / df['close'].rolling(20).std().iloc[-1]
        
        # Volume features
        if 'volume' in df.columns:
            features['volume_regime'] = (
                df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
            )
        else:
            features['volume_regime'] = 1.0  # Neutral
        
        # Correlation features (if multiple assets)
        features['price_acceleration'] = (
            returns.iloc[-1] - returns.iloc[-5]
        ) if len(returns) >= 5 else 0.0
        
        # Skewness and kurtosis for crisis detection
        if len(returns) >= 30:
            features['returns_skewness'] = returns.tail(30).skew()
            features['returns_kurtosis'] = returns.tail(30).kurtosis()
        else:
            features['returns_skewness'] = 0.0
            features['returns_kurtosis'] = 0.0
        
        # Liquidity proxy
        features['liquidity_proxy'] = features['volume_regime'] * features['volatility']
        
        return features
    
    def _statistical_regime_detection(self, features: Dict) -> Tuple[RegimeState, float]:
        """Statistical method to detect regime"""
        # Calculate regime scores based on features
        scores = {}
        
        # Crisis regime: High volatility + negative skew + high kurtosis
        crisis_score = (
            min(features.get('volatility', 0) / 0.03, 1.0) * 0.4 +
            min(max(features.get('returns_skewness', 0), 0) / 0.5, 1.0) * 0.3 +
            min(features.get('returns_kurtosis', 0) / 5.0, 1.0) * 0.3
        )
        
        # Trend regime: Strong trend + momentum
        trend_score = (
            min(features.get('trend_strength', 0) / 0.1, 1.0) * 0.5 +
            min(abs(features.get('momentum_indicator', 0)) / 2.0, 1.0) * 0.5
        )
        
        # Mean reversion: High volatility + weak trend
        mean_rev_score = (
            min(features.get('volatility', 0) / 0.03, 1.0) * 0.6 +
            (1 - min(features.get('trend_strength', 0) / 0.1, 1.0)) * 0.4
        )
        
        # Low volatility regime
        low_vol_score = 1 - min(features.get('volatility', 0) / 0.01, 1.0)
        
        # Low liquidity regime
        low_liq_score = 1 - min(features.get('volume_regime', 1.0), 1.0)
        
        # Determine dominant regime
        scores = {
            RegimeState.CRISIS: crisis_score,
            RegimeState.TREND: trend_score,
            RegimeState.MEAN_REVERT: mean_rev_score,
            RegimeState.LOW_VOLATILITY: low_vol_score,
            RegimeState.LOW_LIQUIDITY: low_liq_score
        }
        
        # Add normal regime as baseline
        normal_score = max(0.3, 1 - max(crisis_score, trend_score, mean_rev_score))
        scores[RegimeState.NORMAL] = normal_score
        
        # Get regime with highest score
        dominant_regime = max(scores, key=scores.get)
        confidence = scores[dominant_regime]
        
        # Additional logic for regime transitions
        if dominant_regime == RegimeState.NORMAL:
            # Check if we should classify as HIGH_VOLATILITY instead
            if features.get('volatility', 0) > 0.03:
                dominant_regime = RegimeState.HIGH_VOLATILITY
                confidence = min(crisis_score, 0.8)
        
        return dominant_regime, confidence
    
    def adjust_risk_for_regime(self, base_risk: float, regime: RegimeState) -> float:
        """Adjust risk allocation based on current regime"""
        adjustment_factors = {
            RegimeState.CRISIS: 0.3,  # Reduce exposure by 70%
            RegimeState.HIGH_VOLATILITY: 0.5,  # Reduce exposure by 50%
            RegimeState.LOW_LIQUIDITY: 0.6,  # Reduce exposure by 40%
            RegimeState.TREND: 1.2,  # Increase exposure by 20%
            RegimeState.MEAN_REVERT: 1.0,  # Normal exposure
            RegimeState.LOW_VOLATILITY: 1.1,  # Slightly increase exposure
            RegimeState.NORMAL: 1.0  # Normal exposure
        }
        
        factor = adjustment_factors.get(regime, 1.0)
        adjusted_risk = base_risk * factor
        
        # Ensure reasonable bounds
        adjusted_risk = max(min(adjusted_risk, base_risk * 2.0), base_risk * 0.1)
        
        logger.debug(f"⚖️ Risk adjustment: {base_risk:.4f} → {adjusted_risk:.4f} (factor: {factor:.2f}, regime: {regime.value})")
        
        return adjusted_risk
    
    def get_regime_summary(self) -> Dict:
        """Get summary of current regime state"""
        return {
            'current_regime': self.current_regime.value,
            'regime_confidence': self.confidence_history[-1] if self.confidence_history else 0.5,
            'regime_duration': self._get_regime_duration(),
            'regime_transition_probability': self._get_transition_probabilities(),
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_regime_duration(self) -> int:
        """Get duration of current regime"""
        if not self.regime_history:
            return 0
        
        # Count consecutive periods in same regime
        current_regime = self.regime_history[-1][1]
        count = 0
        for _, regime, _ in reversed(self.regime_history):
            if regime == current_regime:
                count += 1
            else:
                break
        
        return count
    
    def _get_transition_probabilities(self) -> Dict:
        """Get probabilities of transitioning to different regimes"""
        # Placeholder - in reality would use HMM transition probabilities
        return {
            'to_crisis': 0.1,
            'to_trend': 0.2,
            'to_mean_revert': 0.3,
            'to_normal': 0.4
        }
    
    def update_with_new_data(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> Optional[RegimeEvent]:
        """Update regime detection with new data and return event if regime changed"""
        try:
            new_regime, confidence, features = self.detect_regime(df, symbol)
            
            # Check if regime changed
            regime_changed = new_regime != self.current_regime
            
            regime_event = RegimeEvent(
                symbol=symbol,
                regime=new_regime,
                probability=0.0,  # Would be calculated from HMM
                confidence=confidence,
                features=features
            )
            
            if regime_changed:
                logger.info(f"🔄 Regime change detected: {self.current_regime.value} → {new_regime.value}")
            
            return regime_event
            
        except Exception as e:
            logger.error(f"❌ Regime update failed: {e}")
            return None

class RegimeAwareRiskManager:
    """Risk manager that adjusts parameters based on detected regime"""
    
    def __init__(self, base_risk_manager):
        self.base_risk_manager = base_risk_manager
        self.regime_engine = RegimeDetectionEngine()
        self.active_regime = RegimeState.NORMAL
        
        logger.info("🛡️ Regime-Aware Risk Manager initialized")
    
    def update_risk_limits(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Update risk limits based on current regime"""
        try:
            regime_event = self.regime_engine.update_with_new_data(df, symbol)
            
            if regime_event:
                self.active_regime = regime_event.regime
                
                # Adjust risk limits based on regime
                base_limits = self.base_risk_manager.get_current_limits()
                adjusted_limits = self._adjust_limits_for_regime(base_limits, self.active_regime)
                
                logger.info(f"📈 Risk limits adjusted for {self.active_regime.value} regime")
                return adjusted_limits
            
            return self.base_risk_manager.get_current_limits()
            
        except Exception as e:
            logger.error(f"❌ Risk limit adjustment failed: {e}")
            return self.base_risk_manager.get_current_limits()
    
    def _adjust_limits_for_regime(self, base_limits: Dict, regime: RegimeState) -> Dict:
        """Adjust risk limits based on regime"""
        adjusted_limits = base_limits.copy()
        
        # Adjust position sizing
        if regime in [RegimeState.CRISIS, RegimeState.HIGH_VOLATILITY, RegimeState.LOW_LIQUIDITY]:
            adjusted_limits['max_risk_per_trade'] *= 0.3  # Reduce to 30% of normal
            adjusted_limits['max_portfolio_risk'] *= 0.5  # Reduce to 50% of normal
        elif regime in [RegimeState.TREND]:
            adjusted_limits['max_risk_per_trade'] *= 1.2  # Increase by 20%
            adjusted_limits['max_portfolio_risk'] *= 1.1  # Increase by 10%
        
        # Adjust drawdown limits
        if regime == RegimeState.CRISIS:
            adjusted_limits['max_drawdown'] *= 0.7  # Tighten drawdown limit
        
        # Adjust correlation limits
        if regime == RegimeState.CRISIS:
            adjusted_limits['max_correlation_risk'] *= 0.8  # Reduce correlation tolerance
        
        return adjusted_limits