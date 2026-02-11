"""
Market Regime Detection Engine for Chloe 0.6
Professional HMM-based regime classification with Bayesian switching
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Market regime states"""
    STABLE = "STABLE"          # Low volatility, mean-reverting
    TRENDING = "TRENDING"      # Strong directional movement
    VOLATILE = "VOLATILE"      # High volatility, choppy
    CRISIS = "CRISIS"          # Extreme market stress

@dataclass
class RegimeState:
    """Current regime state with confidence"""
    regime: MarketRegime
    confidence: float
    timestamp: datetime
    features: Dict[str, float]
    transition_probabilities: Dict[MarketRegime, float]

class HiddenMarkovModel:
    """Simplified HMM implementation for regime detection"""
    
    def __init__(self, n_states: int = 4):
        self.n_states = n_states
        self.states = [MarketRegime.STABLE, MarketRegime.TRENDING, MarketRegime.VOLATILE, MarketRegime.CRISIS]
        
        # Transition matrix (state persistence)
        self.transition_matrix = np.array([
            [0.85, 0.05, 0.08, 0.02],  # STABLE: mostly stays stable
            [0.10, 0.75, 0.12, 0.03],  # TRENDING: moderate persistence
            [0.15, 0.10, 0.65, 0.10],  # VOLATILE: less persistent
            [0.05, 0.05, 0.15, 0.75]   # CRISIS: high persistence
        ])
        
        # Emission probabilities (feature likelihoods per state)
        self.emission_params = {
            MarketRegime.STABLE: {
                'volatility_mean': 0.015,
                'volatility_std': 0.005,
                'trend_strength_mean': 0.1,
                'trend_strength_std': 0.05,
                'autocorr_mean': 0.2,
                'autocorr_std': 0.1
            },
            MarketRegime.TRENDING: {
                'volatility_mean': 0.025,
                'volatility_std': 0.01,
                'trend_strength_mean': 0.6,
                'trend_strength_std': 0.15,
                'autocorr_mean': 0.5,
                'autocorr_std': 0.1
            },
            MarketRegime.VOLATILE: {
                'volatility_mean': 0.04,
                'volatility_std': 0.015,
                'trend_strength_mean': 0.3,
                'trend_strength_std': 0.2,
                'autocorr_mean': 0.1,
                'autocorr_std': 0.15
            },
            MarketRegime.CRISIS: {
                'volatility_mean': 0.08,
                'volatility_std': 0.03,
                'trend_strength_mean': 0.4,
                'trend_strength_std': 0.25,
                'autocorr_mean': -0.2,
                'autocorr_std': 0.2
            }
        }
        
        # Initial state probabilities
        self.initial_probs = np.array([0.4, 0.3, 0.2, 0.1])  # Start biased toward stable
        
        logger.info(f"HMM Regime Detector initialized with {n_states} states")

    def _calculate_emission_probability(self, features: Dict[str, float], state: MarketRegime) -> float:
        """Calculate emission probability for given features and state"""
        params = self.emission_params[state]
        
        # Calculate likelihood for each feature
        vol_likelihood = self._gaussian_pdf(
            features.get('volatility', 0.02),
            params['volatility_mean'],
            params['volatility_std']
        )
        
        trend_likelihood = self._gaussian_pdf(
            abs(features.get('trend_strength', 0)),
            params['trend_strength_mean'],
            params['trend_strength_std']
        )
        
        autocorr_likelihood = self._gaussian_pdf(
            features.get('autocorrelation', 0),
            params['autocorr_mean'],
            params['autocorr_std']
        )
        
        # Combine likelihoods (assuming independence)
        combined_likelihood = vol_likelihood * trend_likelihood * autocorr_likelihood
        return max(combined_likelihood, 1e-10)  # Avoid zero probabilities

    def _gaussian_pdf(self, x: float, mean: float, std: float) -> float:
        """Calculate Gaussian probability density function"""
        if std <= 0:
            return 1e-10
        coefficient = 1 / (std * np.sqrt(2 * np.pi))
        exponent = -0.5 * ((x - mean) / std) ** 2
        return coefficient * np.exp(exponent)

    def forward_algorithm(self, observation_sequence: List[Dict[str, float]]) -> Tuple[np.ndarray, float]:
        """Forward algorithm for HMM inference"""
        T = len(observation_sequence)
        alpha = np.zeros((T, self.n_states))
        
        # Initialize
        for i, state in enumerate(self.states):
            alpha[0, i] = self.initial_probs[i] * self._calculate_emission_probability(
                observation_sequence[0], state
            )
        
        # Forward pass
        for t in range(1, T):
            for j in range(self.n_states):
                # Sum over all previous states
                alpha[t, j] = sum(
                    alpha[t-1, i] * self.transition_matrix[i, j]
                    for i in range(self.n_states)
                ) * self._calculate_emission_probability(
                    observation_sequence[t], self.states[j]
                )
        
        # Likelihood of observations
        likelihood = sum(alpha[T-1, :])
        return alpha, likelihood

    def viterbi_algorithm(self, observation_sequence: List[Dict[str, float]]) -> Tuple[List[int], float]:
        """Viterbi algorithm for most likely state sequence"""
        T = len(observation_sequence)
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)
        
        # Initialize
        for i, state in enumerate(self.states):
            delta[0, i] = self.initial_probs[i] * self._calculate_emission_probability(
                observation_sequence[0], state
            )
        
        # Forward pass
        for t in range(1, T):
            for j in range(self.n_states):
                max_prob = 0
                max_state = 0
                for i in range(self.n_states):
                    prob = delta[t-1, i] * self.transition_matrix[i, j]
                    if prob > max_prob:
                        max_prob = prob
                        max_state = i
                delta[t, j] = max_prob * self._calculate_emission_probability(
                    observation_sequence[t], self.states[j]
                )
                psi[t, j] = max_state
        
        # Backtrack
        best_path = [0] * T
        best_path[T-1] = np.argmax(delta[T-1, :])
        
        for t in range(T-2, -1, -1):
            best_path[t] = psi[t+1, best_path[t+1]]
        
        # Path probability
        path_prob = delta[T-1, best_path[T-1]]
        return best_path, path_prob

class BayesianRegimeDetector:
    """Bayesian switching model for regime detection"""
    
    def __init__(self):
        # Prior beliefs about regimes
        self.prior_probs = {
            MarketRegime.STABLE: 0.4,
            MarketRegime.TRENDING: 0.3,
            MarketRegime.VOLATILE: 0.2,
            MarketRegime.CRISIS: 0.1
        }
        
        # Regime-specific parameters
        self.regime_params = {
            MarketRegime.STABLE: {
                'expected_volatility': 0.015,
                'expected_return': 0.0001,
                'volatility_of_volatility': 0.3
            },
            MarketRegime.TRENDING: {
                'expected_volatility': 0.025,
                'expected_return': 0.001,
                'volatility_of_volatility': 0.5
            },
            MarketRegime.VOLATILE: {
                'expected_volatility': 0.04,
                'expected_return': 0.0,
                'volatility_of_volatility': 0.8
            },
            MarketRegime.CRISIS: {
                'expected_volatility': 0.08,
                'expected_return': -0.002,
                'volatility_of_volatility': 1.2
            }
        }
        
        logger.info("Bayesian Regime Detector initialized")

    def update_posterior(self, current_features: Dict[str, float], 
                        previous_regime: MarketRegime) -> Dict[MarketRegime, float]:
        """Update posterior probabilities using Bayes' theorem"""
        posteriors = {}
        evidence = 0
        
        current_volatility = current_features.get('volatility', 0.02)
        current_return = current_features.get('return', 0.0)
        
        # Calculate likelihood for each regime
        for regime in MarketRegime:
            if regime == MarketRegime.STABLE:
                # For stable regimes, favor low volatility and mean-reverting returns
                likelihood = self._calculate_stable_likelihood(
                    current_volatility, current_return, regime
                )
            elif regime == MarketRegime.TRENDING:
                # For trending regimes, favor directional returns
                likelihood = self._calculate_trending_likelihood(
                    current_volatility, current_return
                )
            elif regime == MarketRegime.VOLATILE:
                # For volatile regimes, favor high volatility
                likelihood = self._calculate_volatile_likelihood(
                    current_volatility, current_return
                )
            else:  # CRISIS
                # For crisis regimes, favor high volatility and negative returns
                likelihood = self._calculate_crisis_likelihood(
                    current_volatility, current_return
                )
            
            # Apply transition probability from previous regime
            transition_prob = self._get_transition_probability(previous_regime, regime)
            
            # Calculate posterior
            posterior = self.prior_probs[regime] * likelihood * transition_prob
            posteriors[regime] = posterior
            evidence += posterior
        
        # Normalize posteriors
        if evidence > 0:
            for regime in posteriors:
                posteriors[regime] /= evidence
        else:
            # Fallback to uniform distribution
            uniform_prob = 1.0 / len(MarketRegime)
            posteriors = {regime: uniform_prob for regime in MarketRegime}
        
        return posteriors

    def _calculate_stable_likelihood(self, volatility: float, return_val: float, 
                                   regime: MarketRegime) -> float:
        """Calculate likelihood for stable/mean-reverting regimes"""
        params = self.regime_params[regime]
        
        # Low volatility preference
        vol_likelihood = np.exp(-0.5 * (volatility / params['expected_volatility']) ** 2)
        
        # Mean-reverting return preference (closer to zero)
        return_likelihood = np.exp(-2.0 * abs(return_val) / abs(params['expected_return'] or 0.001))
        
        return vol_likelihood * return_likelihood

    def _calculate_trending_likelihood(self, volatility: float, return_val: float) -> float:
        """Calculate likelihood for trending regime"""
        params = self.regime_params[MarketRegime.TRENDING]
        
        # Moderate volatility preference
        vol_likelihood = np.exp(-0.5 * ((volatility - params['expected_volatility']) / 
                                      params['volatility_of_volatility']) ** 2)
        
        # Directional return preference
        return_likelihood = np.exp(abs(return_val) / abs(params['expected_return'] or 0.001))
        
        return vol_likelihood * return_likelihood

    def _calculate_volatile_likelihood(self, volatility: float, return_val: float) -> float:
        """Calculate likelihood for volatile regime"""
        params = self.regime_params[MarketRegime.VOLATILE]
        
        # High volatility preference
        vol_likelihood = 1 - np.exp(-2.0 * volatility / params['expected_volatility'])
        
        # Random return preference (any direction)
        return_likelihood = 1.0  # Neutral to return direction
        
        return vol_likelihood * return_likelihood

    def _calculate_crisis_likelihood(self, volatility: float, return_val: float) -> float:
        """Calculate likelihood for crisis regime"""
        params = self.regime_params[MarketRegime.CRISIS]
        
        # Very high volatility preference
        vol_likelihood = 1 - np.exp(-volatility / params['expected_volatility'])
        
        # Negative return preference
        return_likelihood = np.exp(return_val / abs(params['expected_return'] or 0.001))
        
        return vol_likelihood * return_likelihood

    def _get_transition_probability(self, from_regime: MarketRegime, 
                                  to_regime: MarketRegime) -> float:
        """Get transition probability between regimes"""
        # Simplified transition matrix
        transition_matrix = {
            MarketRegime.STABLE: {
                MarketRegime.STABLE: 0.85,
                MarketRegime.TRENDING: 0.05,
                MarketRegime.VOLATILE: 0.08,
                MarketRegime.CRISIS: 0.02
            },
            MarketRegime.TRENDING: {
                MarketRegime.STABLE: 0.10,
                MarketRegime.TRENDING: 0.75,
                MarketRegime.VOLATILE: 0.12,
                MarketRegime.CRISIS: 0.03
            },
            MarketRegime.VOLATILE: {
                MarketRegime.STABLE: 0.15,
                MarketRegime.TRENDING: 0.10,
                MarketRegime.VOLATILE: 0.65,
                MarketRegime.CRISIS: 0.10
            },
            MarketRegime.CRISIS: {
                MarketRegime.STABLE: 0.05,
                MarketRegime.TRENDING: 0.05,
                MarketRegime.VOLATILE: 0.15,
                MarketRegime.CRISIS: 0.75
            }
        }
        
        return transition_matrix.get(from_regime, {}).get(to_regime, 0.25)

class RegimeFeatureExtractor:
    """Extract regime-relevant features from market data"""
    
    def __init__(self):
        logger.info("Regime Feature Extractor initialized")

    def extract_features(self, price_series: pd.Series, 
                        volume_series: Optional[pd.Series] = None) -> Dict[str, float]:
        """Extract comprehensive regime features"""
        try:
            features = {}
            
            # Basic statistics
            returns = price_series.pct_change().dropna()
            
            if len(returns) < 5:
                return self._get_default_features()
            
            # Volatility measures
            features['volatility'] = returns.std()
            features['realized_volatility'] = np.sqrt((returns ** 2).mean())
            
            # Trend measures
            features['trend_strength'] = self._calculate_trend_strength(returns)
            features['price_trend'] = (price_series.iloc[-1] / price_series.iloc[0]) - 1
            
            # Autocorrelation measures
            features['autocorrelation'] = self._calculate_autocorrelation(returns, lag=1)
            features['autocorr_5'] = self._calculate_autocorrelation(returns, lag=5)
            
            # Return distribution features
            features['skewness'] = self._calculate_skewness(returns)
            features['kurtosis'] = self._calculate_kurtosis(returns)
            features['tail_risk'] = self._calculate_tail_risk(returns)
            
            # Momentum features
            features['momentum_5'] = self._calculate_momentum(price_series, 5)
            features['momentum_20'] = self._calculate_momentum(price_series, 20)
            
            # Volume features (if available)
            if volume_series is not None and len(volume_series) == len(price_series):
                features['volume_trend'] = self._calculate_volume_trend(volume_series)
                features['volume_volatility'] = volume_series.pct_change().std()
            
            # Regime-specific ratios
            features['volatility_ratio'] = self._calculate_volatility_ratio(returns)
            features['trend_volatility_ratio'] = abs(features['trend_strength']) / (features['volatility'] + 1e-8)
            
            # Market stress indicators
            features['market_stress'] = self._calculate_market_stress(returns, features['volatility'])
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return self._get_default_features()

    def _calculate_trend_strength(self, returns: pd.Series) -> float:
        """Calculate trend strength using linear regression slope"""
        if len(returns) < 10:
            return 0.0
        
        x = np.arange(len(returns))
        slope, _ = np.polyfit(x, returns.cumsum(), 1)
        return slope

    def _calculate_autocorrelation(self, returns: pd.Series, lag: int = 1) -> float:
        """Calculate return autocorrelation"""
        if len(returns) <= lag:
            return 0.0
        return returns.autocorr(lag=lag)

    def _calculate_skewness(self, returns: pd.Series) -> float:
        """Calculate return skewness"""
        return returns.skew() if len(returns) > 3 else 0.0

    def _calculate_kurtosis(self, returns: pd.Series) -> float:
        """Calculate return kurtosis"""
        return returns.kurtosis() if len(returns) > 4 else 3.0

    def _calculate_tail_risk(self, returns: pd.Series) -> float:
        """Calculate tail risk using VaR-like measure"""
        if len(returns) < 10:
            return 0.0
        return abs(np.percentile(returns, 5))  # 5% quantile

    def _calculate_momentum(self, prices: pd.Series, period: int) -> float:
        """Calculate price momentum"""
        if len(prices) <= period:
            return 0.0
        return (prices.iloc[-1] / prices.iloc[-period-1]) - 1

    def _calculate_volume_trend(self, volume: pd.Series) -> float:
        """Calculate volume trend"""
        if len(volume) < 10:
            return 0.0
        return volume.pct_change(periods=5).mean()

    def _calculate_volatility_ratio(self, returns: pd.Series) -> float:
        """Calculate ratio of recent to historical volatility"""
        if len(returns) < 20:
            return 1.0
        
        recent_vol = returns[-10:].std()
        hist_vol = returns[:-10].std()
        
        return recent_vol / (hist_vol + 1e-8)

    def _calculate_market_stress(self, returns: pd.Series, volatility: float) -> float:
        """Calculate composite market stress indicator"""
        if len(returns) < 10:
            return 0.0
            
        # Combine multiple stress indicators
        tail_risk = abs(np.percentile(returns, 10))
        return_signaling = abs(returns.mean()) / (volatility + 1e-8)
        volatility_spike = volatility / (returns.std() + 1e-8)
        
        stress_score = (tail_risk * 0.4 + return_signaling * 0.3 + volatility_spike * 0.3)
        return min(stress_score, 1.0)

    def _get_default_features(self) -> Dict[str, float]:
        """Return default features when calculation fails"""
        return {
            'volatility': 0.02,
            'realized_volatility': 0.02,
            'trend_strength': 0.0,
            'price_trend': 0.0,
            'autocorrelation': 0.0,
            'autocorr_5': 0.0,
            'skewness': 0.0,
            'kurtosis': 3.0,
            'tail_risk': 0.01,
            'momentum_5': 0.0,
            'momentum_20': 0.0,
            'volume_trend': 0.0,
            'volume_volatility': 0.1,
            'volatility_ratio': 1.0,
            'trend_volatility_ratio': 0.0,
            'market_stress': 0.1
        }

class MarketRegimeDetector:
    """Main regime detection engine combining HMM and Bayesian approaches"""
    
    def __init__(self):
        self.hmm_model = HiddenMarkovModel()
        self.bayesian_detector = BayesianRegimeDetector()
        self.feature_extractor = RegimeFeatureExtractor()
        
        # State tracking
        self.current_regime = MarketRegime.STABLE
        self.regime_confidence = 0.5
        self.regime_history = []
        self.feature_history = []
        
        logger.info("Market Regime Detector initialized")

    def detect_regime(self, price_data: pd.Series, 
                     volume_data: Optional[pd.Series] = None,
                     method: str = 'hybrid') -> RegimeState:
        """Detect current market regime using specified method"""
        try:
            # Extract features
            features = self.feature_extractor.extract_features(price_data, volume_data)
            self.feature_history.append(features)
            
            # Limit history size
            if len(self.feature_history) > 100:
                self.feature_history = self.feature_history[-50:]
            
            # Detect regime based on method
            if method == 'hmm':
                regime_state = self._detect_with_hmm(features)
            elif method == 'bayesian':
                regime_state = self._detect_with_bayesian(features)
            else:  # hybrid
                regime_state = self._detect_hybrid(features)
            
            # Update state tracking
            self.current_regime = regime_state.regime
            self.regime_confidence = regime_state.confidence
            self.regime_history.append(regime_state)
            
            # Limit history size
            if len(self.regime_history) > 1000:
                self.regime_history = self.regime_history[-500:]
            
            return regime_state
            
        except Exception as e:
            logger.error(f"Regime detection failed: {e}")
            return self._get_default_regime_state()

    def _detect_with_hmm(self, current_features: Dict[str, float]) -> RegimeState:
        """Detect regime using HMM approach"""
        # Create observation sequence from recent features
        recent_features = self.feature_history[-20:] if len(self.feature_history) >= 20 else self.feature_history
        if not recent_features:
            recent_features = [current_features]
        
        # Simplified HMM detection (would use full sequence in production)
        regime_scores = {}
        for regime in MarketRegime:
            emission_prob = self.hmm_model._calculate_emission_probability(current_features, regime)
            regime_scores[regime] = emission_prob
        
        # Select best regime
        best_regime = max(regime_scores.keys(), key=lambda k: regime_scores[k])
        confidence = regime_scores[best_regime] / sum(regime_scores.values())
        
        return RegimeState(
            regime=best_regime,
            confidence=min(confidence, 1.0),
            timestamp=datetime.now(),
            features=current_features,
            transition_probabilities={regime: 0.25 for regime in MarketRegime}  # Placeholder
        )

    def _detect_with_bayesian(self, current_features: Dict[str, float]) -> RegimeState:
        """Detect regime using Bayesian approach"""
        posteriors = self.bayesian_detector.update_posterior(
            current_features, self.current_regime
        )
        
        # Select regime with highest posterior
        best_regime = max(posteriors.keys(), key=lambda k: posteriors[k])
        confidence = posteriors[best_regime]
        
        return RegimeState(
            regime=best_regime,
            confidence=confidence,
            timestamp=datetime.now(),
            features=current_features,
            transition_probabilities=posteriors
        )

    def _detect_hybrid(self, current_features: Dict[str, float]) -> RegimeState:
        """Hybrid detection combining both approaches"""
        # Get results from both methods
        hmm_state = self._detect_with_hmm(current_features)
        bayesian_state = self._detect_with_bayesian(current_features)
        
        # Weighted combination (Bayesian gets higher weight due to better theoretical foundation)
        hybrid_confidence = 0.6 * bayesian_state.confidence + 0.4 * hmm_state.confidence
        
        # Select regime (prefer Bayesian result when confident)
        if bayesian_state.confidence > 0.6:
            final_regime = bayesian_state.regime
        elif hmm_state.confidence > 0.7:
            final_regime = hmm_state.regime
        else:
            # Use weighted voting
            regime_weights = {}
            for regime in MarketRegime:
                hmm_weight = 0.4 if hmm_state.regime == regime else 0.1
                bayes_weight = 0.6 if bayesian_state.regime == regime else 0.15
                regime_weights[regime] = hmm_weight + bayes_weight
            
            final_regime = max(regime_weights.keys(), key=lambda k: regime_weights[k])
        
        return RegimeState(
            regime=final_regime,
            confidence=min(hybrid_confidence, 1.0),
            timestamp=datetime.now(),
            features=current_features,
            transition_probabilities=bayesian_state.transition_probabilities
        )

    def _get_default_regime_state(self) -> RegimeState:
        """Return default regime state when detection fails"""
        return RegimeState(
            regime=MarketRegime.STABLE,
            confidence=0.5,
            timestamp=datetime.now(),
            features=self.feature_extractor._get_default_features(),
            transition_probabilities={regime: 0.25 for regime in MarketRegime}
        )

    def get_regime_summary(self) -> Dict:
        """Get summary of regime detection performance"""
        if not self.regime_history:
            return {"status": "NO_HISTORY"}
        
        recent_regimes = self.regime_history[-50:]  # Last 50 detections
        regime_counts = {}
        avg_confidence = 0.0
        
        for state in recent_regimes:
            regime = state.regime.value
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
            avg_confidence += state.confidence
        
        avg_confidence /= len(recent_regimes)
        
        return {
            "total_detections": len(self.regime_history),
            "recent_regime_distribution": regime_counts,
            "average_confidence": avg_confidence,
            "current_regime": self.current_regime.value,
            "current_confidence": self.regime_confidence
        }

# Global instance
_regime_detector = None

def get_regime_detector() -> MarketRegimeDetector:
    """Get singleton regime detector instance"""
    global _regime_detector
    if _regime_detector is None:
        _regime_detector = MarketRegimeDetector()
    return _regime_detector

def main():
    """Example usage"""
    print("Market Regime Detection Engine ready")
    print("Professional HMM + Bayesian regime classification")

if __name__ == "__main__":
    main()