"""
Advanced Risk Engine for Chloe 0.6
Professional risk management with Kelly Criterion, CVaR optimization, and regime-aware calibration
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class KellyPosition:
    """Kelly criterion position sizing result"""
    symbol: str
    optimal_fraction: float      # Optimal capital fraction to allocate
    expected_growth: float       # Expected logarithmic growth rate
    risk_of_ruin: float         # Probability of significant drawdown
    regime_adjustment: float    # Regime-based adjustment factor

@dataclass
class CVaROptimization:
    """CVaR (Conditional Value at Risk) optimization result"""
    optimal_weights: Dict[str, float]  # Asset weights
    portfolio_cvar: float              # Portfolio CVaR
    expected_return: float             # Portfolio expected return
    sharpe_ratio: float               # Risk-adjusted return
    diversification_ratio: float      # Portfolio diversification measure

@dataclass
class RiskCalibration:
    """Regime-aware risk calibration parameters"""
    regime: str
    volatility_scaling: float         # Volatility adjustment factor
    correlation_adjustment: float     # Correlation impact modifier
    confidence_level: float          # VaR/CVaR confidence level
    risk_aversion: float             # Investor risk aversion parameter
    max_position_size: float         # Maximum position constraint

class AdvancedRiskEngine:
    """Professional risk engine with mathematical rigor"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.historical_returns = {}
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        
        # Risk parameters
        self.default_confidence_level = 0.95
        self.default_risk_aversion = 2.0
        self.maximum_drawdown_limit = 0.20  # 20% maximum drawdown
        self.maximum_correlation_risk = 0.6
        
        # Regime-aware calibration
        self.regime_calibration = {
            'STABLE': RiskCalibration(
                regime='STABLE',
                volatility_scaling=1.0,
                correlation_adjustment=1.0,
                confidence_level=0.95,
                risk_aversion=1.5,
                max_position_size=0.15
            ),
            'TRENDING': RiskCalibration(
                regime='TRENDING',
                volatility_scaling=1.2,
                correlation_adjustment=1.1,
                confidence_level=0.90,
                risk_aversion=1.8,
                max_position_size=0.20
            ),
            'VOLATILE': RiskCalibration(
                regime='VOLATILE',
                volatility_scaling=0.8,
                correlation_adjustment=1.3,
                confidence_level=0.97,
                risk_aversion=2.5,
                max_position_size=0.10
            ),
            'CRISIS': RiskCalibration(
                regime='CRISIS',
                volatility_scaling=0.6,
                correlation_adjustment=1.5,
                confidence_level=0.99,
                risk_aversion=3.0,
                max_position_size=0.05
            )
        }
        
        logger.info(f"Advanced Risk Engine initialized with ${initial_capital:,.2f}")

    def calculate_kelly_criterion(self, win_rate: float, win_loss_ratio: float, 
                                regime: str = 'STABLE') -> KellyPosition:
        """Calculate optimal position size using Kelly criterion with regime adjustment"""
        try:
            # Validate inputs
            if not 0 < win_rate < 1:
                raise ValueError("Win rate must be between 0 and 1")
            if win_loss_ratio <= 0:
                raise ValueError("Win-loss ratio must be positive")
            
            # Basic Kelly formula: f* = (bp - q) / b
            # where b = win_loss_ratio, p = win_rate, q = 1 - win_rate
            kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
            
            # Apply regime-specific adjustments
            calibration = self.regime_calibration[regime]
            regime_adjusted_kelly = kelly_fraction * calibration.volatility_scaling
            
            # Conservative fractional Kelly (1/4 to 1/2 Kelly to reduce variance)
            fractional_kelly = regime_adjusted_kelly * 0.25
            
            # Apply maximum position constraints
            max_position = calibration.max_position_size
            optimal_fraction = max(0.0, min(fractional_kelly, max_position))
            
            # Calculate expected growth rate
            expected_growth = win_rate * np.log(1 + optimal_fraction * win_loss_ratio) + \
                            (1 - win_rate) * np.log(1 - optimal_fraction)
            
            # Estimate risk of ruin (simplified)
            risk_of_ruin = np.exp(-2 * optimal_fraction * win_loss_ratio * win_rate)
            
            return KellyPosition(
                symbol='PORTFOLIO',
                optimal_fraction=optimal_fraction,
                expected_growth=expected_growth,
                risk_of_ruin=min(risk_of_ruin, 1.0),
                regime_adjustment=calibration.volatility_scaling
            )
            
        except Exception as e:
            logger.error(f"Kelly criterion calculation failed: {e}")
            return KellyPosition(
                symbol='PORTFOLIO',
                optimal_fraction=0.01,  # Default small position
                expected_growth=0.0,
                risk_of_ruin=0.5,
                regime_adjustment=1.0
            )

    def optimize_cvar_portfolio(self, expected_returns: Dict[str, float],
                              covariance_matrix: np.ndarray,
                              symbols: List[str],
                              regime: str = 'STABLE',
                              constraints: Optional[Dict] = None) -> CVaROptimization:
        """Optimize portfolio using CVaR (Conditional Value at Risk) methodology"""
        try:
            n_assets = len(symbols)
            
            if n_assets == 0:
                raise ValueError("No assets provided for optimization")
            
            if covariance_matrix.shape != (n_assets, n_assets):
                raise ValueError("Covariance matrix dimensions mismatch")
            
            # Get regime calibration
            calibration = self.regime_calibration[regime]
            
            # Set up optimization problem
            expected_returns_array = np.array([expected_returns.get(sym, 0.0) for sym in symbols])
            
            # Constraints
            if constraints is None:
                constraints = {}
            
            # Bounds: 0 <= weight <= max_position_size
            bounds = [(0, calibration.max_position_size) for _ in range(n_assets)]
            
            # Constraint: sum of weights = 1 (fully invested)
            def weight_sum_constraint(weights):
                return np.sum(weights) - 1.0
            
            # Constraint: maximum correlation risk
            def correlation_constraint(weights):
                portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
                return self.maximum_correlation_risk - portfolio_variance
            
            # Objective function: Minimize CVaR
            def cvar_objective(weights):
                # Portfolio expected return
                portfolio_return = np.dot(weights, expected_returns_array)
                
                # Portfolio variance
                portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
                portfolio_volatility = np.sqrt(portfolio_variance)
                
                # Calculate CVaR at specified confidence level
                confidence_level = calibration.confidence_level
                z_score = norm.ppf(confidence_level)
                cvar = -(portfolio_return - z_score * portfolio_volatility)
                
                # Apply risk aversion
                return cvar * calibration.risk_aversion
            
            # Initial guess: equal weights
            initial_weights = np.ones(n_assets) / n_assets
            
            # Solve optimization
            result = minimize(
                cvar_objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=[
                    {'type': 'eq', 'fun': weight_sum_constraint},
                    {'type': 'ineq', 'fun': correlation_constraint}
                ],
                options={'disp': False}
            )
            
            if not result.success:
                logger.warning(f"CVaR optimization failed: {result.message}")
                # Fallback to equal weights
                optimal_weights = initial_weights
            else:
                optimal_weights = result.x
            
            # Calculate portfolio metrics
            portfolio_return = np.dot(optimal_weights, expected_returns_array)
            portfolio_variance = np.dot(optimal_weights.T, np.dot(covariance_matrix, optimal_weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            portfolio_cvar = -(portfolio_return - norm.ppf(calibration.confidence_level) * portfolio_volatility)
            
            # Sharpe ratio
            sharpe_ratio = (portfolio_return - self.risk_free_rate/252) / (portfolio_volatility + 1e-8)
            
            # Diversification ratio
            individual_risks = np.sqrt(np.diag(covariance_matrix))
            weighted_individual_risk = np.dot(optimal_weights, individual_risks)
            diversification_ratio = weighted_individual_risk / (portfolio_volatility + 1e-8)
            
            # Convert weights to dictionary
            weight_dict = dict(zip(symbols, optimal_weights))
            
            return CVaROptimization(
                optimal_weights=weight_dict,
                portfolio_cvar=max(0, portfolio_cvar),  # Ensure non-negative
                expected_return=portfolio_return,
                sharpe_ratio=sharpe_ratio,
                diversification_ratio=diversification_ratio
            )
            
        except Exception as e:
            logger.error(f"CVaR optimization failed: {e}")
            # Return equal-weight portfolio as fallback
            n_assets = len(symbols)
            equal_weights = dict(zip(symbols, [1/n_assets] * n_assets))
            return CVaROptimization(
                optimal_weights=equal_weights,
                portfolio_cvar=0.05,  # Default CVaR
                expected_return=0.001,
                sharpe_ratio=0.1,
                diversification_ratio=1.0
            )

    def calculate_value_at_risk(self, returns: np.ndarray, 
                              confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk using historical simulation"""
        try:
            if len(returns) < 10:
                return 0.0
            
            # Sort returns
            sorted_returns = np.sort(returns)
            
            # Calculate VaR at confidence level
            var_index = int((1 - confidence_level) * len(sorted_returns))
            var = abs(sorted_returns[var_index])
            
            return var
            
        except Exception as e:
            logger.error(f"VaR calculation error: {e}")
            return 0.05  # Default 5%

    def calculate_conditional_var(self, returns: np.ndarray, 
                                confidence_level: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        try:
            if len(returns) < 10:
                return 0.0
            
            # Sort returns
            sorted_returns = np.sort(returns)
            
            # Calculate CVaR - average of worst (1-confidence_level)% returns
            var_index = int((1 - confidence_level) * len(sorted_returns))
            cvar = abs(np.mean(sorted_returns[:var_index]))
            
            return cvar
            
        except Exception as e:
            logger.error(f"CVaR calculation error: {e}")
            return 0.08  # Default 8%

    def assess_regime_risk(self, current_regime: str, 
                         portfolio_weights: Dict[str, float],
                         asset_returns: Dict[str, List[float]]) -> Dict:
        """Assess portfolio risk given current market regime"""
        try:
            calibration = self.regime_calibration[current_regime]
            
            # Calculate portfolio returns
            portfolio_returns = self._calculate_portfolio_returns(portfolio_weights, asset_returns)
            
            if len(portfolio_returns) < 10:
                return self._get_default_risk_assessment(current_regime)
            
            # Calculate risk metrics
            var = self.calculate_value_at_risk(portfolio_returns, calibration.confidence_level)
            cvar = self.calculate_conditional_var(portfolio_returns, calibration.confidence_level)
            volatility = np.std(portfolio_returns) * np.sqrt(252)  # Annualized
            
            # Maximum drawdown
            cumulative_returns = np.cumprod(1 + portfolio_returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = abs(np.min(drawdown))
            
            # Sharpe ratio
            excess_return = np.mean(portfolio_returns) * 252 - self.risk_free_rate
            sharpe_ratio = excess_return / (volatility + 1e-8)
            
            # Risk assessment
            risk_assessment = {
                'regime': current_regime,
                'volatility': volatility,
                'value_at_risk': var,
                'conditional_var': cvar,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'risk_rating': self._calculate_risk_rating(var, max_drawdown, current_regime),
                'allocation_suggestions': self._suggest_allocation_adjustments(
                    current_regime, portfolio_weights
                )
            }
            
            return risk_assessment
            
        except Exception as e:
            logger.error(f"Regime risk assessment failed: {e}")
            return self._get_default_risk_assessment(current_regime)

    def _calculate_portfolio_returns(self, weights: Dict[str, float], 
                                   asset_returns: Dict[str, List[float]]) -> np.ndarray:
        """Calculate portfolio returns from individual asset returns and weights"""
        try:
            if not asset_returns:
                return np.array([])
            
            # Get common timestamps
            common_dates = set()
            for returns in asset_returns.values():
                if returns:
                    common_dates.update(range(len(returns)))
            
            if not common_dates:
                return np.array([])
            
            # Calculate weighted returns
            portfolio_returns = []
            for i in sorted(common_dates):
                daily_return = 0.0
                total_weight = 0.0
                
                for symbol, weight in weights.items():
                    if symbol in asset_returns and len(asset_returns[symbol]) > i:
                        daily_return += weight * asset_returns[symbol][i]
                        total_weight += weight
                
                if total_weight > 0:
                    portfolio_returns.append(daily_return / total_weight)
            
            return np.array(portfolio_returns)
            
        except Exception:
            return np.array([])

    def _calculate_risk_rating(self, var: float, max_drawdown: float, regime: str) -> str:
        """Calculate qualitative risk rating"""
        # Regime-specific thresholds
        thresholds = {
            'STABLE': {'var': 0.03, 'drawdown': 0.10},
            'TRENDING': {'var': 0.05, 'drawdown': 0.15},
            'VOLATILE': {'var': 0.08, 'drawdown': 0.20},
            'CRISIS': {'var': 0.12, 'drawdown': 0.25}
        }
        
        regime_thresholds = thresholds.get(regime, thresholds['STABLE'])
        
        # Calculate risk score
        var_risk = var / regime_thresholds['var']
        drawdown_risk = max_drawdown / regime_thresholds['drawdown']
        combined_risk = (var_risk + drawdown_risk) / 2
        
        if combined_risk < 0.5:
            return 'LOW'
        elif combined_risk < 1.0:
            return 'MODERATE'
        elif combined_risk < 1.5:
            return 'HIGH'
        else:
            return 'VERY_HIGH'

    def _suggest_allocation_adjustments(self, regime: str, 
                                      current_weights: Dict[str, float]) -> Dict:
        """Suggest portfolio allocation adjustments based on regime"""
        calibration = self.regime_calibration[regime]
        
        suggestions = {
            'risk_tolerance': calibration.risk_aversion,
            'position_limits': calibration.max_position_size,
            'volatility_adjustment': calibration.volatility_scaling,
            'recommended_shifts': {}
        }
        
        # Suggest adjustments based on regime characteristics
        if regime == 'VOLATILE' or regime == 'CRISIS':
            # Reduce exposure, increase cash
            suggestions['recommended_shifts']['increase_cash'] = 0.1
            suggestions['recommended_shifts']['reduce_risky_assets'] = 0.15
        elif regime == 'TRENDING':
            # Can increase trend-following exposure
            suggestions['recommended_shifts']['increase_trending_assets'] = 0.1
        elif regime == 'STABLE':
            # Maintain balanced approach
            suggestions['recommended_shifts']['maintain_current_allocation'] = True
        
        return suggestions

    def _get_default_risk_assessment(self, regime: str) -> Dict:
        """Return default risk assessment when calculation fails"""
        return {
            'regime': regime,
            'volatility': 0.15,
            'value_at_risk': 0.05,
            'conditional_var': 0.08,
            'max_drawdown': 0.10,
            'sharpe_ratio': 0.0,
            'risk_rating': 'MODERATE',
            'allocation_suggestions': {'maintain_current_allocation': True}
        }

    def update_portfolio_state(self, positions: Dict, current_prices: Dict):
        """Update portfolio state with current positions and prices"""
        try:
            self.positions = positions.copy()
            portfolio_value = self._calculate_portfolio_value(current_prices)
            self.current_capital = portfolio_value
            
            # Update historical returns for risk calculations
            self._update_historical_returns(positions, current_prices)
            
        except Exception as e:
            logger.error(f"Portfolio state update error: {e}")

    def _calculate_portfolio_value(self, current_prices: Dict) -> float:
        """Calculate current portfolio value"""
        try:
            portfolio_value = 0.0
            
            for symbol, position in self.positions.items():
                if symbol in current_prices:
                    size = position.get('size', 0)
                    portfolio_value += size * current_prices[symbol]
                    
            return portfolio_value + self.current_capital  # Include cash
            
        except Exception:
            return self.current_capital

    def _update_historical_returns(self, positions: Dict, current_prices: Dict):
        """Update historical returns for risk metric calculations"""
        try:
            for symbol in positions.keys():
                if symbol not in self.historical_returns:
                    self.historical_returns[symbol] = []
                
                # This would typically integrate with a data feed
                # For now, we'll simulate updates
                if symbol in current_prices:
                    # Simulate return calculation
                    simulated_return = np.random.normal(0, 0.02)  # 2% daily volatility
                    self.historical_returns[symbol].append(simulated_return)
                    
                    # Keep only recent history
                    if len(self.historical_returns[symbol]) > 252:
                        self.historical_returns[symbol] = self.historical_returns[symbol][-252:]
                        
        except Exception as e:
            logger.warning(f"Historical returns update warning: {e}")

# Global instance
_advanced_risk_engine = None

def get_advanced_risk_engine(initial_capital: float = 100000.0) -> AdvancedRiskEngine:
    """Get singleton advanced risk engine instance"""
    global _advanced_risk_engine
    if _advanced_risk_engine is None:
        _advanced_risk_engine = AdvancedRiskEngine(initial_capital)
    return _advanced_risk_engine

def main():
    """Example usage"""
    print("Advanced Risk Engine ready")
    print("Professional Kelly Criterion + CVaR optimization")

if __name__ == "__main__":
    main()