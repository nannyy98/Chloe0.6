"""
Top-Down Portfolio Construction for Chloe 0.6
Professional portfolio construction from objectives down to individual trades
Instead of signal → trade, we use portfolio objective → allocate → trade approach
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class PortfolioObjective(Enum):
    """Portfolio construction objectives"""
    MAXIMIZE_RETURN = "MAXIMIZE_RETURN"
    MINIMIZE_RISK = "MINIMIZE_RISK"
    RISK_PARITY = "RISK_PARITY"
    MAXIMIZE_SHARPE = "MAXIMIZE_SHARPE"
    TARGET_VOLATILITY = "TARGET_VOLATILITY"
    ABSOLUTE_RETURN = "ABSOLUTE_RETURN"

@dataclass
class PortfolioConstraint:
    """Portfolio construction constraints"""
    max_position_size: float = 0.20      # Maximum weight per asset
    min_position_size: float = 0.01      # Minimum weight per asset
    max_sector_exposure: float = 0.40    # Maximum sector concentration
    max_correlation_risk: float = 0.60   # Maximum correlation exposure
    target_volatility: float = 0.15      # Target portfolio volatility
    max_tracking_error: float = 0.05     # Maximum deviation from benchmark

@dataclass
class AssetAllocation:
    """Individual asset allocation result"""
    symbol: str
    target_weight: float           # Target portfolio weight
    target_dollar_amount: float    # Dollar amount to allocate
    edge_probability: float        # Edge probability from edge models
    risk_contribution: float       # Risk contribution to portfolio
    expected_return: float         # Expected return
    regime_adjustment: float       # Regime-based adjustment factor
    priority_score: float          # Priority for execution

@dataclass
class PortfolioConstructionPlan:
    """Complete portfolio construction plan"""
    objective: PortfolioObjective
    target_portfolio_value: float
    current_portfolio_value: float
    allocations: List[AssetAllocation]
    portfolio_metrics: Dict[str, float]  # Sharpe, volatility, etc.
    risk_budget_allocation: Dict[str, float]  # Risk budget by asset
    execution_priority: List[str]            # Symbols in execution order
    rebalance_required: bool                 # Whether rebalancing is needed

class TopDownPortfolioConstructor:
    """Professional top-down portfolio construction engine"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.current_positions = {}
        self.portfolio_constraints = PortfolioConstraint()
        self.risk_free_rate = 0.02  # 2% annual
        
        # Risk parity parameters
        self.risk_budget = {}  # Risk budget allocation
        self.risk_contributions = {}  # Individual asset risk contributions
        
        # Regime-aware adjustments
        self.regime_adjustments = {
            'STABLE': {'risk_multiplier': 1.0, 'return_multiplier': 1.0},
            'TRENDING': {'risk_multiplier': 1.2, 'return_multiplier': 1.3},
            'VOLATILE': {'risk_multiplier': 0.7, 'return_multiplier': 0.8},
            'CRISIS': {'risk_multiplier': 0.5, 'return_multiplier': 0.6}
        }
        
        logger.info(f"Top-Down Portfolio Constructor initialized with ${initial_capital:,.2f}")

    def construct_portfolio(self, 
                          market_view: Dict[str, float],  # Symbol -> edge_probability
                          expected_returns: Dict[str, float],  # Symbol -> expected_return
                          covariance_matrix: np.ndarray,
                          symbols: List[str],
                          current_regime: str,
                          objective: PortfolioObjective = PortfolioObjective.MAXIMIZE_SHARPE,
                          constraints: Optional[PortfolioConstraint] = None) -> PortfolioConstructionPlan:
        """Construct portfolio using top-down approach"""
        try:
            # Apply regime adjustments
            regime_params = self.regime_adjustments.get(current_regime, 
                                                      self.regime_adjustments['STABLE'])
            
            # Adjust expected returns based on regime
            adjusted_returns = {}
            for symbol in symbols:
                base_return = expected_returns.get(symbol, 0.0)
                adjusted_returns[symbol] = base_return * regime_params['return_multiplier']
            
            # Adjust risk parameters
            risk_multiplier = regime_params['risk_multiplier']
            
            # Apply constraints
            if constraints is None:
                constraints = self.portfolio_constraints
            
            # Select optimization method based on objective
            if objective == PortfolioObjective.RISK_PARITY:
                allocations = self._optimize_risk_parity(
                    adjusted_returns, covariance_matrix, symbols, constraints, risk_multiplier
                )
            elif objective == PortfolioObjective.MAXIMIZE_SHARPE:
                allocations = self._optimize_sharpe_ratio(
                    adjusted_returns, covariance_matrix, symbols, constraints, risk_multiplier
                )
            elif objective == PortfolioObjective.MINIMIZE_RISK:
                allocations = self._optimize_minimum_risk(
                    adjusted_returns, covariance_matrix, symbols, constraints, risk_multiplier
                )
            elif objective == PortfolioObjective.TARGET_VOLATILITY:
                allocations = self._optimize_target_volatility(
                    adjusted_returns, covariance_matrix, symbols, constraints, risk_multiplier
                )
            else:  # MAXIMIZE_RETURN or ABSOLUTE_RETURN
                allocations = self._optimize_maximum_return(
                    adjusted_returns, market_view, symbols, constraints
                )
            
            # Calculate portfolio metrics
            portfolio_metrics = self._calculate_portfolio_metrics(
                allocations, covariance_matrix, adjusted_returns
            )
            
            # Determine risk budget allocation
            risk_budget = self._allocate_risk_budget(allocations, covariance_matrix)
            
            # Set execution priority
            execution_priority = self._determine_execution_priority(allocations, market_view)
            
            # Check if rebalancing is required
            rebalance_required = self._check_rebalance_requirement(allocations)
            
            # Create construction plan
            plan = PortfolioConstructionPlan(
                objective=objective,
                target_portfolio_value=self.current_capital,
                current_portfolio_value=self._calculate_current_portfolio_value(),
                allocations=allocations,
                portfolio_metrics=portfolio_metrics,
                risk_budget_allocation=risk_budget,
                execution_priority=execution_priority,
                rebalance_required=rebalance_required
            )
            
            return plan
            
        except Exception as e:
            logger.error(f"Portfolio construction failed: {e}")
            return self._get_default_construction_plan(objective)

    def _optimize_risk_parity(self, expected_returns: Dict[str, float],
                            covariance_matrix: np.ndarray,
                            symbols: List[str],
                            constraints: PortfolioConstraint,
                            risk_multiplier: float) -> List[AssetAllocation]:
        """Optimize for equal risk contribution (risk parity)"""
        try:
            n_assets = len(symbols)
            
            # Risk parity optimization
            def risk_parity_objective(weights):
                # Portfolio risk contributions
                portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
                marginal_contributions = np.dot(covariance_matrix, weights)
                risk_contributions = weights * marginal_contributions
                
                # Target: equal risk contributions
                target_risk = np.mean(risk_contributions)
                deviations = np.sum((risk_contributions - target_risk) ** 2)
                
                return deviations
            
            # Constraints
            bounds = [(constraints.min_position_size, constraints.max_position_size) 
                     for _ in range(n_assets)]
            
            def weight_sum_constraint(weights):
                return np.sum(weights) - 1.0
            
            # Initial guess: equal weights
            initial_weights = np.ones(n_assets) / n_assets
            
            # Solve optimization
            result = minimize(
                risk_parity_objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'eq', 'fun': weight_sum_constraint}
            )
            
            if result.success:
                optimal_weights = result.x
            else:
                # Fallback to equal weights
                optimal_weights = initial_weights
            
            # Create allocations
            allocations = []
            for i, symbol in enumerate(symbols):
                weight = optimal_weights[i]
                dollar_amount = weight * self.current_capital
                
                allocation = AssetAllocation(
                    symbol=symbol,
                    target_weight=weight,
                    target_dollar_amount=dollar_amount,
                    edge_probability=0.5,  # Would come from edge models
                    risk_contribution=0.0,  # Would be calculated
                    expected_return=expected_returns.get(symbol, 0.0),
                    regime_adjustment=risk_multiplier,
                    priority_score=weight  # Higher weight = higher priority
                )
                allocations.append(allocation)
            
            return allocations
            
        except Exception as e:
            logger.error(f"Risk parity optimization failed: {e}")
            return self._create_equal_weight_allocations(symbols)

    def _optimize_sharpe_ratio(self, expected_returns: Dict[str, float],
                             covariance_matrix: np.ndarray,
                             symbols: List[str],
                             constraints: PortfolioConstraint,
                             risk_multiplier: float) -> List[AssetAllocation]:
        """Optimize for maximum Sharpe ratio"""
        try:
            n_assets = len(symbols)
            expected_returns_array = np.array([expected_returns.get(sym, 0.0) for sym in symbols])
            
            # Sharpe ratio maximization
            def sharpe_objective(weights):
                portfolio_return = np.dot(weights, expected_returns_array)
                portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
                portfolio_volatility = np.sqrt(portfolio_variance)
                
                # Sharpe ratio (annualized)
                sharpe = (portfolio_return * 252 - self.risk_free_rate) / (portfolio_volatility * np.sqrt(252) + 1e-8)
                return -sharpe  # Negative because we minimize
            
            # Constraints
            bounds = [(constraints.min_position_size, constraints.max_position_size) 
                     for _ in range(n_assets)]
            
            def weight_sum_constraint(weights):
                return np.sum(weights) - 1.0
            
            # Initial guess
            initial_weights = np.ones(n_assets) / n_assets
            
            # Solve optimization
            result = minimize(
                sharpe_objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'eq', 'fun': weight_sum_constraint}
            )
            
            if result.success:
                optimal_weights = result.x
            else:
                optimal_weights = initial_weights
            
            # Create allocations
            allocations = []
            for i, symbol in enumerate(symbols):
                weight = optimal_weights[i]
                dollar_amount = weight * self.current_capital
                
                allocation = AssetAllocation(
                    symbol=symbol,
                    target_weight=weight,
                    target_dollar_amount=dollar_amount,
                    edge_probability=0.5,
                    risk_contribution=0.0,
                    expected_return=expected_returns_array[i],
                    regime_adjustment=risk_multiplier,
                    priority_score=weight * expected_returns_array[i]  # Weight × Return priority
                )
                allocations.append(allocation)
            
            return allocations
            
        except Exception as e:
            logger.error(f"Sharpe optimization failed: {e}")
            return self._create_equal_weight_allocations(symbols)

    def _optimize_minimum_risk(self, expected_returns: Dict[str, float],
                             covariance_matrix: np.ndarray,
                             symbols: List[str],
                             constraints: PortfolioConstraint,
                             risk_multiplier: float) -> List[AssetAllocation]:
        """Optimize for minimum portfolio risk"""
        try:
            n_assets = len(symbols)
            
            # Minimum variance optimization
            def min_variance_objective(weights):
                portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
                return portfolio_variance
            
            # Constraints
            bounds = [(constraints.min_position_size, constraints.max_position_size) 
                     for _ in range(n_assets)]
            
            def weight_sum_constraint(weights):
                return np.sum(weights) - 1.0
            
            # Initial guess
            initial_weights = np.ones(n_assets) / n_assets
            
            # Solve optimization
            result = minimize(
                min_variance_objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'eq', 'fun': weight_sum_constraint}
            )
            
            if result.success:
                optimal_weights = result.x
            else:
                optimal_weights = initial_weights
            
            # Create allocations
            allocations = []
            for i, symbol in enumerate(symbols):
                weight = optimal_weights[i]
                dollar_amount = weight * self.current_capital
                
                allocation = AssetAllocation(
                    symbol=symbol,
                    target_weight=weight,
                    target_dollar_amount=dollar_amount,
                    edge_probability=0.5,
                    risk_contribution=0.0,
                    expected_return=expected_returns.get(symbol, 0.0),
                    regime_adjustment=risk_multiplier,
                    priority_score=1.0 - weight  # Lower weight = higher priority for rebalancing
                )
                allocations.append(allocation)
            
            return allocations
            
        except Exception as e:
            logger.error(f"Minimum risk optimization failed: {e}")
            return self._create_equal_weight_allocations(symbols)

    def _optimize_target_volatility(self, expected_returns: Dict[str, float],
                                  covariance_matrix: np.ndarray,
                                  symbols: List[str],
                                  constraints: PortfolioConstraint,
                                  risk_multiplier: float) -> List[AssetAllocation]:
        """Optimize for target volatility constraint"""
        try:
            n_assets = len(symbols)
            expected_returns_array = np.array([expected_returns.get(sym, 0.0) for sym in symbols])
            
            target_vol = constraints.target_volatility * risk_multiplier
            
            # Target volatility optimization
            def target_vol_objective(weights):
                portfolio_return = np.dot(weights, expected_returns_array)
                portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
                portfolio_volatility = np.sqrt(portfolio_variance)
                
                # Penalize deviation from target volatility
                vol_deviation = abs(portfolio_volatility - target_vol)
                return -portfolio_return + 10 * vol_deviation  # Maximize return, penalize vol deviation
            
            # Constraints
            bounds = [(constraints.min_position_size, constraints.max_position_size) 
                     for _ in range(n_assets)]
            
            def weight_sum_constraint(weights):
                return np.sum(weights) - 1.0
            
            def volatility_constraint(weights):
                portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
                portfolio_volatility = np.sqrt(portfolio_variance)
                return target_vol - portfolio_volatility  # Must be >= 0
            
            # Initial guess
            initial_weights = np.ones(n_assets) / n_assets
            
            # Solve optimization
            result = minimize(
                target_vol_objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=[
                    {'type': 'eq', 'fun': weight_sum_constraint},
                    {'type': 'ineq', 'fun': volatility_constraint}
                ]
            )
            
            if result.success:
                optimal_weights = result.x
            else:
                optimal_weights = initial_weights
            
            # Create allocations
            allocations = []
            for i, symbol in enumerate(symbols):
                weight = optimal_weights[i]
                dollar_amount = weight * self.current_capital
                
                allocation = AssetAllocation(
                    symbol=symbol,
                    target_weight=weight,
                    target_dollar_amount=dollar_amount,
                    edge_probability=0.5,
                    risk_contribution=0.0,
                    expected_return=expected_returns_array[i],
                    regime_adjustment=risk_multiplier,
                    priority_score=abs(weight - 1/len(symbols))  # Priority based on deviation from equal weight
                )
                allocations.append(allocation)
            
            return allocations
            
        except Exception as e:
            logger.error(f"Target volatility optimization failed: {e}")
            return self._create_equal_weight_allocations(symbols)

    def _optimize_maximum_return(self, expected_returns: Dict[str, float],
                               market_view: Dict[str, float],
                               symbols: List[str],
                               constraints: PortfolioConstraint) -> List[AssetAllocation]:
        """Optimize for maximum expected return"""
        try:
            # Simple approach: weight by edge probability and expected return
            total_score = 0
            scores = {}
            
            for symbol in symbols:
                edge_prob = market_view.get(symbol, 0.5)
                expected_ret = expected_returns.get(symbol, 0.0)
                score = edge_prob * expected_ret
                scores[symbol] = score
                total_score += score
            
            # Create allocations
            allocations = []
            remaining_weight = 1.0
            
            # Sort by score (descending)
            sorted_symbols = sorted(symbols, key=lambda s: scores[s], reverse=True)
            
            for symbol in sorted_symbols:
                if remaining_weight <= 0:
                    break
                    
                # Allocate proportional to score
                if total_score > 0:
                    weight = min(scores[symbol] / total_score, 
                               constraints.max_position_size,
                               remaining_weight)
                else:
                    weight = min(1.0 / len(symbols), constraints.max_position_size)
                
                weight = max(weight, constraints.min_position_size)
                weight = min(weight, remaining_weight)
                
                dollar_amount = weight * self.current_capital
                
                allocation = AssetAllocation(
                    symbol=symbol,
                    target_weight=weight,
                    target_dollar_amount=dollar_amount,
                    edge_probability=market_view.get(symbol, 0.5),
                    risk_contribution=0.0,
                    expected_return=expected_returns.get(symbol, 0.0),
                    regime_adjustment=1.0,
                    priority_score=scores[symbol]
                )
                allocations.append(allocation)
                remaining_weight -= weight
            
            # Distribute remaining weight equally among top assets
            if remaining_weight > 0 and allocations:
                additional_weight = remaining_weight / len(allocations)
                for allocation in allocations:
                    allocation.target_weight += additional_weight
                    allocation.target_dollar_amount = allocation.target_weight * self.current_capital
            
            return allocations
            
        except Exception as e:
            logger.error(f"Maximum return optimization failed: {e}")
            return self._create_equal_weight_allocations(symbols)

    def _calculate_portfolio_metrics(self, allocations: List[AssetAllocation],
                                   covariance_matrix: np.ndarray,
                                   expected_returns: Dict[str, float]) -> Dict[str, float]:
        """Calculate portfolio performance metrics"""
        try:
            weights = np.array([alloc.target_weight for alloc in allocations])
            symbols = [alloc.symbol for alloc in allocations]
            
            # Expected return
            returns_array = np.array([expected_returns.get(sym, 0.0) for sym in symbols])
            expected_return = np.dot(weights, returns_array) * 252  # Annualized
            
            # Portfolio risk
            portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance) * np.sqrt(252)  # Annualized
            
            # Sharpe ratio
            sharpe_ratio = (expected_return - self.risk_free_rate) / (portfolio_volatility + 1e-8)
            
            # Diversification ratio
            individual_vols = np.sqrt(np.diag(covariance_matrix)) * np.sqrt(252)
            weighted_avg_vol = np.dot(weights, individual_vols)
            diversification_ratio = weighted_avg_vol / (portfolio_volatility + 1e-8)
            
            return {
                'expected_return': expected_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio,
                'diversification_ratio': diversification_ratio,
                'portfolio_variance': portfolio_variance
            }
            
        except Exception as e:
            logger.error(f"Portfolio metrics calculation failed: {e}")
            return {
                'expected_return': 0.0,
                'volatility': 0.15,
                'sharpe_ratio': 0.0,
                'diversification_ratio': 1.0,
                'portfolio_variance': 0.0225
            }

    def _allocate_risk_budget(self, allocations: List[AssetAllocation],
                            covariance_matrix: np.ndarray) -> Dict[str, float]:
        """Allocate risk budget across assets"""
        try:
            weights = np.array([alloc.target_weight for alloc in allocations])
            n_assets = len(allocations)
            
            # Calculate risk contributions
            marginal_contributions = np.dot(covariance_matrix, weights)
            risk_contributions = weights * marginal_contributions
            
            # Total portfolio risk
            total_risk = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
            
            # Risk budget allocation
            risk_budget = {}
            for i, alloc in enumerate(allocations):
                if total_risk > 0:
                    budget = abs(risk_contributions[i]) / total_risk
                else:
                    budget = 1.0 / n_assets
                risk_budget[alloc.symbol] = budget
            
            return risk_budget
            
        except Exception:
            # Equal risk budget as fallback
            return {alloc.symbol: 1.0/len(allocations) for alloc in allocations}

    def _determine_execution_priority(self, allocations: List[AssetAllocation],
                                    market_view: Dict[str, float]) -> List[str]:
        """Determine execution priority for portfolio changes"""
        try:
            # Priority based on: existing position deviation + edge probability
            priority_scores = []
            
            for alloc in allocations:
                current_weight = self.current_positions.get(alloc.symbol, {}).get('weight', 0.0)
                weight_deviation = abs(alloc.target_weight - current_weight)
                edge_prob = market_view.get(alloc.symbol, 0.5)
                
                # Higher priority for larger deviations and higher edge probabilities
                priority = weight_deviation * edge_prob
                priority_scores.append((alloc.symbol, priority))
            
            # Sort by priority (descending)
            priority_scores.sort(key=lambda x: x[1], reverse=True)
            return [symbol for symbol, _ in priority_scores]
            
        except Exception:
            # Alphabetical order as fallback
            return sorted([alloc.symbol for alloc in allocations])

    def _check_rebalance_requirement(self, allocations: List[AssetAllocation]) -> bool:
        """Check if portfolio rebalancing is required"""
        try:
            threshold = 0.05  # 5% threshold for rebalancing
            
            for alloc in allocations:
                current_weight = self.current_positions.get(alloc.symbol, {}).get('weight', 0.0)
                if abs(alloc.target_weight - current_weight) > threshold:
                    return True
            
            return False
            
        except Exception:
            return True  # Conservative approach - rebalance if unsure

    def _calculate_current_portfolio_value(self) -> float:
        """Calculate current portfolio value"""
        try:
            portfolio_value = self.current_capital
            # Would add current positions value here
            return portfolio_value
        except Exception:
            return self.current_capital

    def _create_equal_weight_allocations(self, symbols: List[str]) -> List[AssetAllocation]:
        """Create equal weight allocations as fallback"""
        n_assets = len(symbols)
        equal_weight = 1.0 / n_assets if n_assets > 0 else 0.0
        
        allocations = []
        for symbol in symbols:
            allocation = AssetAllocation(
                symbol=symbol,
                target_weight=equal_weight,
                target_dollar_amount=equal_weight * self.current_capital,
                edge_probability=0.5,
                risk_contribution=1.0 / n_assets,
                expected_return=0.0,
                regime_adjustment=1.0,
                priority_score=1.0
            )
            allocations.append(allocation)
        
        return allocations

    def _get_default_construction_plan(self, objective: PortfolioObjective) -> PortfolioConstructionPlan:
        """Return default construction plan when optimization fails"""
        return PortfolioConstructionPlan(
            objective=objective,
            target_portfolio_value=self.current_capital,
            current_portfolio_value=self.current_capital,
            allocations=[],
            portfolio_metrics={
                'expected_return': 0.0,
                'volatility': 0.15,
                'sharpe_ratio': 0.0,
                'diversification_ratio': 1.0
            },
            risk_budget_allocation={},
            execution_priority=[],
            rebalance_required=False
        )

    def update_portfolio_state(self, positions: Dict, current_prices: Dict):
        """Update portfolio state with current positions"""
        try:
            self.current_positions = positions.copy()
            # Update current capital based on positions and prices
            portfolio_value = sum(
                pos.get('size', 0) * current_prices.get(symbol, 0)
                for symbol, pos in positions.items()
            )
            self.current_capital = max(portfolio_value, self.initial_capital * 0.5)  # Minimum 50% of initial
            
        except Exception as e:
            logger.error(f"Portfolio state update failed: {e}")

# Global instance
_portfolio_constructor = None

def get_portfolio_constructor(initial_capital: float = 100000.0) -> TopDownPortfolioConstructor:
    """Get singleton portfolio constructor instance"""
    global _portfolio_constructor
    if _portfolio_constructor is None:
        _portfolio_constructor = TopDownPortfolioConstructor(initial_capital)
    return _portfolio_constructor

def main():
    """Example usage"""
    print("Top-Down Portfolio Constructor ready")
    print("Professional portfolio construction from objectives down to trades")

if __name__ == "__main__":
    main()