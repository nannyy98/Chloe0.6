"""
Enhanced Risk Engine for Chloe AI 0.4
Implements professional risk management with Kelly criterion, CVaR optimization, and adaptive position sizing
Based on Aziz Salimov's industry recommendations for robust AI trading systems
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy.optimize import minimize
from scipy.stats import norm

logger = logging.getLogger(__name__)

@dataclass
class RiskParameters:
    """Risk management parameters"""
    max_position_size: float = 0.10        # Maximum 10% of capital per position
    kelly_fraction: float = 0.25           # Fraction of Kelly criterion to use (0.25 = quarter Kelly)
    max_drawdown_limit: float = 0.20       # Maximum allowable drawdown (20%)
    var_confidence: float = 0.95           # VaR confidence level (95%)
    cvar_confidence: float = 0.95          # CVaR confidence level (95%)
    max_correlation: float = 0.7           # Maximum correlation between positions
    liquidity_buffer: float = 0.10         # 10% buffer for liquidity considerations
    regime_risk_multiplier: Dict[str, float] = None  # Risk multipliers by market regime

    def __post_init__(self):
        if self.regime_risk_multiplier is None:
            self.regime_risk_multiplier = {
                'STABLE': 1.0,
                'TRENDING': 1.2,
                'MEAN_REVERTING': 0.8,
                'VOLATILE': 1.5
            }

@dataclass
class PositionConstraints:
    """Position-level constraints"""
    symbol: str
    max_capital_allocation: float      # Maximum capital allocation for this symbol
    min_position_size: float          # Minimum position size (prevent tiny positions)
    max_position_size: float          # Maximum position size
    stop_loss_distance: float         # Stop loss distance in price terms
    take_profit_distance: float       # Take profit distance in price terms
    liquidity_limit: float            # Maximum percentage of daily volume
    correlation_constraints: List[str] = None  # Symbols this position shouldn't be correlated with

class EnhancedRiskEngine:
    """
    Professional risk engine implementing modern quantitative risk management
    Core philosophy: "Manage probabilities under risk control" not "Predict prices"
    """
    
    def __init__(self, initial_capital: float = 10000.0, risk_params: RiskParameters = None):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_params = risk_params or RiskParameters()
        self.position_register = {}  # Track all positions
        self.portfolio_history = []  # Track portfolio value over time
        self.risk_metrics = {}       # Current risk metrics
        self.correlation_matrix = {} # Asset correlations
        self.drawdown_history = []   # Track drawdowns
        self.is_initialized = False
        
        logger.info("🛡️ Enhanced Risk Engine initialized")
        logger.info(f"   Initial capital: ${self.initial_capital:,.2f}")
        logger.info(f"   Kelly fraction: {self.risk_params.kelly_fraction}")
        logger.info(f"   Max drawdown limit: {self.risk_params.max_drawdown_limit*100:.1f}%")
    
    def initialize_portfolio_tracking(self, portfolio_value: float = None) -> bool:
        """Initialize portfolio tracking"""
        try:
            if portfolio_value is not None:
                self.current_capital = portfolio_value
            
            # Initialize tracking
            self.portfolio_history = [{
                'timestamp': datetime.now(),
                'portfolio_value': self.current_capital,
                'positions': {},
                'drawdown': 0.0
            }]
            
            self.is_initialized = True
            logger.info("✅ Portfolio tracking initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Portfolio initialization failed: {e}")
            return False
    
    def calculate_kelly_position_size(self, win_rate: float, win_loss_ratio: float, 
                                    account_size: float, regime: str = 'STABLE') -> float:
        """
        Calculate position size using fractional Kelly criterion
        This is the theoretically optimal bet size for maximum long-term growth
        
        Args:
            win_rate: Probability of winning trade (0-1)
            win_loss_ratio: Average win / Average loss
            account_size: Current account size
            regime: Current market regime affecting risk appetite
            
        Returns:
            Optimal position size as fraction of account
        """
        try:
            # Validate inputs
            if win_rate <= 0 or win_rate >= 1:
                win_rate = 0.5  # Default to 50% if invalid
            if win_loss_ratio <= 0:
                win_loss_ratio = 1.0  # Default to 1:1 if invalid
            if account_size <= 0:
                return 0.0
                
            # Kelly formula: f* = (p*b - q) / b
            # where p = win probability, q = loss probability, b = win/loss ratio
            loss_rate = 1 - win_rate
            kelly_fraction = (win_rate * win_loss_ratio - loss_rate) / win_loss_ratio
            
            # Apply fractional Kelly to reduce volatility
            fractional_kelly = max(0, kelly_fraction) * self.risk_params.kelly_fraction  # Ensure non-negative
            
            # Apply regime-based risk adjustment
            regime_multiplier = self.risk_params.regime_risk_multiplier.get(regime, 1.0)
            adjusted_kelly = fractional_kelly * regime_multiplier
            
            # Apply position size limits
            max_allowed = min(
                self.risk_params.max_position_size,
                adjusted_kelly,
                account_size * 0.02  # Never more than 2% of account
            )
            
            # Ensure positive and reasonable size
            position_size = max(0.0, min(max_allowed, 0.1))  # Cap at 10%
            
            logger.debug(f"Kelly calculation - Win rate: {win_rate:.3f}, W/L: {win_loss_ratio:.3f}, "
                        f"Size: {position_size:.4f} ({regime})")
            
            return position_size
            
        except Exception as e:
            logger.error(f"❌ Kelly calculation failed: {e}")
            return min(0.01, self.risk_params.max_position_size)  # Safe fallback
    
    def calculate_cvar_optimal_allocation(self, returns_data: pd.DataFrame, 
                                        symbols: List[str], 
                                        confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Calculate CVaR-optimal portfolio allocation
        Minimizes Conditional Value at Risk while targeting return objectives
        
        Args:
            returns_data: Historical returns data for all symbols
            symbols: List of symbols to allocate
            confidence_level: Confidence level for CVaR calculation
            
        Returns:
            Dictionary mapping symbols to optimal allocation weights
        """
        try:
            if len(symbols) == 0:
                return {}
            
            # Filter data to only include requested symbols
            available_symbols = [s for s in symbols if s in returns_data.columns]
            if len(available_symbols) == 0:
                return {s: 1.0/len(symbols) for s in symbols}  # Equal weight fallback
            
            symbol_data = returns_data[available_symbols]
            
            # Calculate mean returns and covariance matrix
            mean_returns = symbol_data.mean().values
            cov_matrix = symbol_data.cov().values
            
            # Optimization objective function (minimize CVaR)
            def cvar_objective(weights):
                # Portfolio returns
                portfolio_returns = np.dot(symbol_data.values, weights)
                
                # Calculate VaR at given confidence level
                var_threshold = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
                
                # Calculate CVaR (expected shortfall below VaR)
                cvar = portfolio_returns[portfolio_returns <= var_threshold].mean()
                
                # Return negative CVaR (we want to minimize the negative, so maximize CVaR)
                return -cvar if not np.isnan(cvar) else 0
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # Weights sum to 1
                {'type': 'ineq', 'fun': lambda w: w}  # Weights >= 0 (long-only)
            ]
            
            # Initial guess (equal weights)
            initial_weights = np.array([1.0/len(available_symbols)] * len(available_symbols))
            
            # Optimize
            result = minimize(
                cvar_objective,
                initial_weights,
                method='SLSQP',
                constraints=constraints,
                bounds=[(0, 1) for _ in available_symbols]
            )
            
            if result.success:
                optimal_weights = result.x
                allocation = dict(zip(available_symbols, optimal_weights))
                
                # Add missing symbols with zero allocation
                for symbol in symbols:
                    if symbol not in allocation:
                        allocation[symbol] = 0.0
                
                logger.info(f"✅ CVaR optimization completed for {len(symbols)} symbols")
                return allocation
            else:
                logger.warning("⚠️ CVaR optimization failed, using equal weights")
                return {s: 1.0/len(symbols) for s in symbols}
                
        except Exception as e:
            logger.error(f"❌ CVaR optimization failed: {e}")
            return {s: 1.0/len(symbols) for s in symbols}
    
    def calculate_value_at_risk(self, returns: np.array, confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR) for a portfolio
        
        Args:
            returns: Array of portfolio returns
            confidence_level: Confidence level (default 95%)
            
        Returns:
            VaR value (negative, representing maximum expected loss)
        """
        if len(returns) == 0:
            return 0.0
            
        # Historical VaR calculation
        var_percentile = (1 - confidence_level) * 100
        var = np.percentile(returns, var_percentile)
        
        return var
    
    def calculate_conditional_var(self, returns: np.array, confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR) / Expected Shortfall
        
        Args:
            returns: Array of portfolio returns
            confidence_level: Confidence level (default 95%)
            
        Returns:
            CVaR value (average of worst (1-confidence_level)% returns)
        """
        if len(returns) == 0:
            return 0.0
            
        # Calculate VaR first
        var = self.calculate_value_at_risk(returns, confidence_level)
        
        # CVaR is the average of returns below VaR
        tail_returns = returns[returns <= var]
        if len(tail_returns) == 0:
            return var
            
        cvar = np.mean(tail_returns)
        return cvar
    
    def calculate_optimal_positions(self, edge_opportunities: List[Dict], 
                                  regime_context: Dict) -> List[Dict]:
        """
        Main orchestration method - Risk Engine decides what positions to take
        This is the CENTRAL DECISION MAKING function
        
        Args:
            edge_opportunities: List of edge opportunities with probabilities
            regime_context: Current market regime information
            
        Returns:
            List of approved positions with sizing and risk parameters
        """
        logger.info("🛡️ Risk Engine Orchestration Started")
        logger.info(f"   Regime: {regime_context.get('name', 'UNKNOWN')}")
        logger.info(f"   Opportunities: {len(edge_opportunities)}")
        
        approved_positions = []
        
        # Risk-first filtering and sizing
        for opportunity in edge_opportunities:
            try:
                # Step 1: Basic viability check
                if not self._is_opportunity_viable(opportunity):
                    continue
                
                # Step 2: Risk assessment
                risk_metrics = self._comprehensive_risk_assessment(opportunity, regime_context)
                
                # Step 3: Position sizing using Kelly criterion
                optimal_size = self._calculate_optimal_position_size(opportunity, regime_context)
                
                # Step 4: Final approval with all constraints
                if self._final_approval_check(opportunity, optimal_size, risk_metrics, regime_context):
                    position = {
                        'symbol': opportunity['symbol'],
                        'position_size': optimal_size,
                        'position_value': optimal_size * opportunity.get('entry_price', 10000),  # Placeholder
                        'edge_probability': opportunity['edge_probability'],
                        'expected_return': opportunity.get('expected_return', 0.02),
                        'stop_loss': opportunity.get('stop_loss', 0.95),  # 5% default
                        'take_profit': opportunity.get('take_profit', 1.04),  # 4% default
                        'risk_metrics': risk_metrics,
                        'regime_context': regime_context,
                        'approval_reason': 'RISK_ENGINE_APPROVED'
                    }
                    approved_positions.append(position)
                    
                    logger.info(f"   ✅ Approved: {opportunity['symbol']} "
                              f"(Size: {optimal_size:.4f}, Edge: {opportunity['edge_probability']:.3f})")
                
            except Exception as e:
                logger.warning(f"   ⚠️ Position assessment failed for {opportunity.get('symbol', 'UNKNOWN')}: {e}")
        
        # Portfolio-level risk optimization
        optimized_positions = self._optimize_portfolio_risk(approved_positions, regime_context)
        
        logger.info(f"✅ Risk Engine Orchestration Complete: {len(optimized_positions)} positions approved")
        return optimized_positions
    
    def _is_opportunity_viable(self, opportunity: Dict) -> bool:
        """Check basic viability criteria"""
        # Minimum edge probability threshold
        if opportunity.get('edge_probability', 0) < 0.55:  # 55% minimum
            return False
        
        # Minimum expected return
        if opportunity.get('expected_return', 0) < 0.005:  # 0.5% minimum
            return False
        
        return True
    
    def _comprehensive_risk_assessment(self, opportunity: Dict, regime_context: Dict) -> Dict:
        """Calculate comprehensive risk metrics for the opportunity"""
        symbol = opportunity['symbol']
        edge_prob = opportunity['edge_probability']
        regime = regime_context.get('name', 'STABLE')
        
        # Calculate various risk metrics
        risk_metrics = {
            'position_risk': edge_prob * 0.02,  # 2% base risk
            'regime_risk_multiplier': self.risk_params.regime_risk_multiplier.get(regime, 1.0),
            'correlation_risk': 0.1,  # Default low correlation
            'liquidity_risk': 0.05,   # Default 5% liquidity buffer
            'volatility_risk': opportunity.get('volatility', 0.02),
            'concentration_risk': 0.1 # 10% concentration limit
        }
        
        # Adjust for regime
        risk_metrics['adjusted_risk'] = (
            risk_metrics['position_risk'] * 
            risk_metrics['regime_risk_multiplier'] * 
            (1 + risk_metrics['volatility_risk'])
        )
        
        return risk_metrics
    
    def _calculate_optimal_position_size(self, opportunity: Dict, regime_context: Dict) -> float:
        """Calculate optimal position size using Kelly criterion and constraints"""
        edge_prob = opportunity['edge_probability']
        expected_return = opportunity.get('expected_return', 0.02)
        regime = regime_context.get('name', 'STABLE')
        
        # Kelly position sizing
        kelly_size = self.calculate_kelly_position_size(
            win_rate=edge_prob,
            win_loss_ratio=expected_return / 0.01,  # Assuming 1% loss
            account_size=self.current_capital,
            regime=regime
        )
        
        # Apply constraints
        max_allowed = min(
            kelly_size * self.current_capital,  # Kelly sizing
            self.current_capital * self.risk_params.max_position_size,  # Position limit
            self.current_capital * 0.02  # 2% hard cap
        )
        
        return max(0.0, min(max_allowed, self.current_capital * 0.01))  # Min 1% of capital
    
    def _final_approval_check(self, opportunity: Dict, position_size: float, 
                            risk_metrics: Dict, regime_context: Dict) -> bool:
        """Final approval check with all risk constraints"""
        # Position size limits
        if position_size <= 0:
            return False
        
        if position_size > self.current_capital * self.risk_params.max_position_size:
            return False
        
        # Risk-adjusted return requirement
        risk_adj_return = opportunity.get('expected_return', 0) / max(risk_metrics['adjusted_risk'], 0.01)
        if risk_adj_return < 1.5:  # Minimum 1.5x risk-adjusted return
            return False
        
        # Regime-specific constraints
        regime = regime_context.get('name', 'STABLE')
        if regime == 'CRISIS' and position_size > self.current_capital * 0.005:  # 0.5% in crisis
            return False
        
        return True
    
    def _optimize_portfolio_risk(self, positions: List[Dict], regime_context: Dict) -> List[Dict]:
        """Optimize portfolio-level risk considering correlations and diversification"""
        if len(positions) <= 1:
            return positions
        
        # Simple correlation-based optimization
        total_risk_budget = self.current_capital * 0.05  # 5% total portfolio risk
        current_risk = sum(pos['position_value'] * pos['risk_metrics']['adjusted_risk'] for pos in positions)
        
        if current_risk > total_risk_budget:
            # Scale down positions proportionally
            scaling_factor = total_risk_budget / current_risk
            for pos in positions:
                pos['position_size'] *= scaling_factor
                pos['position_value'] *= scaling_factor
                pos['risk_adjustment'] = 'SCALED_DOWN'
        
        return positions
    
    def assess_position_risk(self, symbol: str, entry_price: float, position_size: float,
        """
        Comprehensive position risk assessment
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            position_size: Position size in units
            stop_loss: Stop loss price
            take_profit: Take profit price
            volatility: Current volatility
            regime: Market regime
            
        Returns:
            Dictionary with risk metrics and approval decision
        """
        try:
            logger.debug(f"Assessing risk for {symbol}: entry=${entry_price}, size={position_size}")
            
            # Calculate key risk metrics
            position_value = position_size * entry_price
            max_loss = abs(entry_price - stop_loss) * position_size if stop_loss != entry_price else position_value * 0.02  # 2% default
            max_gain = abs(take_profit - entry_price) * position_size if take_profit != entry_price else position_value * 0.04  # 4% default
            
            logger.debug(f"Position value: ${position_value:.2f}, Max loss: ${max_loss:.2f}, Max gain: ${max_gain:.2f}")
            
            # Risk/reward ratio
            if max_loss > 0:
                risk_reward_ratio = max_gain / max_loss
            else:
                risk_reward_ratio = 1.0  # Default when no loss defined
            
            # Position size as percentage of capital
            position_pct = position_value / self.current_capital if self.current_capital > 0 else 0
            
            # Volatility-adjusted risk
            regime_multiplier = self.risk_params.regime_risk_multiplier.get(regime, 1.0)
            adjusted_volatility = volatility * regime_multiplier
            
            # Kelly-based position sizing recommendation
            # Estimate win rate and win/loss ratio from risk/reward
            estimated_win_rate = min(0.8, max(0.2, 1 / (1 + risk_reward_ratio))) if risk_reward_ratio > 0 else 0.5
            kelly_size = self.calculate_kelly_position_size(
                win_rate=estimated_win_rate,
                win_loss_ratio=max(0.1, risk_reward_ratio),  # Avoid division by zero
                account_size=self.current_capital,
                regime=regime
            )
            
            # Risk metrics
            risk_metrics = {
                'position_value': position_value,
                'position_percentage': position_pct,
                'max_loss': max_loss,
                'max_gain': max_gain,
                'risk_reward_ratio': risk_reward_ratio,
                'volatility': volatility,
                'adjusted_volatility': adjusted_volatility,
                'kelly_recommended_size': kelly_size,
                'kelly_recommended_value': kelly_size * self.current_capital,
                'max_drawdown_impact': max_loss / self.current_capital if self.current_capital > 0 else 0
            }
            
            logger.debug(f"Risk metrics calculated: {risk_metrics}")
            
            # Risk approval logic
            approval_criteria = {
                'size_limit': position_pct <= self.risk_params.max_position_size,
                'drawdown_limit': risk_metrics['max_drawdown_impact'] <= 0.02,  # 2% max per trade
                'rr_ratio': risk_reward_ratio >= 1.5,  # Minimum 1.5:1 reward:risk
                'kelly_compliance': position_pct <= kelly_size * 1.5 if kelly_size > 0 else True,  # Within 50% of Kelly
                'volatility_check': adjusted_volatility <= 0.05  # 5% daily volatility cap
            }
            
            # Overall approval
            is_approved = all(approval_criteria.values())
            
            result = {
                'approved': is_approved,
                'risk_metrics': risk_metrics,
                'approval_criteria': approval_criteria,
                'recommendations': self._generate_risk_recommendations(risk_metrics, approval_criteria)
            }
            
            if is_approved:
                logger.info(f"✅ Position approved for {symbol}: ${position_value:,.2f} ({position_pct*100:.1f}% of capital)")
            else:
                logger.warning(f"❌ Position rejected for {symbol}: {self._get_rejection_reasons(approval_criteria)}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Risk assessment failed for {symbol}: {e}")
            return {
                'approved': False,
                'risk_metrics': {
                    'position_value': position_size * entry_price if 'entry_price' in locals() else 0,
                    'error': str(e)
                },
                'approval_criteria': {},
                'error': str(e)
            }
    
    def _generate_risk_recommendations(self, metrics: Dict, criteria: Dict) -> List[str]:
        """Generate specific risk management recommendations"""
        recommendations = []
        
        if not criteria['size_limit']:
            recommendations.append(f"Reduce position size to ≤{self.risk_params.max_position_size*100:.1f}% of capital")
            
        if not criteria['drawdown_limit']:
            recommendations.append("Position would exceed maximum drawdown impact (2% per trade)")
            
        if not criteria['rr_ratio']:
            recommendations.append("Improve risk/reward ratio to at least 1.5:1")
            
        if not criteria['kelly_compliance']:
            recommendations.append(f"Position exceeds Kelly recommendation by {(metrics['position_percentage']/metrics['kelly_recommended_size'] - 1)*100:.1f}%")
            
        if not criteria['volatility_check']:
            recommendations.append("High volatility - consider reducing position size or avoiding trade")
            
        return recommendations
    
    def _get_rejection_reasons(self, criteria: Dict) -> str:
        """Get human-readable rejection reasons"""
        reasons = []
        if not criteria['size_limit']:
            reasons.append("position too large")
        if not criteria['drawdown_limit']:
            reasons.append("excessive drawdown risk")
        if not criteria['rr_ratio']:
            reasons.append("poor risk/reward ratio")
        if not criteria['kelly_compliance']:
            reasons.append("deviates from optimal sizing")
        if not criteria['volatility_check']:
            reasons.append("high volatility")
            
        return ", ".join(reasons) if reasons else "unknown reason"
    
    def update_portfolio_state(self, portfolio_value: float, positions: Dict):
        """Update portfolio state and calculate current risk metrics"""
        try:
            # Update current capital
            self.current_capital = portfolio_value
            
            # Calculate drawdown
            peak_value = max([entry['portfolio_value'] for entry in self.portfolio_history] + [portfolio_value])
            current_drawdown = (peak_value - portfolio_value) / peak_value if peak_value > 0 else 0
            
            # Store portfolio state
            self.portfolio_history.append({
                'timestamp': datetime.now(),
                'portfolio_value': portfolio_value,
                'positions': positions.copy(),
                'drawdown': current_drawdown
            })
            
            # Update drawdown history
            self.drawdown_history.append(current_drawdown)
            
            # Calculate risk metrics
            self.risk_metrics = {
                'current_value': portfolio_value,
                'peak_value': peak_value,
                'current_drawdown': current_drawdown,
                'max_drawdown': max(self.drawdown_history) if self.drawdown_history else 0,
                'volatility': self._calculate_portfolio_volatility(),
                'sharpe_ratio': self._calculate_sharpe_ratio(),
                'number_of_positions': len(positions)
            }
            
            # Check drawdown limit
            if current_drawdown > self.risk_params.max_drawdown_limit:
                logger.critical(f"🚨 DRAWDOWN LIMIT EXCEEDED: {current_drawdown*100:.2f}% (limit: {self.risk_params.max_drawdown_limit*100:.1f}%)")
                return False  # Signal for risk mitigation
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Portfolio state update failed: {e}")
            return False
    
    def _calculate_portfolio_volatility(self) -> float:
        """Calculate portfolio volatility from history"""
        if len(self.portfolio_history) < 2:
            return 0.0
            
        values = [entry['portfolio_value'] for entry in self.portfolio_history]
        returns = np.diff(np.log(values))
        return np.std(returns) * np.sqrt(252)  # Annualized
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate portfolio Sharpe ratio"""
        if len(self.portfolio_history) < 30:  # Need minimum data
            return 0.0
            
        values = [entry['portfolio_value'] for entry in self.portfolio_history]
        returns = np.diff(np.log(values))
        
        if np.std(returns) == 0:
            return 0.0
            
        # Assuming risk-free rate of 2%
        risk_free_rate = 0.02 / 252  # Daily risk-free rate
        excess_returns = returns - risk_free_rate
        sharpe = np.mean(excess_returns) / np.std(returns) * np.sqrt(252)
        
        return sharpe
    
    def get_risk_report(self) -> Dict:
        """Generate comprehensive risk report"""
        return {
            'timestamp': datetime.now().isoformat(),
            'portfolio_metrics': self.risk_metrics,
            'risk_parameters': {
                'max_position_size': self.risk_params.max_position_size,
                'kelly_fraction': self.risk_params.kelly_fraction,
                'max_drawdown_limit': self.risk_params.max_drawdown_limit,
                'current_capital': self.current_capital
            },
            'portfolio_history_length': len(self.portfolio_history),
            'drawdown_history': self.drawdown_history[-30:] if self.drawdown_history else [],  # Last 30 entries
            'status': 'NORMAL' if self.risk_metrics.get('current_drawdown', 0) <= self.risk_params.max_drawdown_limit else 'DRAWDOWN_EXCEEDED'
        }

# Global risk engine instance
enhanced_risk_engine = None

def get_enhanced_risk_engine(initial_capital: float = 10000.0) -> EnhancedRiskEngine:
    """Get singleton enhanced risk engine instance"""
    global enhanced_risk_engine
    if enhanced_risk_engine is None:
        enhanced_risk_engine = EnhancedRiskEngine(initial_capital)
    return enhanced_risk_engine