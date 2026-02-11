"""
Enhanced Risk Engine for Chloe AI 0.4
Professional risk management with Kelly criterion, CVaR optimization, and regime-aware calibration
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy import stats
from collections import deque

logger = logging.getLogger(__name__)

@dataclass 
class RiskMetrics:
    """Comprehensive risk metrics"""
    var_95: float
    cvar_95: float
    max_drawdown: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    beta: float
    correlation_risk: float
    liquidity_risk: float
    regime_risk_multiplier: float

@dataclass
class PositionRisk:
    """Detailed position risk assessment"""
    approved: bool
    risk_metrics: Dict
    rejection_reason: Optional[str] = None

class EnhancedRiskEngine:
    """Professional risk engine with advanced risk management"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.historical_returns = deque(maxlen=252)  # 1 year daily data
        self.risk_free_rate = 0.02  # 2% annual
        
        # Risk parameters
        self.max_position_size = 0.15  # 15% max per position
        self.max_portfolio_exposure = 0.8  # 80% max exposure
        self.max_correlation_risk = 0.6   # Max correlation between positions
        self.target_var = 0.02  # 2% VaR target
        self.stop_loss_percent = 0.05  # 5% stop loss
        
        # Regime-aware parameters
        self.regime_parameters = {
            'STABLE': {'risk_multiplier': 1.0, 'position_size_adj': 1.0},
            'TRENDING': {'risk_multiplier': 1.2, 'position_size_adj': 1.1},
            'VOLATILE': {'risk_multiplier': 0.7, 'position_size_adj': 0.8},
            'MEAN_REVERTING': {'risk_multiplier': 0.8, 'position_size_adj': 0.9},
            'CRISIS': {'risk_multiplier': 0.5, 'position_size_adj': 0.6}
        }
        
        logger.info(f"Enhanced Risk Engine initialized with ${initial_capital:,.2f}")

    def calculate_kelly_position_size(self, win_rate: float, win_loss_ratio: float, 
                                    account_size: float, regime: str = 'STABLE') -> float:
        """Calculate optimal position size using Kelly criterion with regime adjustment"""
        try:
            if win_loss_ratio <= 0 or win_rate <= 0 or win_rate >= 1:
                return account_size * 0.01  # Default small position
            
            # Kelly formula: f* = (bp - q) / b
            # where b = win_loss_ratio, p = win_rate, q = 1 - win_rate
            kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
            
            # Apply regime adjustments
            regime_params = self.regime_parameters.get(regime.upper(), 
                                                     self.regime_parameters['STABLE'])
            regime_multiplier = regime_params['risk_multiplier']
            
            # Conservative Kelly - use fraction of full Kelly
            adjusted_kelly = max(0.0, min(1.0, kelly_fraction * regime_multiplier * 0.25))
            
            position_size = adjusted_kelly * account_size
            max_allowed = account_size * self.max_position_size * regime_params['position_size_adj']
            
            return min(position_size, max_allowed)
            
        except Exception as e:
            logger.error(f"Kelly calculation error: {e}")
            return account_size * 0.01

    def calculate_value_at_risk(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
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

    def calculate_conditional_var(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
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

    def assess_position_risk(self, symbol: str, entry_price: float, position_size: float,
                           stop_loss: float, take_profit: float, volatility: float, 
                           regime: str = 'STABLE') -> PositionRisk:
        """Comprehensive position risk assessment"""
        try:
            # Get regime parameters
            regime_params = self.regime_parameters.get(regime.upper(), 
                                                     self.regime_parameters['STABLE'])
            risk_multiplier = regime_params['risk_multiplier']
            
            # Calculate basic metrics
            risk_amount = abs(entry_price - stop_loss) * position_size
            reward_amount = abs(take_profit - entry_price) * position_size
            risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
            
            # Position sizing check
            position_value = entry_price * position_size
            position_percentage = (position_value / self.current_capital) * 100
            
            # Risk metrics
            stop_loss_percent = abs(entry_price - stop_loss) / entry_price
            take_profit_percent = abs(take_profit - entry_price) / entry_price
            
            # Volatility check (adjusted for regime)
            max_acceptable_vol = 0.15 * risk_multiplier  # Base 15% volatility limit
            volatility_check = volatility <= max_acceptable_vol
            
            # Risk-reward check
            min_rr_ratio = 1.5 * risk_multiplier  # Minimum risk-reward ratio
            risk_reward_check = risk_reward_ratio >= min_rr_ratio
            
            # Position size check
            max_position_pct = self.max_position_size * 100 * regime_params['position_size_adj']
            position_size_check = position_percentage <= max_position_pct
            
            # Portfolio exposure check
            current_exposure = self._calculate_current_exposure() + position_percentage
            exposure_check = current_exposure <= (self.max_portfolio_exposure * 100)
            
            # Correlation risk check
            correlation_risk = self._calculate_correlation_risk(symbol)
            correlation_check = correlation_risk <= (self.max_correlation_risk * risk_multiplier)
            
            # Final approval logic
            approved = (
                volatility_check and
                risk_reward_check and
                position_size_check and
                exposure_check and
                correlation_check and
                risk_multiplier > 0.3  # Minimum regime confidence
            )
            
            risk_metrics = {
                'position_value': position_value,
                'position_percentage': position_percentage,
                'risk_amount': risk_amount,
                'reward_amount': reward_amount,
                'risk_reward_ratio': risk_reward_ratio,
                'volatility': volatility,
                'regime_multiplier': risk_multiplier,
                'stop_loss_distance': stop_loss_percent * 100,
                'take_profit_distance': take_profit_percent * 100,
                'current_exposure': current_exposure,
                'correlation_risk': correlation_risk
            }
            
            if not approved:
                reasons = []
                if not volatility_check:
                    reasons.append(f"High volatility ({volatility:.1%} > {max_acceptable_vol:.1%})")
                if not risk_reward_check:
                    reasons.append(f"Poor risk-reward ({risk_reward_ratio:.2f} < {min_rr_ratio:.2f})")
                if not position_size_check:
                    reasons.append(f"Position too large ({position_percentage:.1f}% > {max_position_pct:.1f}%)")
                if not exposure_check:
                    reasons.append(f"Exceeds exposure limit ({current_exposure:.1f}% > {self.max_portfolio_exposure*100:.1f}%)")
                if not correlation_check:
                    reasons.append(f"High correlation risk ({correlation_risk:.2f})")
                if risk_multiplier <= 0.3:
                    reasons.append("Unfavorable market regime")
                
                rejection_reason = "; ".join(reasons)
            else:
                rejection_reason = None
            
            return PositionRisk(
                approved=approved,
                risk_metrics=risk_metrics,
                rejection_reason=rejection_reason
            )
            
        except Exception as e:
            logger.error(f"Risk assessment error for {symbol}: {e}")
            return PositionRisk(
                approved=False,
                risk_metrics={},
                rejection_reason="System error"
            )

    def update_portfolio_state(self, positions: Dict, current_prices: Dict):
        """Update portfolio state with current positions and prices"""
        try:
            self.positions = positions.copy()
            portfolio_value = self._calculate_portfolio_value(current_prices)
            self.current_capital = portfolio_value
            
            # Update historical returns if we have enough data
            if len(positions) > 0:
                returns = self._calculate_portfolio_returns(positions, current_prices)
                if returns != 0:
                    self.historical_returns.append(returns)
                    
        except Exception as e:
            logger.error(f"Portfolio state update error: {e}")

    def get_portfolio_risk_metrics(self) -> RiskMetrics:
        """Calculate comprehensive portfolio risk metrics"""
        try:
            returns_array = np.array(list(self.historical_returns))
            
            if len(returns_array) < 10:
                # Return default metrics for insufficient data
                return RiskMetrics(
                    var_95=0.02,
                    cvar_95=0.03,
                    max_drawdown=0.0,
                    volatility=0.15,
                    sharpe_ratio=0.0,
                    sortino_ratio=0.0,
                    beta=1.0,
                    correlation_risk=0.5,
                    liquidity_risk=0.3,
                    regime_risk_multiplier=1.0
                )
            
            # Calculate metrics
            var_95 = self.calculate_value_at_risk(returns_array, 0.95)
            cvar_95 = self.calculate_conditional_var(returns_array, 0.95)
            volatility = np.std(returns_array) * np.sqrt(252)  # Annualized
            max_drawdown = self._calculate_max_drawdown(returns_array)
            
            # Sharpe ratio
            excess_return = np.mean(returns_array) * 252 - self.risk_free_rate
            sharpe_ratio = excess_return / (volatility if volatility > 0 else 0.01)
            
            # Sortino ratio (using downside deviation)
            negative_returns = returns_array[returns_array < 0]
            downside_deviation = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else 0.01
            sortino_ratio = excess_return / downside_deviation
            
            # Beta (simplified - would use market benchmark in practice)
            beta = 1.0
            
            # Other risk metrics
            correlation_risk = self._calculate_portfolio_correlation_risk()
            liquidity_risk = self._calculate_liquidity_risk()
            regime_risk_multiplier = self._get_current_regime_multiplier()
            
            return RiskMetrics(
                var_95=var_95,
                cvar_95=cvar_95,
                max_drawdown=max_drawdown,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                beta=beta,
                correlation_risk=correlation_risk,
                liquidity_risk=liquidity_risk,
                regime_risk_multiplier=regime_risk_multiplier
            )
            
        except Exception as e:
            logger.error(f"Risk metrics calculation error: {e}")
            return RiskMetrics(
                var_95=0.05,
                cvar_95=0.08,
                max_drawdown=0.1,
                volatility=0.2,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                beta=1.0,
                correlation_risk=0.5,
                liquidity_risk=0.5,
                regime_risk_multiplier=1.0
            )

    def _calculate_current_exposure(self) -> float:
        """Calculate current portfolio exposure percentage"""
        try:
            if not self.positions:
                return 0.0
            
            total_exposure = sum(
                pos.get('size', 0) * pos.get('entry_price', 0) 
                for pos in self.positions.values()
            )
            return (total_exposure / self.current_capital) * 100
            
        except Exception:
            return 0.0

    def _calculate_correlation_risk(self, symbol: str) -> float:
        """Calculate correlation risk with existing positions"""
        try:
            if not self.positions:
                return 0.0
            
            # Simplified correlation calculation
            # In practice, would use actual price correlations
            symbols = list(self.positions.keys()) + [symbol]
            unique_assets = len(set(symbols))
            total_assets = len(symbols)
            
            # Higher correlation when assets are similar
            correlation_risk = 1.0 - (unique_assets / total_assets)
            return min(1.0, correlation_risk)
            
        except Exception:
            return 0.5

    def _calculate_portfolio_value(self, current_prices: Dict) -> float:
        """Calculate current portfolio value"""
        try:
            portfolio_value = self.current_capital  # Cash
            
            for symbol, position in self.positions.items():
                if symbol in current_prices:
                    size = position.get('size', 0)
                    current_value = size * current_prices[symbol]
                    portfolio_value += current_value
                    
            return portfolio_value
            
        except Exception:
            return self.current_capital

    def _calculate_portfolio_returns(self, positions: Dict, current_prices: Dict) -> float:
        """Calculate portfolio returns for risk metrics"""
        try:
            if not positions:
                return 0.0
            
            total_return = 0.0
            total_investment = 0.0
            
            for symbol, position in positions.items():
                if symbol in current_prices:
                    size = position.get('size', 0)
                    entry_price = position.get('entry_price', current_prices.get(symbol, 0))
                    current_price = current_prices[symbol]
                    
                    investment = size * entry_price
                    current_value = size * current_price
                    position_return = (current_value - investment) / investment if investment > 0 else 0
                    
                    total_return += position_return * investment
                    total_investment += investment
            
            return total_return / total_investment if total_investment > 0 else 0.0
            
        except Exception:
            return 0.0

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown from returns"""
        try:
            if len(returns) == 0:
                return 0.0
            
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            return abs(np.min(drawdown))
            
        except Exception:
            return 0.0

    def _calculate_portfolio_correlation_risk(self) -> float:
        """Calculate overall portfolio correlation risk"""
        try:
            if len(self.positions) <= 1:
                return 0.0
            
            # Simplified calculation - in practice would use correlation matrix
            return min(0.8, len(self.positions) * 0.1)
            
        except Exception:
            return 0.5

    def _calculate_liquidity_risk(self) -> float:
        """Calculate portfolio liquidity risk"""
        try:
            if not self.positions:
                return 0.0
            
            # Assess based on position sizes and asset types
            large_positions = sum(1 for pos in self.positions.values() 
                                if pos.get('size', 0) * pos.get('entry_price', 0) > self.current_capital * 0.05)
            
            liquidity_risk = min(1.0, large_positions * 0.2)
            return liquidity_risk
            
        except Exception:
            return 0.3

    def _get_current_regime_multiplier(self) -> float:
        """Get current regime risk multiplier (placeholder)"""
        # Would integrate with regime detection system
        return 1.0

# Global instance
_enhanced_risk_engine = None

def get_enhanced_risk_engine(initial_capital: float = 100000.0) -> EnhancedRiskEngine:
    """Get singleton enhanced risk engine instance"""
    global _enhanced_risk_engine
    if _enhanced_risk_engine is None:
        _enhanced_risk_engine = EnhancedRiskEngine(initial_capital)
    return _enhanced_risk_engine

if __name__ == "__main__":
    print("Enhanced Risk Engine ready for professional trading")