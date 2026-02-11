"""
Simple Working Risk Engine for Integration Demo
Clean, functional risk engine without syntax errors
"""

import logging
from typing import Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RiskAssessment:
    """Risk assessment result"""
    approved: bool
    risk_metrics: Dict
    rejection_reason: Optional[str] = None

class SimpleRiskEngine:
    """Simple but functional risk engine for demonstration"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_size = 0.1  # 10% max position
        self.max_portfolio_exposure = 0.3  # 30% max exposure
        self.stop_loss_percent = 0.05  # 5% stop loss
        logger.info(f"Simple Risk Engine initialized with ${initial_capital:,.2f}")

    def calculate_kelly_position_size(self, win_rate: float, win_loss_ratio: float, 
                                    account_size: float, regime: str = 'STABLE') -> float:
        """Calculate position size using Kelly criterion"""
        try:
            # Kelly formula: f* = (bp - q) / b
            # where b = win_loss_ratio, p = win_rate, q = 1 - win_rate
            if win_loss_ratio <= 0:
                return 0.0
            
            kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
            
            # Apply regime adjustments
            regime_multipliers = {
                'STABLE': 1.0,
                'TRENDING': 1.2,
                'VOLATILE': 0.7,
                'MEAN_REVERTING': 0.8,
                'CRISIS': 0.5
            }
            
            multiplier = regime_multipliers.get(regime.upper(), 1.0)
            adjusted_kelly = max(0.0, min(1.0, kelly_fraction * multiplier * 0.25))  # Conservative fraction
            
            position_size = adjusted_kelly * account_size
            return min(position_size, account_size * self.max_position_size)
            
        except Exception as e:
            logger.error(f"Kelly calculation error: {e}")
            return account_size * 0.01  # Default small position

    def assess_position_risk(self, symbol: str, entry_price: float, position_size: float,
                           stop_loss: float, take_profit: float, volatility: float, 
                           regime: str = 'STABLE') -> RiskAssessment:
        """Assess position risk with comprehensive checks"""
        try:
            # Calculate risk metrics
            risk_amount = abs(entry_price - stop_loss) * position_size
            reward_amount = abs(take_profit - entry_price) * position_size
            risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
            
            # Position size check
            position_value = entry_price * position_size
            position_percentage = (position_value / self.current_capital) * 100
            
            # Volatility check
            volatility_check = volatility <= 0.15  # Max 15% volatility acceptable
            
            # Risk-reward check
            risk_reward_check = risk_reward_ratio >= 1.5  # Minimum 1.5:1 ratio
            
            # Position size check
            position_size_check = position_percentage <= (self.max_position_size * 100)
            
            # Regime-specific adjustments
            regime_multipliers = {
                'STABLE': 1.0,
                'TRENDING': 1.2,
                'VOLATILE': 0.6,
                'MEAN_REVERTING': 0.8,
                'CRISIS': 0.3
            }
            
            regime_multiplier = regime_multipliers.get(regime.upper(), 1.0)
            
            # Final approval logic
            approved = (
                volatility_check and
                risk_reward_check and
                position_size_check and
                regime_multiplier > 0.3  # Minimum regime confidence
            )
            
            risk_metrics = {
                'position_value': position_value,
                'position_percentage': position_percentage,
                'risk_amount': risk_amount,
                'reward_amount': reward_amount,
                'risk_reward_ratio': risk_reward_ratio,
                'volatility': volatility,
                'regime_multiplier': regime_multiplier,
                'stop_loss_distance': abs(entry_price - stop_loss) / entry_price * 100
            }
            
            if not approved:
                reasons = []
                if not volatility_check:
                    reasons.append("High volatility")
                if not risk_reward_check:
                    reasons.append("Poor risk-reward ratio")
                if not position_size_check:
                    reasons.append("Position too large")
                if regime_multiplier <= 0.3:
                    reasons.append("Unfavorable market regime")
                
                rejection_reason = "; ".join(reasons)
            else:
                rejection_reason = None
            
            return RiskAssessment(
                approved=approved,
                risk_metrics=risk_metrics,
                rejection_reason=rejection_reason
            )
            
        except Exception as e:
            logger.error(f"Risk assessment error for {symbol}: {e}")
            return RiskAssessment(
                approved=False,
                risk_metrics={},
                rejection_reason="System error"
            )

    def update_portfolio_state(self, new_capital: float):
        """Update current capital state"""
        self.current_capital = new_capital

# Global instance
_simple_risk_engine = None

def get_simple_risk_engine(initial_capital: float = 100000.0) -> SimpleRiskEngine:
    """Get singleton simple risk engine instance"""
    global _simple_risk_engine
    if _simple_risk_engine is None:
        _simple_risk_engine = SimpleRiskEngine(initial_capital)
    return _simple_risk_engine

if __name__ == "__main__":
    print("Simple Risk Engine ready for integration")