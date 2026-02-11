"""
Institutional Risk Engine
Professional risk management with kill-switch functionality
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM" 
    HIGH = "HIGH"
    EXTREME = "EXTREME"

@dataclass
class RiskLimits:
    """Risk limits configuration"""
    max_risk_per_trade: float = 0.01      # 1% per trade
    max_portfolio_risk: float = 0.05      # 5% total portfolio
    max_drawdown: float = 0.15            # 15% max drawdown
    max_daily_loss: float = 0.03          # 3% daily loss limit
    max_position_concentration: float = 0.25  # 25% per position
    max_correlation_risk: float = 0.7     # 70% correlation threshold
    stop_loss_multiplier: float = 2.0     # ATR multiplier for stop loss

class RiskManager:
    """Professional risk management system"""
    
    def __init__(self, portfolio, limits: RiskLimits = None):
        self.portfolio = portfolio
        self.limits = limits or RiskLimits()
        self.trade_history: List[Dict] = []
        self.daily_losses: List[float] = []
        self.risk_events: List[Dict] = []
        self.circuit_breaker_active = False
        self.last_risk_check = datetime.now()
        self.correlation_matrix: Dict[str, Dict[str, float]] = {}
        self.position_correlations: Dict[str, float] = {}
        self.max_correlation_alert = 0.7  # Alert when correlations exceed this
        
    def check_trade_risk(self, symbol: str, signal: str, quantity: float, 
                        entry_price: float, atr: float = None) -> tuple[bool, str]:
        """Check if trade complies with risk limits"""
        if self.circuit_breaker_active:
            return False, "CIRCUIT_BREAKER_ACTIVE"
            
        # 1. Check maximum risk per trade
        max_trade_risk_amount = self.portfolio.current_capital * self.limits.max_risk_per_trade
        position_value = abs(quantity) * entry_price
        
        if position_value > max_trade_risk_amount:
            return False, f"POSITION_SIZE_EXCEEDS_LIMIT_${max_trade_risk_amount:,.2f}"
            
        # 2. Check portfolio exposure
        total_portfolio_value = self.portfolio.calculate_total_value()
        current_exposure = self.portfolio.calculate_exposure()
        new_exposure = (position_value / total_portfolio_value) * 100
        
        if (current_exposure + new_exposure) > (self.limits.max_portfolio_risk * 100):
            return False, f"PORTFOLIO_EXPOSURE_EXCEEDS_LIMIT_{self.limits.max_portfolio_risk*100}%"
            
        # 3. Check position concentration
        if new_exposure > (self.limits.max_position_concentration * 100):
            return False, f"POSITION_CONCENTRATION_EXCEEDS_LIMIT_{self.limits.max_position_concentration*100}%"
            
        # 4. Check correlation risk with existing positions
        correlation_check, correlation_msg = self.calculate_correlation_risk(symbol)
        if not correlation_check:
            return False, correlation_msg
            
        # 5. Calculate and check stop loss distance
        if atr:
            stop_loss_distance = atr * self.limits.stop_loss_multiplier
            stop_loss_percent = (stop_loss_distance / entry_price) * 100
            if stop_loss_percent > 10:  # Cap stop loss at 10%
                stop_loss_distance = entry_price * 0.10
                
            if signal == 'BUY':
                stop_loss_price = entry_price - stop_loss_distance
            else:
                stop_loss_price = entry_price + stop_loss_distance
                
            # Stop loss should not be too tight (<1%) or too wide (>10%)
            if stop_loss_percent < 1:
                return False, f"STOP_LOSS_TOO_TIGHT_{stop_loss_percent:.1f}%"
            if stop_loss_percent > 10:
                return False, f"STOP_LOSS_TOO_WIDE_{stop_loss_percent:.1f}%"
                
        logger.info(f"✅ Trade risk check passed for {symbol}")
        return True, "RISK_CHECK_PASSED"
        
    def _is_crypto_symbol(self, symbol: str) -> bool:
        """Simple crypto detection"""
        crypto_indicators = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'ADA']
        return any(indicator in symbol.upper() for indicator in crypto_indicators)
        
    def update_portfolio_risk(self) -> tuple[bool, str]:
        """Check overall portfolio risk and trigger kill switch if needed"""
        if self.circuit_breaker_active:
            return False, "CIRCUIT_BREAKER_ALREADY_ACTIVE"
            
        # Get current metrics
        current_value = self.portfolio.calculate_total_value()
        current_drawdown = self.portfolio.calculate_drawdown()
        current_exposure = self.portfolio.calculate_exposure()
        
        # Check max drawdown
        if current_drawdown > (self.limits.max_drawdown * 100):
            self._trigger_circuit_breaker(f"MAX_DRAWDOWN_EXCEEDED_{current_drawdown:.2f}%")
            return False, f"MAX_DRAWDOWN_EXCEEDED_{current_drawdown:.2f}%"
            
        # Check portfolio risk concentration
        if current_exposure > (self.limits.max_portfolio_risk * 100):
            return False, f"PORTFOLIO_RISK_TOO_HIGH_{current_exposure:.2f}%"
            
        # Check recent daily performance
        self._update_daily_losses()
        recent_daily_losses = [loss for loss in self.daily_losses[-5:] if loss < 0]
        if len(recent_daily_losses) >= 3:
            total_recent_loss = sum(recent_daily_losses)
            if abs(total_recent_loss) > (self.limits.max_daily_loss * 3 * 100):  # 3-day threshold
                self._trigger_circuit_breaker(f"CONSECUTIVE_LOSSES_DETECTED_{total_recent_loss:.2f}%")
                return False, f"CONSECUTIVE_LOSSES_EXCEEDED"
                
        # Risk check passed
        self.last_risk_check = datetime.now()
        return True, "PORTFOLIO_RISK_CHECK_PASSED"
        
    def _update_daily_losses(self):
        """Update daily loss tracking"""
        if len(self.portfolio.equity_curve) < 2:
            return
            
        # Group by day and calculate daily returns
        equity_df = self.portfolio.get_equity_history()
        if len(equity_df) > 1:
            equity_df['date'] = equity_df['timestamp'].dt.date
            daily_equity = equity_df.groupby('date')['total_equity'].last()
            daily_returns = daily_equity.pct_change().fillna(0)
            
            # Keep only recent days (last 30)
            if len(daily_returns) > 30:
                daily_returns = daily_returns[-30:]
                
            # Update daily losses list
            self.daily_losses = daily_returns.tolist()
            
    def calculate_position_size(self, symbol: str, signal: str, entry_price: float, 
                               atr: float, volatility: float = None) -> float:
        """Calculate optimal position size using risk-adjusted sizing"""
        # Base risk amount
        max_risk_amount = self.portfolio.current_capital * self.limits.max_risk_per_trade
        
        # Use ATR-based stop loss for size calculation
        if atr:
            stop_loss_distance = atr * self.limits.stop_loss_multiplier
            
            # Calculate position size
            if stop_loss_distance > 0:
                position_size = max_risk_amount / stop_loss_distance
            else:
                # Fallback using volatility if ATR is problematic
                if volatility:
                    position_size = max_risk_amount / (entry_price * volatility * 3)
                else:
                    position_size = 0
                    
            # Also cap by maximum portfolio risk
            max_value_position = self.portfolio.calculate_total_value() * self.limits.max_portfolio_risk
            max_units_position = max_value_position / entry_price
            position_size = min(position_size, max_units_position)
            
        else:
            # Volatility-based sizing when no ATR
            base_size = max_risk_amount / (entry_price * 0.03)  # Default 3% stop
            position_size = min(base_size, (self.limits.max_portfolio_risk * 
                                          self.portfolio.calculate_total_value() / entry_price))
            
        # Apply signal direction
        if signal == 'SELL':
            position_size = -position_size
            
        logger.info(f"📏 Calculated position size for {symbol}: {position_size:.4f} units")
        return position_size
        
    def calculate_correlation_risk(self, new_symbol: str) -> tuple[bool, str]:
        """Calculate correlation risk with existing positions"""
        existing_positions = self.portfolio.get_all_positions()
        
        if not existing_positions:
            return True, "NO_EXISTING_POSITIONS"
            
        # Simplified correlation check - in production would use actual price data
        correlations = {}
        high_corr_symbols = []
        
        for symbol in existing_positions.keys():
            # For demo purposes, using fixed correlations based on asset type
            # In real system would calculate actual correlation from price data
            if self._are_assets_correlated(new_symbol, symbol):
                corr_value = 0.8  # High correlation assumption
                correlations[symbol] = corr_value
                if corr_value > self.max_correlation_alert:
                    high_corr_symbols.append(symbol)
        
        if high_corr_symbols:
            return False, f"HIGH_CORRELATION_WITH_{','.join(high_corr_symbols)}"
        
        return True, "CORRELATION_CHECK_PASSED"
        
    def _are_assets_correlated(self, symbol1: str, symbol2: str) -> bool:
        """Check if two assets are likely to be correlated"""
        # Group by asset type
        crypto_major = {'BTC', 'ETH', 'SOL', 'XRP', 'ADA', 'AVAX', 'DOT', 'LINK'}
        crypto_alt = {'DOGE', 'SHIB', 'MATIC', 'ATOM', 'UNI', 'SUSHI'}
        stablecoins = {'USDT', 'USDC', 'DAI', 'BUSD'}
        
        # Extract base currency from symbols like 'BTC/USDT'
        sym1_base = symbol1.split('/')[0] if '/' in symbol1 else symbol1
        sym2_base = symbol2.split('/')[0] if '/' in symbol2 else symbol2
        
        # Major cryptos tend to be correlated
        if sym1_base in crypto_major and sym2_base in crypto_major:
            return True
            
        # Alt coins often correlate with majors
        if (sym1_base in crypto_alt and sym2_base in crypto_major) or \
           (sym1_base in crypto_major and sym2_base in crypto_alt):
            return True
            
        # Same asset class
        if sym1_base in crypto_major and sym2_base in crypto_major:
            return True
        if sym1_base in crypto_alt and sym2_base in crypto_alt:
            return True
        if sym1_base in stablecoins and sym2_base in stablecoins:
            return True
            
        return False
        
    def _trigger_circuit_breaker(self, reason: str):
        """Trigger emergency circuit breaker - close all positions"""
        logger.critical(f"🚨 CIRCUIT BREAKER TRIGGERED: {reason}")
        
        # Close all positions
        positions = self.portfolio.get_all_positions()
        for symbol, position in positions.items():
            try:
                # Market sell all positions
                self.portfolio.exit_position(
                    symbol=symbol, 
                    quantity=position.quantity,
                    price=position.current_price  # Use current price for immediate exit
                )
                logger.info(f"🛑 Emergency exit: {symbol} {position.quantity}")
            except Exception as e:
                logger.error(f"❌ Error during emergency exit {symbol}: {e}")
                
        self.circuit_breaker_active = True
        self.risk_events.append({
            'timestamp': datetime.now(),
            'event_type': 'CIRCUIT_BREAKER',
            'reason': reason,
            'portfolio_value_before': self.portfolio.calculate_total_value()
        })
        
        # Send emergency alert (would integrate with Telegram/slack)
        logger.critical("🚨 EMERGENCY: All positions closed. Trading suspended.")
        
    def reset_circuit_breaker(self):
        """Reset circuit breaker after manual review"""
        if self.circuit_breaker_active:
            self.circuit_breaker_active = False
            logger.info("✅ Circuit breaker reset. Manual review required before resuming.")
            
    def get_risk_report(self) -> Dict:
        """Generate comprehensive risk report"""
        portfolio_metrics = self.portfolio.get_performance_metrics()
        
        return {
            'timestamp': datetime.now(),
            'portfolio_metrics': portfolio_metrics,
            'risk_limits': {
                'max_risk_per_trade': self.limits.max_risk_per_trade,
                'max_drawdown': self.limits.max_drawdown,
                'max_daily_loss': self.limits.max_daily_loss,
                'max_position_concentration': self.limits.max_position_concentration
            },
            'current_risk_status': {
                'circuit_breaker_active': self.circuit_breaker_active,
                'current_drawdown': portfolio_metrics['current_drawdown_percent'],
                'portfolio_exposure': portfolio_metrics['portfolio_exposure_percent'],
                'number_of_positions': portfolio_metrics['number_of_positions']
            },
            'recent_daily_losses': self.daily_losses[-10:] if self.daily_losses else [],
            'risk_events_count': len(self.risk_events)
        }

# Global risk manager instance
risk_manager = None

def initialize_risk_manager(portfolio, limits: RiskLimits = None) -> RiskManager:
    """Initialize global risk manager"""
    global risk_manager
    risk_manager = RiskManager(portfolio, limits)
    logger.info("🛡️ Risk manager initialized with institutional limits")
    return risk_manager