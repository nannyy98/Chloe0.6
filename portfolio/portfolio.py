"""
Portfolio Engine - Core Trading Account Management
Professional portfolio accounting like institutional funds
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class Position:
    """Individual position in portfolio"""
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    pnl: float = 0.0
    pnl_percent: float = 0.0
    exposure: float = 0.0
    last_updated: datetime = None
    
    def update_price(self, new_price: float):
        """Update position with new market price"""
        self.current_price = new_price
        self.pnl = (new_price - self.entry_price) * self.quantity
        self.pnl_percent = ((new_price - self.entry_price) / self.entry_price) * 100
        self.exposure = abs(self.quantity * new_price)
        self.last_updated = datetime.now()
        
    def get_value(self) -> float:
        """Get current position value"""
        return self.quantity * self.current_price

@dataclass
class Portfolio:
    """Main portfolio management system"""
    initial_capital: float
    current_capital: float = None
    positions: Dict[str, Position] = field(default_factory=dict)
    transaction_history: List[Dict] = field(default_factory=list)
    equity_curve: List[Dict] = field(default_factory=list)
    
    def __post_init__(self):
        if self.current_capital is None:
            self.current_capital = self.initial_capital
        self.start_time = datetime.now()
        self.max_equity = self.initial_capital
        self.min_equity = self.initial_capital
        self._update_equity_curve()
        
    def _update_equity_curve(self):
        """Update equity curve tracking"""
        total_positions_value = sum(pos.get_value() for pos in self.positions.values())
        total_equity = self.current_capital + total_positions_value
        
        equity_record = {
            'timestamp': datetime.now(),
            'cash': self.current_capital,
            'positions_value': total_positions_value,
            'total_equity': total_equity,
            'drawdown': self.calculate_drawdown(total_equity)
        }
        self.equity_curve.append(equity_record)
        
        # Update max/min equity
        self.max_equity = max(self.max_equity, total_equity)
        self.min_equity = min(self.min_equity, total_equity)
        
    def enter_position(self, symbol: str, quantity: float, price: float, 
                      order_cost: float = 0.0, commission: float = 0.0):
        """Enter new position"""
        # Check if we have enough capital
        total_cost = (abs(quantity) * price) + order_cost + commission
        if total_cost > self.current_capital:
            raise ValueError(f"Insufficient capital. Need ${total_cost:.2f}, have ${self.current_capital:.2f}")
            
        # Create new position
        position = Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=price,
            entry_time=datetime.now(),
            current_price=price
        )
        position.update_price(price)
        
        self.positions[symbol] = position
        self.current_capital -= total_cost
        
        # Log transaction
        transaction = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'action': 'BUY' if quantity > 0 else 'SELL',
            'quantity': quantity,
            'price': price,
            'cost': total_cost,
            'commission': commission
        }
        self.transaction_history.append(transaction)
        
        self._update_equity_curve()
        logger.info(f"✅ Entered position: {symbol} {quantity}@${price:.2f}")
        
    def exit_position(self, symbol: str, quantity: float = None, price: float = None,
                     order_cost: float = 0.0, commission: float = 0.0):
        """Exit position (partial or full)"""
        if symbol not in self.positions:
            raise ValueError(f"No position found for {symbol}")
            
        position = self.positions[symbol]
        
        # Default to full position exit if quantity not specified
        if quantity is None:
            quantity = position.quantity
            
        # Check if we're trying to exit more than we have
        if abs(quantity) > abs(position.quantity):
            raise ValueError(f"Cannot exit {quantity} units, only {position.quantity} units held")
            
        # Default to current market price if not specified
        if price is None:
            price = position.current_price
            
        # Calculate transaction values
        proceeds = abs(quantity) * price
        total_value = proceeds - order_cost - commission
        
        # Update position
        if abs(quantity) >= abs(position.quantity):
            # Full exit
            del self.positions[symbol]
        else:
            # Partial exit
            position.quantity -= quantity
            position.update_price(position.current_price)
            
        self.current_capital += total_value
        
        # Log transaction
        transaction = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'action': 'SELL' if quantity > 0 else 'BUY_TO_COVER',
            'quantity': quantity,
            'price': price,
            'proceeds': proceeds,
            'commission': commission
        }
        self.transaction_history.append(transaction)
        
        self._update_equity_curve()
        logger.info(f"✅ Exited position: {symbol} {quantity}@${price:.2f}")
        
    def update_market_prices(self, symbol_prices: Dict[str, float]):
        """Update all position prices with current market data"""
        for symbol, price in symbol_prices.items():
            if symbol in self.positions:
                self.positions[symbol].update_price(price)
        self._update_equity_curve()
        
    def calculate_total_value(self) -> float:
        """Calculate total portfolio value"""
        positions_value = sum(pos.get_value() for pos in self.positions.values())
        return self.current_capital + positions_value
        
    def calculate_total_pnl(self) -> float:
        """Calculate total profit/loss"""
        total_positions_pnl = sum(pos.pnl for pos in self.positions.values())
        return total_positions_pnl
        
    def calculate_total_return(self) -> float:
        """Calculate total portfolio return %"""
        total_value = self.calculate_total_value()
        return ((total_value - self.initial_capital) / self.initial_capital) * 100
        
    def calculate_drawdown(self, current_equity: float = None) -> float:
        """Calculate current drawdown %"""
        if current_equity is None:
            current_equity = self.calculate_total_value()
        return ((self.max_equity - current_equity) / self.max_equity) * 100
        
    def calculate_exposure(self) -> float:
        """Calculate total portfolio exposure %"""
        total_value = self.calculate_total_value()
        if total_value == 0:
            return 0.0
        positions_exposure = sum(pos.exposure for pos in self.positions.values())
        return (positions_exposure / total_value) * 100
        
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get specific position"""
        return self.positions.get(symbol)
        
    def get_all_positions(self) -> Dict[str, Position]:
        """Get all current positions"""
        return self.positions.copy()
        
    def get_performance_metrics(self) -> Dict:
        """Get comprehensive performance metrics"""
        total_value = self.calculate_total_value()
        total_pnl = self.calculate_total_pnl()
        total_return = self.calculate_total_return()
        current_drawdown = self.calculate_drawdown()
        exposure = self.calculate_exposure()
        
        # Calculate Sharpe-like ratio (simplified)
        if len(self.equity_curve) > 1:
            equity_values = [record['total_equity'] for record in self.equity_curve]
            returns = np.diff(equity_values) / equity_values[:-1]
            if np.std(returns) > 0:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0
            
        return {
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'total_value': total_value,
            'total_pnl': total_pnl,
            'total_return_percent': total_return,
            'current_drawdown_percent': current_drawdown,
            'portfolio_exposure_percent': exposure,
            'number_of_positions': len(self.positions),
            'sharpe_ratio': sharpe_ratio,
            'max_equity': self.max_equity,
            'min_equity': self.min_equity
        }
        
    def get_equity_history(self) -> pd.DataFrame:
        """Get equity curve as DataFrame"""
        return pd.DataFrame(self.equity_curve)

# Global portfolio instance
portfolio = None

def initialize_portfolio(initial_capital: float) -> Portfolio:
    """Initialize global portfolio instance"""
    global portfolio
    portfolio = Portfolio(initial_capital=initial_capital)
    logger.info(f"💼 Portfolio initialized with ${initial_capital:,.2f}")
    return portfolio