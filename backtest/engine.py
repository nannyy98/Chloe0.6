"""
Professional Backtesting Engine
Realistic backtesting with slippage, commissions, and market impact
"""

import logging
from typing import Dict, List, Optional, Callable
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class TradeDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"

@dataclass
class Trade:
    """Individual trade record"""
    symbol: str
    direction: TradeDirection
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    quantity: float = 1.0
    pnl: float = 0.0
    pnl_percent: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0

class BacktestEngine:
    """Professional backtesting engine with realistic market conditions"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: Dict[str, Dict] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []
        self.commission_rate = 0.001  # 0.1% commission
        self.slippage_rate = 0.0005   # 0.05% slippage
        self.min_position_size = 0.001
        
        # Performance metrics
        self.max_equity = initial_capital
        self.min_equity = initial_capital
        self.peak_equity = initial_capital
        
    def add_commission(self, amount: float) -> float:
        """Calculate and return commission cost"""
        return amount * self.commission_rate
        
    def add_slippage(self, price: float, direction: str) -> float:
        """Add realistic slippage to price"""
        slippage_amount = price * self.slippage_rate
        if direction == 'BUY':
            return price + slippage_amount  # Buying pushes price up
        else:
            return price - slippage_amount  # Selling pushes price down
            
    def calculate_position_size(self, price: float, risk_percent: float = 0.02) -> float:
        """Calculate position size based on risk management"""
        risk_amount = self.current_capital * risk_percent
        # Simple position sizing - in practice would use ATR or volatility
        position_size = risk_amount / (price * 0.05)  # Assume 5% stop loss
        return max(position_size, self.min_position_size)
        
    def enter_position(self, symbol: str, direction: str, price: float, 
                      timestamp: datetime, signal_confidence: float = 1.0):
        """Enter a new position"""
        # Apply slippage
        execution_price = self.add_slippage(price, 'BUY' if direction == 'LONG' else 'SELL')
        
        # Calculate position size
        position_size = self.calculate_position_size(execution_price)
        
        # Apply commission
        commission = self.add_commission(position_size * execution_price)
        
        # Check if we have enough capital
        total_cost = (position_size * execution_price) + commission
        if total_cost > self.current_capital:
            logger.warning(f"⚠️ Insufficient capital for {symbol} position")
            return
            
        # Create trade record
        trade = Trade(
            symbol=symbol,
            direction=TradeDirection(direction),
            entry_time=timestamp,
            entry_price=execution_price,
            quantity=position_size,
            commission=commission,
            slippage=abs(execution_price - price)
        )
        
        # Update capital
        self.current_capital -= total_cost
        
        # Store position
        self.positions[symbol] = {
            'trade': trade,
            'current_price': execution_price
        }
        
        self._update_equity_curve(timestamp)
        logger.info(f"📈 Entered {direction} position: {symbol} {position_size:.4f}@${execution_price:.2f}")
        
    def exit_position(self, symbol: str, price: float, timestamp: datetime):
        """Exit existing position"""
        if symbol not in self.positions:
            return
            
        position = self.positions[symbol]
        trade = position['trade']
        
        # Apply slippage
        execution_price = self.add_slippage(price, 'SELL' if trade.direction == TradeDirection.LONG else 'BUY')
        
        # Calculate P&L
        if trade.direction == TradeDirection.LONG:
            pnl = (execution_price - trade.entry_price) * trade.quantity
        else:  # SHORT
            pnl = (trade.entry_price - execution_price) * trade.quantity
            
        # Apply commission on exit
        exit_commission = self.add_commission(trade.quantity * execution_price)
        total_commission = trade.commission + exit_commission
        
        # Update trade record
        trade.exit_time = timestamp
        trade.exit_price = execution_price
        trade.pnl = pnl - total_commission
        trade.pnl_percent = (pnl / (trade.quantity * trade.entry_price)) * 100
        trade.commission = total_commission
        trade.slippage += abs(execution_price - price)
        
        # Update capital
        self.current_capital += (trade.quantity * execution_price) - exit_commission
        self.trades.append(trade)
        
        # Remove position
        del self.positions[symbol]
        
        self._update_equity_curve(timestamp)
        logger.info(f"📉 Exited position: {symbol} P&L: ${trade.pnl:.2f} ({trade.pnl_percent:.2f}%)")
        
    def update_market_prices(self, symbol_prices: Dict[str, float], timestamp: datetime):
        """Update all position values with current market prices"""
        for symbol, price in symbol_prices.items():
            if symbol in self.positions:
                self.positions[symbol]['current_price'] = price
        self._update_equity_curve(timestamp)
        
    def _update_equity_curve(self, timestamp: datetime):
        """Update equity curve tracking"""
        # Calculate total portfolio value
        positions_value = sum(
            pos['trade'].quantity * pos['current_price'] 
            for pos in self.positions.values()
        )
        total_equity = self.current_capital + positions_value
        
        # Update peak equity for drawdown calculation
        self.peak_equity = max(self.peak_equity, total_equity)
        
        equity_record = {
            'timestamp': timestamp,
            'cash': self.current_capital,
            'positions_value': positions_value,
            'total_equity': total_equity,
            'drawdown': self.calculate_drawdown(total_equity),
            'num_positions': len(self.positions)
        }
        
        self.equity_curve.append(equity_record)
        self.max_equity = max(self.max_equity, total_equity)
        self.min_equity = min(self.min_equity, total_equity)
        
    def calculate_drawdown(self, current_equity: float) -> float:
        """Calculate drawdown percentage"""
        return ((self.peak_equity - current_equity) / self.peak_equity) * 100
        
    def run_backtest(self, data: pd.DataFrame, signal_generator: Callable, 
                    symbols: List[str] = None) -> Dict:
        """
        Run backtest on historical data
        
        Args:
            data: DataFrame with OHLCV data
            signal_generator: Function that generates signals
            symbols: List of symbols to test (if None, uses all in data)
        """
        logger.info("🚀 Starting backtest...")
        
        if symbols is None:
            # Assume single symbol data
            symbols = [data.columns[0].split('_')[0] if '_' in data.columns[0] else 'BTC/USDT']
            
        # Reset for new backtest
        self.__init__(self.initial_capital)
        
        # Process data chronologically
        for timestamp, row in data.iterrows():
            # Get current prices for all symbols
            current_prices = {}
            for symbol in symbols:
                if f'{symbol}_close' in row:
                    current_prices[symbol] = row[f'{symbol}_close']
                elif 'close' in row:
                    current_prices[symbol] = row['close']
                    
            # Update position values
            self.update_market_prices(current_prices, timestamp)
            
            # Generate signals for each symbol
            for symbol in symbols:
                if symbol in current_prices:
                    price = current_prices[symbol]
                    signal, confidence = signal_generator(symbol, timestamp, row, self.positions)
                    
                    # Handle signals
                    if signal == 'BUY' and symbol not in self.positions:
                        self.enter_position(symbol, 'LONG', price, timestamp, confidence)
                    elif signal == 'SELL' and symbol in self.positions:
                        self.exit_position(symbol, price, timestamp)
                    elif signal == 'SHORT' and symbol not in self.positions:
                        self.enter_position(symbol, 'SHORT', price, timestamp, confidence)
                    elif signal == 'COVER' and symbol in self.positions:
                        self.exit_position(symbol, price, timestamp)
                        
        # Close all remaining positions at the end
        final_prices = {symbol: data[f'{symbol}_close'].iloc[-1] if f'{symbol}_close' in data.columns 
                       else data['close'].iloc[-1] for symbol in symbols}
        
        for symbol in list(self.positions.keys()):
            self.exit_position(symbol, final_prices[symbol], data.index[-1])
            
        # Generate performance report
        return self.generate_performance_report()
        
    def generate_performance_report(self) -> Dict:
        """Generate comprehensive backtest performance report"""
        if not self.equity_curve:
            return {"error": "No trades executed"}
            
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('timestamp', inplace=True)
        
        # Calculate returns
        equity_df['returns'] = equity_df['total_equity'].pct_change().fillna(0)
        equity_df['cumulative_returns'] = (1 + equity_df['returns']).cumprod()
        
        # Basic metrics
        total_return = ((self.equity_curve[-1]['total_equity'] - self.initial_capital) / 
                       self.initial_capital) * 100
        total_pnl = self.equity_curve[-1]['total_equity'] - self.initial_capital
        
        # Volatility and risk metrics
        volatility = equity_df['returns'].std() * np.sqrt(252)  # Annualized
        sharpe_ratio = (equity_df['returns'].mean() / equity_df['returns'].std()) * np.sqrt(252) if equity_df['returns'].std() > 0 else 0
        
        # Drawdown metrics
        max_drawdown = max(record['drawdown'] for record in self.equity_curve)
        
        # Trade statistics
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl < 0]
        
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # Commission and slippage analysis
        total_commissions = sum(t.commission for t in self.trades)
        total_slippage = sum(t.slippage * t.quantity for t in self.trades)
        
        report = {
            'summary': {
                'initial_capital': self.initial_capital,
                'final_capital': self.equity_curve[-1]['total_equity'],
                'total_return_percent': total_return,
                'total_pnl': total_pnl,
                'total_trades': len(self.trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate_percent': win_rate * 100
            },
            'risk_metrics': {
                'annualized_volatility_percent': volatility * 100,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown_percent': max_drawdown,
                'profit_factor': profit_factor
            },
            'costs': {
                'total_commissions': total_commissions,
                'total_slippage': total_slippage,
                'total_costs': total_commissions + total_slippage
            },
            'equity_curve': self.equity_curve[-10:] if len(self.equity_curve) > 10 else self.equity_curve,
            'recent_trades': self.trades[-10:] if len(self.trades) > 10 else self.trades
        }
        
        logger.info(f"✅ Backtest completed. Total Return: {total_return:.2f}%")
        return report

# Example usage functions
def simple_moving_average_signal(symbol: str, timestamp, row, positions: Dict) -> tuple[str, float]:
    """Simple moving average crossover signal generator"""
    try:
        # This is a placeholder - in practice would use actual indicator values
        # from the row data
        if symbol in positions:
            # Simple exit logic
            return 'SELL', 0.8
        else:
            # Simple entry logic  
            return 'BUY', 0.7
    except:
        return 'HOLD', 0.0

def backtest_example():
    """Example backtest usage"""
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Run backtest
    engine = BacktestEngine(initial_capital=10000)
    results = engine.run_backtest(data, simple_moving_average_signal)
    
    return results