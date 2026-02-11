"""
Advanced Trading Strategies Framework
Professional institutional-grade trading strategies
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass

from core.event_bus import SignalEvent, EventType
from portfolio.portfolio import Portfolio

logger = logging.getLogger(__name__)

@dataclass
class StrategySignal:
    """Trading signal from strategy"""
    symbol: str
    signal: str  # BUY, SELL, HOLD
    confidence: float  # 0.0 to 1.0
    strength: float  # -1.0 to 1.0, negative = SELL
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[float] = None

class BaseStrategy(ABC):
    """Abstract base class for trading strategies"""
    
    def __init__(self, strategy_id: str, name: str, description: str):
        self.strategy_id = strategy_id
        self.name = name
        self.description = description
        self.active_positions = {}
        self.performance_metrics = {}
        
    @abstractmethod
    def generate_signal(self, symbol: str, data: pd.DataFrame, 
                       portfolio: Portfolio) -> Optional[StrategySignal]:
        """Generate trading signal based on market data and portfolio"""
        pass
        
    def update_performance(self, realized_pnl: float):
        """Update strategy performance metrics"""
        if 'total_pnl' not in self.performance_metrics:
            self.performance_metrics['total_pnl'] = 0.0
            self.performance_metrics['total_trades'] = 0
            self.performance_metrics['winning_trades'] = 0
            self.performance_metrics['losing_trades'] = 0
            
        self.performance_metrics['total_pnl'] += realized_pnl
        self.performance_metrics['total_trades'] += 1
        
        if realized_pnl > 0:
            self.performance_metrics['winning_trades'] += 1
        else:
            self.performance_metrics['losing_trades'] += 1
            
    def get_performance_summary(self) -> Dict:
        """Get strategy performance summary"""
        total_trades = self.performance_metrics.get('total_trades', 0)
        winning_trades = self.performance_metrics.get('winning_trades', 0)
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        return {
            'strategy_id': self.strategy_id,
            'name': self.name,
            'total_pnl': self.performance_metrics.get('total_pnl', 0.0),
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': self.performance_metrics.get('losing_trades', 0),
            'win_rate_percent': win_rate,
            'avg_pnl_per_trade': self.performance_metrics.get('total_pnl', 0.0) / total_trades if total_trades > 0 else 0.0
        }

class MeanReversionStrategy(BaseStrategy):
    """Mean reversion strategy using Bollinger Bands"""
    
    def __init__(self, lookback: int = 20, std_dev: float = 2.0, 
                 min_confidence: float = 0.6):
        super().__init__(
            strategy_id='mean_reversion_v1',
            name='Mean Reversion Strategy',
            description='Uses Bollinger Bands to identify mean reversion opportunities'
        )
        self.lookback = lookback
        self.std_dev = std_dev
        self.min_confidence = min_confidence
        
    def generate_signal(self, symbol: str, data: pd.DataFrame, 
                       portfolio: Portfolio) -> Optional[StrategySignal]:
        """Generate mean reversion signal"""
        if len(data) < self.lookback + 5:
            return None
            
        # Calculate Bollinger Bands
        close_prices = data['close'] if 'close' in data.columns else data['Close']
        sma = close_prices.rolling(window=self.lookback).mean()
        std = close_prices.rolling(window=self.lookback).std()
        upper_band = sma + (std * self.std_dev)
        lower_band = sma - (std * self.std_dev)
        
        current_price = close_prices.iloc[-1]
        upper = upper_band.iloc[-1]
        lower = lower_band.iloc[-1]
        
        # Calculate position in bands (0 to 1, where 0.5 is middle)
        position_in_bands = (current_price - lower) / (upper - lower) if (upper - lower) > 0 else 0.5
        
        # Generate signal based on position in bands
        signal = None
        strength = 0.0
        confidence = 0.0
        
        if position_in_bands < 0.1:  # Oversold - BUY signal
            signal = 'BUY'
            strength = min(1.0, (0.5 - position_in_bands) * 2)  # Stronger signal closer to 0
            confidence = min(1.0, strength + 0.1)  # Add minimum confidence
            
        elif position_in_bands > 0.9:  # Overbought - SELL signal
            signal = 'SELL'
            strength = -min(1.0, (position_in_bands - 0.5) * 2)  # Stronger signal closer to 1
            confidence = min(1.0, abs(strength) + 0.1)  # Add minimum confidence
            
        else:
            signal = 'HOLD'
            strength = 0.0
            confidence = 0.0
            
        # Only return signal if confidence meets minimum threshold
        if confidence >= self.min_confidence or signal == 'HOLD':
            # Calculate stop loss and take profit based on ATR
            atr = self._calculate_atr(data)
            stop_loss = self._calculate_stop_loss(current_price, signal, atr)
            take_profit = self._calculate_take_profit(current_price, signal, atr)
            
            return StrategySignal(
                symbol=symbol,
                signal=signal,
                confidence=confidence,
                strength=strength,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
        return None
        
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high = data['high'] if 'high' in data.columns else data['High']
        low = data['low'] if 'low' in data.columns else data['Low']
        close = data['close'] if 'close' in data.columns else data['Close']
        
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        
        return atr if not pd.isna(atr) else current_price * 0.02  # Fallback
        
    def _calculate_stop_loss(self, price: float, signal: str, atr: float) -> Optional[float]:
        """Calculate stop loss based on ATR"""
        if atr <= 0:
            return None
            
        stop_distance = atr * 2.0  # 2x ATR stop loss
        
        if signal == 'BUY':
            return price - stop_distance
        elif signal == 'SELL':
            return price + stop_distance
        else:
            return None
            
    def _calculate_take_profit(self, price: float, signal: str, atr: float) -> Optional[float]:
        """Calculate take profit based on ATR (2:1 risk/reward ratio)"""
        if atr <= 0:
            return None
            
        profit_distance = atr * 4.0  # 4x ATR take profit (2:1 R/R ratio)
        
        if signal == 'BUY':
            return price + profit_distance
        elif signal == 'SELL':
            return price - profit_distance
        else:
            return None

class MomentumStrategy(BaseStrategy):
    """Momentum strategy using moving average convergence/divergence"""
    
    def __init__(self, short_ma: int = 12, long_ma: int = 26, signal_ma: int = 9,
                 min_confidence: float = 0.65):
        super().__init__(
            strategy_id='momentum_v1',
            name='Momentum Strategy',
            description='Uses MACD for momentum-based trading signals'
        )
        self.short_ma = short_ma
        self.long_ma = long_ma
        self.signal_ma = signal_ma
        self.min_confidence = min_confidence
        
    def generate_signal(self, symbol: str, data: pd.DataFrame, 
                       portfolio: Portfolio) -> Optional[StrategySignal]:
        """Generate momentum signal"""
        if len(data) < max(self.long_ma, self.signal_ma) + 5:
            return None
            
        close_prices = data['close'] if 'close' in data.columns else data['Close']
        
        # Calculate MACD
        short_ema = close_prices.ewm(span=self.short_ma).mean()
        long_ema = close_prices.ewm(span=self.long_ma).mean()
        macd_line = short_ema - long_ema
        signal_line = macd_line.ewm(span=self.signal_ma).mean()
        histogram = macd_line - signal_line
        
        # Current values
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        current_hist = histogram.iloc[-1]
        current_price = close_prices.iloc[-1]
        
        # Previous values for trend confirmation
        prev_hist = histogram.iloc[-2] if len(histogram) > 1 else 0
        
        # Generate signal based on MACD crossovers
        signal = None
        strength = 0.0
        confidence = 0.0
        
        # Bullish crossover: MACD line crosses above signal line
        if current_macd > current_signal and prev_hist < current_hist and current_hist > 0:
            signal = 'BUY'
            strength = min(1.0, current_hist * 10)  # Normalize strength
            confidence = min(1.0, 0.5 + abs(current_hist) * 5)  # Higher confidence for stronger moves
            
        # Bearish crossover: MACD line crosses below signal line
        elif current_macd < current_signal and prev_hist > current_hist and current_hist < 0:
            signal = 'SELL'
            strength = -min(1.0, abs(current_hist) * 10)  # Negative strength for sell
            confidence = min(1.0, 0.5 + abs(current_hist) * 5)  # Higher confidence for stronger moves
            
        else:
            signal = 'HOLD'
            strength = 0.0
            confidence = 0.0
            
        # Apply minimum confidence threshold
        if confidence >= self.min_confidence or signal == 'HOLD':
            # Calculate stops based on volatility
            volatility = close_prices.pct_change().rolling(20).std().iloc[-1]
            atr = self._calculate_atr(data)
            
            stop_loss = self._calculate_stop_loss(current_price, signal, atr)
            take_profit = self._calculate_take_profit(current_price, signal, atr)
            
            return StrategySignal(
                symbol=symbol,
                signal=signal,
                confidence=confidence,
                strength=strength,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
        return None
        
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high = data['high'] if 'high' in data.columns else data['High']
        low = data['low'] if 'low' in data.columns else data['Low']
        close = data['close'] if 'close' in data.columns else data['Close']
        
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        
        return atr if not pd.isna(atr) else close.iloc[-1] * 0.02  # Fallback
        
    def _calculate_stop_loss(self, price: float, signal: str, atr: float) -> Optional[float]:
        """Calculate stop loss based on ATR"""
        if atr <= 0:
            return None
            
        stop_distance = atr * 2.0  # 2x ATR stop loss
        
        if signal == 'BUY':
            return price - stop_distance
        elif signal == 'SELL':
            return price + stop_distance
        else:
            return None
            
    def _calculate_take_profit(self, price: float, signal: str, atr: float) -> Optional[float]:
        """Calculate take profit based on ATR (2:1 risk/reward ratio)"""
        if atr <= 0:
            return None
            
        profit_distance = atr * 4.0  # 4x ATR take profit (2:1 R/R ratio)
        
        if signal == 'BUY':
            return price + profit_distance
        elif signal == 'SELL':
            return price - profit_distance
        else:
            return None

class RiskParityStrategy(BaseStrategy):
    """Risk parity strategy for portfolio diversification"""
    
    def __init__(self, risk_budget: float = 0.02, rebalance_threshold: float = 0.05):
        super().__init__(
            strategy_id='risk_parity_v1',
            name='Risk Parity Strategy',
            description='Allocates capital based on risk contribution of each asset'
        )
        self.risk_budget = risk_budget  # 2% risk budget per trade
        self.rebalance_threshold = rebalance_threshold  # 5% threshold for rebalancing
        
    def generate_signal(self, symbol: str, data: pd.DataFrame, 
                       portfolio: Portfolio) -> Optional[StrategySignal]:
        """Generate risk parity allocation signal"""
        # Calculate volatility for the symbol
        if len(data) < 30:
            return None
            
        close_prices = data['close'] if 'close' in data.columns else data['Close']
        returns = close_prices.pct_change().dropna()
        volatility = returns.rolling(20).std().iloc[-1]
        
        if pd.isna(volatility) or volatility <= 0:
            return None
            
        current_price = close_prices.iloc[-1]
        
        # Calculate target position size based on risk budget
        # Risk = Position Size * Price * Volatility
        # So Position Size = Risk Budget / (Price * Volatility)
        target_position_value = portfolio.current_capital * self.risk_budget
        target_position_size = target_position_value / (current_price * volatility)
        
        # Get current position in this symbol
        current_position = portfolio.get_position(symbol)
        current_size = current_position.quantity if current_position else 0
        
        # Calculate difference to determine signal
        size_diff = target_position_size - abs(current_size)
        
        signal = None
        strength = 0.0
        confidence = 0.7  # Medium-high confidence for risk parity
        
        if abs(size_diff) > abs(current_size) * self.rebalance_threshold:
            if size_diff > 0:
                signal = 'BUY'
                strength = min(1.0, size_diff / target_position_size)
            else:
                signal = 'SELL'
                strength = -min(1.0, abs(size_diff) / abs(current_size)) if current_size != 0 else -0.5
        else:
            signal = 'HOLD'
            strength = 0.0
            
        # Calculate stops based on volatility
        atr = self._calculate_atr(data)
        stop_loss = self._calculate_stop_loss(current_price, signal, atr)
        take_profit = self._calculate_take_profit(current_price, signal, atr)
        
        return StrategySignal(
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            strength=strength,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=target_position_size if signal != 'HOLD' else None
        )
        
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high = data['high'] if 'high' in data.columns else data['High']
        low = data['low'] if 'low' in data.columns else data['Low']
        close = data['close'] if 'close' in data.columns else data['Close']
        
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        
        return atr if not pd.isna(atr) else close.iloc[-1] * 0.02  # Fallback
        
    def _calculate_stop_loss(self, price: float, signal: str, atr: float) -> Optional[float]:
        """Calculate stop loss based on ATR"""
        if atr <= 0:
            return None
            
        stop_distance = atr * 2.0  # 2x ATR stop loss
        
        if signal == 'BUY':
            return price - stop_distance
        elif signal == 'SELL':
            return price + stop_distance
        else:
            return None
            
    def _calculate_take_profit(self, price: float, signal: str, atr: float) -> Optional[float]:
        """Calculate take profit based on ATR (2:1 risk/reward ratio)"""
        if atr <= 0:
            return None
            
        profit_distance = atr * 4.0  # 4x ATR take profit (2:1 R/R ratio)
        
        if signal == 'BUY':
            return price + profit_distance
        elif signal == 'SELL':
            return price - profit_distance
        else:
            return None

class StrategyManager:
    """Manages multiple strategies and their signals"""
    
    def __init__(self):
        self.strategies: Dict[str, BaseStrategy] = {}
        self.weights: Dict[str, float] = {}  # Strategy weights for portfolio allocation
        self.active_signals: Dict[str, Dict[str, StrategySignal]] = {}  # symbol -> {strategy_id -> signal}
        
    def add_strategy(self, strategy: BaseStrategy, weight: float = 1.0):
        """Add strategy with optional weight"""
        self.strategies[strategy.strategy_id] = strategy
        self.weights[strategy.strategy_id] = weight
        logger.info(f"✅ Added strategy: {strategy.name} (weight: {weight})")
        
    def generate_signals(self, symbol: str, data: pd.DataFrame, 
                        portfolio: Portfolio) -> List[StrategySignal]:
        """Generate signals from all strategies for a symbol"""
        signals = []
        
        for strategy_id, strategy in self.strategies.items():
            try:
                signal = strategy.generate_signal(symbol, data, portfolio)
                if signal:
                    signals.append(signal)
                    # Store active signal
                    if symbol not in self.active_signals:
                        self.active_signals[symbol] = {}
                    self.active_signals[symbol][strategy_id] = signal
            except Exception as e:
                logger.error(f"❌ Error in strategy {strategy_id}: {e}")
                
        return signals
        
    def combine_signals(self, symbol: str, signals: List[StrategySignal]) -> Optional[StrategySignal]:
        """Combine multiple strategy signals into one"""
        if not signals:
            return None
            
        # Weighted average of signals
        total_weight = 0.0
        weighted_strength = 0.0
        total_confidence = 0.0
        
        for signal in signals:
            weight = self.weights.get(signal.strategy_id, 1.0) if hasattr(signal, 'strategy_id') else 1.0
            total_weight += weight
            weighted_strength += signal.strength * weight
            total_confidence += signal.confidence * weight
            
        if total_weight == 0:
            return None
            
        avg_strength = weighted_strength / total_weight
        avg_confidence = total_confidence / total_weight
        
        # Determine final signal based on average strength
        if avg_strength > 0.1:  # Threshold to avoid weak signals
            final_signal = 'BUY'
        elif avg_strength < -0.1:
            final_signal = 'SELL'
        else:
            final_signal = 'HOLD'
            
        # Use the first signal's stop loss/take profit as default
        stop_loss = signals[0].stop_loss if signals else None
        take_profit = signals[0].take_profit if signals else None
        position_size = signals[0].position_size if signals else None
        
        return StrategySignal(
            symbol=symbol,
            signal=final_signal,
            confidence=max(0.1, min(1.0, avg_confidence)),  # Clamp confidence between 0.1 and 1.0
            strength=avg_strength,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size
        )
        
    def get_strategy_performance(self) -> Dict[str, Dict]:
        """Get performance summary for all strategies"""
        performance = {}
        for strategy_id, strategy in self.strategies.items():
            performance[strategy_id] = strategy.get_performance_summary()
        return performance
        
    def get_active_signals_summary(self) -> Dict:
        """Get summary of all active signals"""
        summary = {}
        for symbol, symbol_signals in self.active_signals.items():
            symbol_summary = {}
            for strategy_id, signal in symbol_signals.items():
                symbol_summary[strategy_id] = {
                    'signal': signal.signal,
                    'confidence': signal.confidence,
                    'strength': signal.strength
                }
            summary[symbol] = symbol_summary
        return summary

# Global strategy manager instance
strategy_manager = None

def initialize_strategy_manager() -> StrategyManager:
    """Initialize global strategy manager with default strategies"""
    global strategy_manager
    strategy_manager = StrategyManager()
    
    # Add default strategies
    strategy_manager.add_strategy(MeanReversionStrategy(), weight=0.4)
    strategy_manager.add_strategy(MomentumStrategy(), weight=0.4)
    strategy_manager.add_strategy(RiskParityStrategy(), weight=0.2)
    
    logger.info("🎯 Strategy manager initialized with 3 institutional-grade strategies")
    return strategy_manager
# Forecast-Based Strategy Classes

class ForecastBasedStrategy(BaseStrategy):
    """Base class for strategies that use forecast information"""
    
    def __init__(self, strategy_id: str, name: str, description: str):
        super().__init__(strategy_id, name, description)
        self.min_forecast_confidence = 0.6
        self.risk_adjustment_factor = 1.0  # Adjust based on forecast uncertainty
    
    def generate_signal_from_forecast(self, symbol: str, forecast: Dict, 
                                   portfolio: Portfolio, data: pd.DataFrame) -> Optional[StrategySignal]:
        """Generate signal based on forecast and market data"""
        # Check forecast confidence
        if forecast.get('confidence', 0) < self.min_forecast_confidence:
            return StrategySignal(
                symbol=symbol,
                signal='HOLD',
                confidence=0.0,
                strength=0.0
            )
        
        # Determine signal based on expected return
        expected_return = forecast.get('expected_return', 0)
        confidence = forecast.get('confidence', 0.5)
        volatility = forecast.get('volatility', 0.02)  # Default 2% daily vol
        
        # Calculate signal strength based on expected return and confidence
        signal_strength = expected_return * confidence
        signal_confidence = confidence
        
        # Determine signal type
        if signal_strength > 0.001:  # Positive expectation
            signal = 'BUY'
            strength = min(1.0, signal_strength * 100)  # Scale appropriately
        elif signal_strength < -0.001:  # Negative expectation
            signal = 'SELL'
            strength = max(-1.0, signal_strength * 100)  # Scale appropriately
        else:
            signal = 'HOLD'
            strength = 0.0
            signal_confidence = 0.0
        
        # Calculate position size based on forecast confidence and volatility
        position_size = self._calculate_position_size_from_forecast(
            forecast, portfolio, symbol, data
        )
        
        # Calculate stops based on forecast uncertainty
        atr = self._calculate_atr(data) if len(data) > 10 else data['close'].iloc[-1] * 0.02
        stop_loss = self._calculate_stop_loss_from_forecast(
            data['close'].iloc[-1], signal, atr, forecast
        )
        take_profit = self._calculate_take_profit_from_forecast(
            data['close'].iloc[-1], signal, atr, forecast
        )
        
        return StrategySignal(
            symbol=symbol,
            signal=signal,
            confidence=signal_confidence,
            strength=strength,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size
        )
    
    def _calculate_position_size_from_forecast(self, forecast: Dict, 
                                            portfolio: Portfolio, 
                                            symbol: str, data: pd.DataFrame) -> float:
        """Calculate position size based on forecast confidence and risk"""
        # Base position size on forecast confidence and risk tolerance
        confidence = forecast.get('confidence', 0.5)
        volatility = forecast.get('volatility', 0.02)
        expected_return = forecast.get('expected_return', 0)
        
        # Calculate risk-adjusted position size
        # Higher confidence = larger position
        # Lower volatility = larger position (safer bet)
        base_size = min(0.1, max(0.01, confidence * 0.1))  # 1-10% of capital
        
        # Adjust for volatility - lower volatility allows larger positions
        vol_adjustment = max(0.5, min(2.0, 0.02 / volatility))  # Inverse relationship
        adjusted_size = base_size * vol_adjustment
        
        # Adjust based on expected return magnitude
        return_adjustment = min(2.0, max(0.5, 1 + abs(expected_return) * 10))
        final_size = adjusted_size * return_adjustment
        
        return final_size
    
    def _calculate_stop_loss_from_forecast(self, price: float, signal: str, 
                                         atr: float, forecast: Dict) -> Optional[float]:
        """Calculate stop loss based on forecast uncertainty"""
        # Use forecast confidence to adjust stop loss distance
        confidence = forecast.get('confidence', 0.5)
        
        # Less confident forecasts get wider stops
        stop_multiplier = max(1.5, min(3.0, 2.5 / confidence))
        stop_distance = atr * stop_multiplier
        
        if signal == 'BUY':
            return price - stop_distance
        elif signal == 'SELL':
            return price + stop_distance
        else:
            return None
    
    def _calculate_take_profit_from_forecast(self, price: float, signal: str, 
                                           atr: float, forecast: Dict) -> Optional[float]:
        """Calculate take profit based on forecast confidence"""
        # More confident forecasts can aim for higher rewards
        confidence = forecast.get('confidence', 0.5)
        expected_return = forecast.get('expected_return', 0)
        
        # Risk/reward ratio based on confidence
        rr_ratio = max(1.5, min(3.0, 1.5 + confidence))  # 1.5 to 3.0:1
        
        # Calculate take profit distance
        profit_distance = atr * 2 * rr_ratio  # Base on stop loss * rr_ratio
        
        if signal == 'BUY':
            return price + profit_distance
        elif signal == 'SELL':
            return price - profit_distance
        else:
            return None


class AdaptivePositionSizingStrategy(ForecastBasedStrategy):
    """Strategy that adjusts position size based on forecast confidence"""
    
    def __init__(self):
        super().__init__(
            strategy_id='adaptive_position_v1',
            name='Adaptive Position Sizing Strategy',
            description='Adjusts position size based on forecast confidence and market conditions'
        )
        self.base_risk_per_trade = 0.02  # 2% risk per trade
        self.max_position_size = 0.10    # 10% max position size
    
    def generate_signal_from_forecast(self, symbol: str, forecast: Dict, 
                                   portfolio: Portfolio, data: pd.DataFrame) -> Optional[StrategySignal]:
        """Generate signal with adaptive position sizing"""
        base_signal = super().generate_signal_from_forecast(symbol, forecast, portfolio, data)
        
        if base_signal is None:
            return None
        
        # Further refine position size based on additional factors
        refined_size = self._refine_position_size(
            forecast, portfolio, symbol, data, base_signal
        )
        
        # Update the position size
        base_signal.position_size = refined_size
        
        return base_signal
    
    def _refine_position_size(self, forecast: Dict, portfolio: Portfolio, 
                            symbol: str, data: pd.DataFrame, signal: StrategySignal) -> float:
        """Refine position size based on additional market factors"""
        base_size = signal.position_size if signal.position_size else 0.05  # Default 5%
        
        # Adjust for portfolio concentration risk
        current_exposure = self._calculate_symbol_exposure(portfolio, symbol)
        if current_exposure > 0.3:  # Already 30%+ in this symbol
            base_size *= 0.5  # Reduce size by half
        
        # Adjust for overall portfolio risk
        portfolio_risk_level = self._assess_portfolio_risk(portfolio)
        if portfolio_risk_level == 'HIGH':
            base_size *= 0.7  # Reduce by 30%
        elif portfolio_risk_level == 'VERY_HIGH':
            base_size *= 0.4  # Reduce by 60%
        
        # Confidence adjustment - higher confidence allows larger positions
        confidence = forecast.get('confidence', 0.5)
        confidence_adjustment = min(2.0, max(0.5, confidence / 0.6))  # 0.5 to 2.0 multiplier
        final_size = base_size * confidence_adjustment
        
        # Ensure within bounds
        final_size = min(self.max_position_size, max(0.001, final_size))
        
        return final_size
    
    def _calculate_symbol_exposure(self, portfolio: Portfolio, symbol: str) -> float:
        """Calculate current exposure to a particular symbol"""
        current_position = portfolio.get_position(symbol)
        if not current_position:
            return 0.0
        
        current_value = abs(current_position.quantity * current_position.current_price)
        total_portfolio_value = portfolio.calculate_total_value()
        
        return current_value / total_portfolio_value if total_portfolio_value > 0 else 0.0
    
    def _assess_portfolio_risk(self, portfolio: Portfolio) -> str:
        """Assess overall portfolio risk level"""
        # Placeholder implementation - in practice would analyze portfolio composition
        return 'MEDIUM'


# Enhanced Strategy Manager with Forecast Integration

class EnhancedStrategyManager(StrategyManager):
    """Strategy manager with forecast integration capabilities"""
    
    def __init__(self):
        super().__init__()
        self.forecast_strategies = {}  # Strategies that use forecasts
        self.forecast_weights = {}     # Weights for forecast-based strategies
    
    def add_forecast_strategy(self, strategy: ForecastBasedStrategy, weight: float = 1.0):
        """Add a strategy that uses forecasts"""
        self.forecast_strategies[strategy.strategy_id] = strategy
        self.forecast_weights[strategy.strategy_id] = weight
        logger.info(f"🔮 Added forecast-based strategy: {strategy.name} (weight: {weight})")
    
    def generate_signals_from_forecast(self, symbol: str, forecast: Dict, 
                                    portfolio: Portfolio, data: pd.DataFrame) -> List[StrategySignal]:
        """Generate signals from forecast-based strategies"""
        signals = []
        
        for strategy_id, strategy in self.forecast_strategies.items():
            try:
                signal = strategy.generate_signal_from_forecast(symbol, forecast, portfolio, data)
                if signal:
                    signals.append(signal)
                    # Store active signal
                    if symbol not in self.active_signals:
                        self.active_signals[symbol] = {}
                    self.active_signals[symbol][strategy_id] = signal
            except Exception as e:
                logger.error(f"❌ Error in forecast strategy {strategy_id}: {e}")
        
        return signals
    
    def combine_forecast_signals(self, symbol: str, forecast_signals: List[StrategySignal], 
                              traditional_signals: List[StrategySignal]) -> Optional[StrategySignal]:
        """Combine forecast-based and traditional signals"""
        all_signals = forecast_signals + traditional_signals
        
        if not all_signals:
            return None
        
        # Weighted combination considering signal source
        total_weight = 0.0
        weighted_strength = 0.0
        total_confidence = 0.0
        
        for signal in all_signals:
            # Determine if this is a forecast-based signal
            is_forecast_signal = any(fs.symbol == signal.symbol for fs in forecast_signals)
            weight = 1.2 if is_forecast_signal else 1.0  # Give slight edge to forecast signals
            weight *= self.forecast_weights.get(signal.strategy_id, 1.0) if is_forecast_signal else self.weights.get(signal.strategy_id, 1.0)
            
            total_weight += weight
            weighted_strength += signal.strength * weight
            total_confidence += signal.confidence * weight
        
        if total_weight == 0:
            return None
        
        avg_strength = weighted_strength / total_weight
        avg_confidence = total_confidence / total_weight
        
        # Determine final signal
        if abs(avg_strength) > 0.05:  # Threshold to avoid weak signals
            if avg_strength > 0:
                final_signal = 'BUY'
            else:
                final_signal = 'SELL'
        else:
            final_signal = 'HOLD'
        
        return StrategySignal(
            symbol=symbol,
            signal=final_signal,
            confidence=max(0.0, min(1.0, avg_confidence)),
            strength=avg_strength
        )
