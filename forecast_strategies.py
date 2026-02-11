"""
Forecast-Based Trading Strategies
Strategies that consume only ForecastEvent from the centralized pipeline
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass

from portfolio.portfolio import Portfolio
from data_pipeline import get_data_pipeline

logger = logging.getLogger(__name__)

@dataclass
class StrategySignal:
    """Trading signal from strategy"""
    symbol: str
    signal: str  # BUY, SELL, HOLD
    confidence: float  # 0.0 to 1.0
    strength: float  # -1.0 to 1.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[float] = None
    expected_return: Optional[float] = None
    volatility: Optional[float] = None

class BaseForecastStrategy(ABC):
    """Base class for strategies that use only forecast data"""
    
    def __init__(self, strategy_id: str, name: str, description: str):
        self.strategy_id = strategy_id
        self.name = name
        self.description = description
        self.data_pipeline = get_data_pipeline()
        self.active_positions = {}
        self.performance_metrics = {}
        
    @abstractmethod
    async def generate_signal(self, symbol: str, 
                            portfolio: Portfolio) -> Optional[StrategySignal]:
        """Generate trading signal based ONLY on forecast data"""
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

class PureForecastStrategy(BaseForecastStrategy):
    """Pure forecast-based strategy - uses only forecast signals"""
    
    def __init__(self, min_confidence: float = 0.6, risk_per_trade: float = 0.02):
        super().__init__(
            strategy_id='pure_forecast_v1',
            name='Pure Forecast Strategy',
            description='Uses only forecast data for trading decisions'
        )
        self.min_confidence = min_confidence
        self.risk_per_trade = risk_per_trade
    
    async def generate_signal(self, symbol: str, 
                            portfolio: Portfolio) -> Optional[StrategySignal]:
        """Generate signal based ONLY on forecast data"""
        try:
            # ONLY use the centralized pipeline - no direct market data access
            trading_signal = await self.data_pipeline.get_trading_signal(symbol, portfolio)
            
            if trading_signal is None:
                logger.warning(f"No forecast signal available for {symbol}")
                return StrategySignal(
                    symbol=symbol,
                    signal='HOLD',
                    confidence=0.0,
                    strength=0.0
                )
            
            # Check minimum confidence requirement
            if trading_signal['confidence'] < self.min_confidence:
                return StrategySignal(
                    symbol=symbol,
                    signal='HOLD',
                    confidence=trading_signal['confidence'],
                    strength=0.0
                )
            
            # Create strategy signal from forecast
            return StrategySignal(
                symbol=symbol,
                signal=trading_signal['signal'],
                confidence=trading_signal['confidence'],
                strength=trading_signal['strength'],
                stop_loss=trading_signal['stop_loss'],
                take_profit=trading_signal['take_profit'],
                position_size=trading_signal['position_size'],
                expected_return=trading_signal['expected_return'],
                volatility=trading_signal['volatility']
            )
            
        except Exception as e:
            logger.error(f"❌ Error in pure forecast strategy for {symbol}: {e}")
            return StrategySignal(
                symbol=symbol,
                signal='HOLD',
                confidence=0.0,
                strength=0.0
            )

class ConfidenceWeightedStrategy(BaseForecastStrategy):
    """Strategy that weights positions based on forecast confidence"""
    
    def __init__(self, confidence_thresholds: Dict[str, float] = None):
        super().__init__(
            strategy_id='confidence_weighted_v1',
            name='Confidence Weighted Strategy',
            description='Adjusts position sizes based on forecast confidence levels'
        )
        
        self.confidence_thresholds = confidence_thresholds or {
            'LOW': 0.5,      # 50% confidence minimum
            'MEDIUM': 0.7,   # 70% confidence for medium positions
            'HIGH': 0.85     # 85% confidence for large positions
        }
    
    async def generate_signal(self, symbol: str, 
                            portfolio: Portfolio) -> Optional[StrategySignal]:
        """Generate signal with confidence-weighted position sizing"""
        try:
            # Get forecast signal
            forecast_signal = await self.data_pipeline.get_trading_signal(symbol, portfolio)
            
            if forecast_signal is None:
                return StrategySignal(
                    symbol=symbol,
                    signal='HOLD',
                    confidence=0.0,
                    strength=0.0
                )
            
            confidence = forecast_signal['confidence']
            base_signal = forecast_signal['signal']
            
            # Determine confidence level
            if confidence >= self.confidence_thresholds['HIGH']:
                confidence_level = 'HIGH'
                position_multiplier = 1.5
            elif confidence >= self.confidence_thresholds['MEDIUM']:
                confidence_level = 'MEDIUM'
                position_multiplier = 1.0
            elif confidence >= self.confidence_thresholds['LOW']:
                confidence_level = 'LOW'
                position_multiplier = 0.5
            else:
                # Below minimum threshold
                return StrategySignal(
                    symbol=symbol,
                    signal='HOLD',
                    confidence=confidence,
                    strength=0.0
                )
            
            # Adjust position size based on confidence
            adjusted_position_size = forecast_signal['position_size'] * position_multiplier
            
            # Create final signal
            return StrategySignal(
                symbol=symbol,
                signal=base_signal,
                confidence=confidence,
                strength=forecast_signal['strength'],
                stop_loss=forecast_signal['stop_loss'],
                take_profit=forecast_signal['take_profit'],
                position_size=adjusted_position_size,
                expected_return=forecast_signal['expected_return'],
                volatility=forecast_signal['volatility']
            )
            
        except Exception as e:
            logger.error(f"❌ Error in confidence-weighted strategy for {symbol}: {e}")
            return StrategySignal(
                symbol=symbol,
                signal='HOLD',
                confidence=0.0,
                strength=0.0
            )

class ConservativeForecastStrategy(BaseForecastStrategy):
    """Conservative strategy that requires high confidence and additional filters"""
    
    def __init__(self, min_confidence: float = 0.8, 
                 min_expected_return: float = 0.005,  # 0.5% minimum expected return
                 max_volatility: float = 0.03):       # 3% maximum volatility
        super().__init__(
            strategy_id='conservative_forecast_v1',
            name='Conservative Forecast Strategy',
            description='High-confidence strategy with additional risk filters'
        )
        self.min_confidence = min_confidence
        self.min_expected_return = min_expected_return
        self.max_volatility = max_volatility
    
    async def generate_signal(self, symbol: str, 
                            portfolio: Portfolio) -> Optional[StrategySignal]:
        """Generate conservative signal with multiple filters"""
        try:
            # Get forecast signal
            forecast_signal = await self.data_pipeline.get_trading_signal(symbol, portfolio)
            
            if forecast_signal is None:
                return StrategySignal(
                    symbol=symbol,
                    signal='HOLD',
                    confidence=0.0,
                    strength=0.0
                )
            
            # Apply multiple filters
            filters_passed = []
            
            # 1. Confidence filter
            confidence_check = forecast_signal['confidence'] >= self.min_confidence
            filters_passed.append(('confidence', confidence_check))
            
            # 2. Expected return filter
            expected_return_check = abs(forecast_signal['expected_return']) >= self.min_expected_return
            filters_passed.append(('expected_return', expected_return_check))
            
            # 3. Volatility filter
            volatility_check = forecast_signal['volatility'] <= self.max_volatility
            filters_passed.append(('volatility', volatility_check))
            
            # Check if all filters passed
            all_filters_passed = all(check for _, check in filters_passed)
            
            if not all_filters_passed:
                failed_filters = [name for name, passed in filters_passed if not passed]
                logger.info(f"❌ {symbol} failed filters: {failed_filters}")
                
                return StrategySignal(
                    symbol=symbol,
                    signal='HOLD',
                    confidence=forecast_signal['confidence'],
                    strength=0.0
                )
            
            # All filters passed - generate trading signal
            logger.info(f"✅ {symbol} passed all conservative filters")
            
            return StrategySignal(
                symbol=symbol,
                signal=forecast_signal['signal'],
                confidence=forecast_signal['confidence'],
                strength=forecast_signal['strength'],
                stop_loss=forecast_signal['stop_loss'],
                take_profit=forecast_signal['take_profit'],
                position_size=forecast_signal['position_size'] * 0.8,  # Even more conservative sizing
                expected_return=forecast_signal['expected_return'],
                volatility=forecast_signal['volatility']
            )
            
        except Exception as e:
            logger.error(f"❌ Error in conservative strategy for {symbol}: {e}")
            return StrategySignal(
                symbol=symbol,
                signal='HOLD',
                confidence=0.0,
                strength=0.0
            )

class StrategyManager:
    """Manages forecast-based strategies"""
    
    def __init__(self):
        self.strategies: Dict[str, BaseForecastStrategy] = {}
        self.weights: Dict[str, float] = {}
        self.active_signals: Dict[str, Dict[str, StrategySignal]] = {}
        
    def add_strategy(self, strategy: BaseForecastStrategy, weight: float = 1.0):
        """Add forecast-based strategy"""
        self.strategies[strategy.strategy_id] = strategy
        self.weights[strategy.strategy_id] = weight
        logger.info(f"✅ Added forecast strategy: {strategy.name} (weight: {weight})")
        
    async def generate_signals(self, symbol: str, 
                             portfolio: Portfolio) -> List[StrategySignal]:
        """Generate signals from all strategies for a symbol"""
        signals = []
        
        for strategy_id, strategy in self.strategies.items():
            try:
                signal = await strategy.generate_signal(symbol, portfolio)
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
        expected_returns = []
        volatilities = []
        
        for signal in signals:
            weight = self.weights.get(signal.strategy_id, 1.0)
            total_weight += weight
            weighted_strength += signal.strength * weight
            total_confidence += signal.confidence * weight
            
            if signal.expected_return is not None:
                expected_returns.append(signal.expected_return)
            if signal.volatility is not None:
                volatilities.append(signal.volatility)
        
        if total_weight == 0:
            return None
            
        avg_strength = weighted_strength / total_weight
        avg_confidence = total_confidence / total_weight
        avg_expected_return = np.mean(expected_returns) if expected_returns else 0.0
        avg_volatility = np.mean(volatilities) if volatilities else 0.02
        
        # Determine final signal based on average strength
        if abs(avg_strength) > 0.05:  # Threshold to avoid weak signals
            if avg_strength > 0:
                final_signal = 'BUY'
            else:
                final_signal = 'SELL'
        else:
            final_signal = 'HOLD'
            
        # Use consensus values for stops and position sizing
        stop_loss = np.mean([s.stop_loss for s in signals if s.stop_loss is not None])
        take_profit = np.mean([s.take_profit for s in signals if s.take_profit is not None])
        position_size = np.mean([s.position_size for s in signals if s.position_size is not None])
        
        return StrategySignal(
            symbol=symbol,
            signal=final_signal,
            confidence=max(0.0, min(1.0, avg_confidence)),
            strength=avg_strength,
            stop_loss=stop_loss if not np.isnan(stop_loss) else None,
            take_profit=take_profit if not np.isnan(take_profit) else None,
            position_size=position_size if not np.isnan(position_size) else None,
            expected_return=avg_expected_return,
            volatility=avg_volatility
        )
        
    def get_strategy_performance(self) -> Dict[str, Dict]:
        """Get performance summary for all strategies"""
        performance = {}
        for strategy_id, strategy in self.strategies.items():
            performance[strategy_id] = strategy.get_performance_summary()
        return performance

# Global strategy manager instance
strategy_manager = None

def initialize_forecast_strategy_manager() -> StrategyManager:
    """Initialize global strategy manager with forecast-based strategies"""
    global strategy_manager
    strategy_manager = StrategyManager()
    
    # Add default forecast-based strategies
    strategy_manager.add_strategy(PureForecastStrategy(min_confidence=0.6), weight=0.5)
    strategy_manager.add_strategy(ConfidenceWeightedStrategy(), weight=0.3)
    strategy_manager.add_strategy(ConservativeForecastStrategy(), weight=0.2)
    
    logger.info("🎯 Forecast strategy manager initialized with 3 pure forecast strategies")
    return strategy_manager

# Backward compatibility
async def get_strategy_signal(symbol: str, portfolio=None) -> Optional[StrategySignal]:
    """Get signal from the default forecast strategy"""
    manager = initialize_forecast_strategy_manager()
    signals = await manager.generate_signals(symbol, portfolio or Portfolio())
    return manager.combine_signals(symbol, signals)