"""
Meta Allocator for Adaptive Institutional AI Trader
Manages capital allocation across strategies based on forecast confidence and regime
"""

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass

from services.forecast.forecast_service import ForecastService
from risk.regime_engine import RegimeDetectionEngine, RegimeState
from strategies.advanced_strategies import StrategyManager

logger = logging.getLogger(__name__)

@dataclass
class AllocationDecision:
    """Result of allocation decision"""
    strategy_weights: Dict[str, float]
    total_capital: float
    regime: RegimeState
    overall_confidence: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class MetaAllocator:
    """Meta allocator that manages capital allocation across strategies"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.forecast_service = ForecastService()
        self.regime_engine = RegimeDetectionEngine()
        self.strategy_manager = StrategyManager()
        
        # Allocation weights for different regimes
        self.regime_allocations = {
            RegimeState.TREND: {'momentum': 0.6, 'mean_reversion': 0.3, 'risk_parity': 0.1},
            RegimeState.MEAN_REVERT: {'mean_reversion': 0.6, 'momentum': 0.2, 'risk_parity': 0.2},
            RegimeState.CRISIS: {'risk_parity': 0.7, 'mean_reversion': 0.2, 'momentum': 0.1},
            RegimeState.HIGH_VOLATILITY: {'risk_parity': 0.5, 'mean_reversion': 0.3, 'momentum': 0.2},
            RegimeState.LOW_VOLATILITY: {'momentum': 0.4, 'mean_reversion': 0.3, 'risk_parity': 0.3},
            RegimeState.LOW_LIQUIDITY: {'risk_parity': 0.8, 'mean_reversion': 0.1, 'momentum': 0.1},
            RegimeState.NORMAL: {'momentum': 0.4, 'mean_reversion': 0.4, 'risk_parity': 0.2}
        }
        
        # Performance tracking
        self.allocation_history = []
        self.performance_metrics = {}
        
        # Risk adjustments
        self.max_single_strategy_allocation = 0.7  # Max 70% to any single strategy
        self.min_allocation_threshold = 0.05       # Min 5% allocation
        
        logger.info(f"💰 Meta Allocator initialized with ${initial_capital:,.2f}")
    
    async def allocate_capital(self, symbols: List[str], data: Dict[str, pd.DataFrame]) -> AllocationDecision:
        """
        Allocate capital across strategies based on forecasts and regime
        
        Args:
            symbols: List of trading symbols
            data: Dictionary of market data for each symbol
            
        Returns:
            AllocationDecision with strategy weights
        """
        try:
            logger.info(f"📊 Allocating capital across {len(symbols)} symbols")
            
            # Detect regime
            regime, regime_confidence = await self._detect_regime(data)
            
            # Get forecasts for each symbol
            forecasts = await self._get_forecasts(symbols)
            
            # Calculate base allocations based on regime
            base_allocations = self._get_regime_based_allocations(regime)
            
            # Adjust allocations based on forecast confidence
            adjusted_allocations = self._adjust_allocations_for_forecasts(
                base_allocations, forecasts
            )
            
            # Further adjust based on drawdown and performance
            final_allocations = self._adjust_for_drawdown(adjusted_allocations)
            
            # Normalize to sum to 1.0
            final_allocations = self._normalize_allocations(final_allocations)
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(
                forecasts, regime_confidence
            )
            
            decision = AllocationDecision(
                strategy_weights=final_allocations,
                total_capital=self.current_capital,
                regime=regime,
                overall_confidence=overall_confidence
            )
            
            # Track allocation
            self.allocation_history.append(decision)
            
            logger.info(f"✅ Allocation completed: {final_allocations}")
            logger.info(f"   Regime: {regime.value}, Confidence: {overall_confidence:.2f}")
            
            return decision
            
        except Exception as e:
            logger.error(f"❌ Capital allocation failed: {e}")
            # Return default allocation in case of error
            default_allocations = {'risk_parity': 0.5, 'mean_reversion': 0.3, 'momentum': 0.2}
            return AllocationDecision(
                strategy_weights=default_allocations,
                total_capital=self.current_capital,
                regime=RegimeState.NORMAL,
                overall_confidence=0.5
            )
    
    async def _detect_regime(self, data: Dict[str, pd.DataFrame]) -> Tuple[RegimeState, float]:
        """Detect market regime from data"""
        try:
            # Use the first symbol's data for regime detection
            # In practice, you might aggregate across symbols
            first_symbol = list(data.keys())[0]
            df = data[first_symbol]
            
            regime, confidence, _ = self.regime_engine.detect_regime(df, first_symbol)
            
            logger.debug(f"🔄 Regime detected: {regime.value} (conf: {confidence:.2f})")
            return regime, confidence
            
        except Exception as e:
            logger.warning(f"⚠️ Regime detection failed: {e}, using NORMAL")
            return RegimeState.NORMAL, 0.5
    
    async def _get_forecasts(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get forecasts for all symbols"""
        forecasts = {}
        
        for symbol in symbols:
            forecast_event = await self.forecast_service.generate_forecast(symbol)
            if forecast_event:
                forecasts[symbol] = {
                    'expected_return': forecast_event.expected_return,
                    'confidence': forecast_event.confidence,
                    'volatility': forecast_event.volatility,
                    'p10': forecast_event.p10,
                    'p50': forecast_event.p50,
                    'p90': forecast_event.p90
                }
        
        logger.debug(f"🔮 Forecasts obtained for {len(forecasts)} symbols")
        return forecasts
    
    def _get_regime_based_allocations(self, regime: RegimeState) -> Dict[str, float]:
        """Get base allocations based on regime"""
        return self.regime_allocations.get(regime, self.regime_allocations[RegimeState.NORMAL]).copy()
    
    def _adjust_allocations_for_forecasts(self, base_allocations: Dict[str, float], 
                                        forecasts: Dict[str, Dict]) -> Dict[str, float]:
        """Adjust allocations based on forecast confidence"""
        adjusted = base_allocations.copy()
        
        # If we have forecasts, adjust based on expected returns and confidence
        if forecasts:
            # Calculate average expected return and confidence
            avg_expected_return = np.mean([f['expected_return'] for f in forecasts.values()])
            avg_confidence = np.mean([f['confidence'] for f in forecasts.values()])
            
            # Boost allocations to strategies that align with market forecast
            for strategy_name in adjusted:
                # Adjust based on market conditions
                if avg_expected_return > 0.001:  # Positive trend expected
                    if strategy_name == 'momentum':
                        adjusted[strategy_name] *= 1.1  # Boost momentum in uptrend
                    elif strategy_name == 'mean_reversion':
                        adjusted[strategy_name] *= 0.9  # Reduce mean reversion in uptrend
                elif avg_expected_return < -0.001:  # Negative trend expected
                    if strategy_name == 'mean_reversion':
                        adjusted[strategy_name] *= 1.1  # Boost mean reversion in downtrend
                    elif strategy_name == 'momentum':
                        adjusted[strategy_name] *= 0.9  # Reduce momentum in downtrend
                
                # Adjust based on confidence
                confidence_factor = avg_confidence * 2  # Amplify confidence effect
                adjusted[strategy_name] *= (0.8 + 0.4 * confidence_factor)  # Scale between 0.8 and 1.2
        
        return adjusted
    
    def _adjust_for_drawdown(self, allocations: Dict[str, float]) -> Dict[str, float]:
        """Adjust allocations based on strategy drawdown"""
        adjusted = allocations.copy()
        
        # Get strategy performance (in practice, this would come from performance tracking)
        # For now, we'll use a simple approach
        for strategy_name in adjusted:
            # Placeholder: adjust based on some performance metric
            # In reality, you'd use actual performance data
            perf_factor = 1.0  # Placeholder
            adjusted[strategy_name] *= perf_factor
        
        return adjusted
    
    def _normalize_allocations(self, allocations: Dict[str, float]) -> Dict[str, float]:
        """Normalize allocations to sum to 1.0 with constraints"""
        # Apply constraints
        constrained = {}
        total = 0.0
        
        for strategy, weight in allocations.items():
            # Apply max allocation constraint
            clamped_weight = min(weight, self.max_single_strategy_allocation)
            # Apply min allocation threshold
            clamped_weight = max(clamped_weight, self.min_allocation_threshold if clamped_weight > 0 else 0)
            
            constrained[strategy] = clamped_weight
            total += clamped_weight
        
        # Normalize to sum to 1.0
        if total > 0:
            normalized = {k: v/total for k, v in constrained.items()}
        else:
            # Fallback to equal allocation
            n_strats = len(constrained)
            normalized = {k: 1.0/n_strats for k in constrained.keys()} if n_strats > 0 else {}
        
        return normalized
    
    def _calculate_overall_confidence(self, forecasts: Dict[str, Dict], 
                                   regime_confidence: float) -> float:
        """Calculate overall confidence in allocation decision"""
        if not forecasts:
            return regime_confidence * 0.7  # Lower confidence without forecasts
        
        # Average forecast confidence
        forecast_confidence = np.mean([f['confidence'] for f in forecasts.values()])
        
        # Weighted average of regime and forecast confidence
        overall_confidence = 0.6 * forecast_confidence + 0.4 * regime_confidence
        
        return max(0.0, min(1.0, overall_confidence))  # Clamp between 0 and 1
    
    async def rebalance_if_needed(self, symbols: List[str], data: Dict[str, pd.DataFrame], 
                                rebalance_threshold: float = 0.1) -> Optional[AllocationDecision]:
        """Rebalance if allocation deviates significantly from target"""
        try:
            # Get current allocation
            current_allocation = await self.allocate_capital(symbols, data)
            
            # Compare with previous allocation if available
            if self.allocation_history:
                prev_allocation = self.allocation_history[-1]
                
                # Calculate deviation
                deviations = {}
                all_strategies = set(current_allocation.strategy_weights.keys()) | \
                               set(prev_allocation.strategy_weights.keys())
                
                for strat in all_strategies:
                    curr_weight = current_allocation.strategy_weights.get(strat, 0)
                    prev_weight = prev_allocation.strategy_weights.get(strat, 0)
                    deviations[strat] = abs(curr_weight - prev_weight)
                
                max_deviation = max(deviations.values()) if deviations else 0
                
                if max_deviation > rebalance_threshold:
                    logger.info(f"🔄 Rebalancing triggered: max deviation {max_deviation:.2f} > {rebalance_threshold}")
                    return current_allocation
                else:
                    logger.debug(f"📊 No rebalancing needed: max deviation {max_deviation:.2f} <= {rebalance_threshold}")
                    return None
            else:
                return current_allocation
                
        except Exception as e:
            logger.error(f"❌ Rebalancing check failed: {e}")
            return None
    
    def get_allocation_summary(self) -> Dict:
        """Get summary of current allocation state"""
        if not self.allocation_history:
            return {
                'current_allocation': {},
                'regime': RegimeState.NORMAL.value,
                'last_rebalanced': None,
                'allocation_confidence': 0.5
            }
        
        last_decision = self.allocation_history[-1]
        
        return {
            'current_allocation': last_decision.strategy_weights,
            'regime': last_decision.regime.value,
            'last_rebalanced': last_decision.timestamp.isoformat(),
            'allocation_confidence': last_decision.overall_confidence,
            'total_capital': last_decision.total_capital
        }
    
    def update_capital(self, realized_pnl: float):
        """Update capital based on realized PnL"""
        self.current_capital += realized_pnl
        logger.debug(f"💳 Capital updated: ${self.current_capital:,.2f} (PnL: ${realized_pnl:,.2f})")
    
    def get_strategy_performance(self) -> Dict[str, Dict]:
        """Get performance metrics for each strategy"""
        # Placeholder - in practice would track strategy-specific performance
        return {
            'momentum': {
                'allocation': 0.4,
                'return': 0.05,
                'sharpe': 1.2,
                'drawdown': 0.03
            },
            'mean_reversion': {
                'allocation': 0.4,
                'return': 0.04,
                'sharpe': 1.1,
                'drawdown': 0.02
            },
            'risk_parity': {
                'allocation': 0.2,
                'return': 0.03,
                'sharpe': 1.0,
                'drawdown': 0.01
            }
        }

class AdaptiveCapitalAllocator(MetaAllocator):
    """Enhanced allocator with adaptive learning capabilities"""
    
    def __init__(self, initial_capital: float = 100000.0):
        super().__init__(initial_capital)
        
        # Decision logging for learning
        self.decision_log = []
        
        # Performance attribution
        self.performance_attribution = {}
        
        logger.info("🤖 Adaptive Capital Allocator initialized")
    
    async def make_adaptive_allocation(self, symbols: List[str], data: Dict[str, pd.DataFrame]) -> AllocationDecision:
        """Make allocation decision with learning from past decisions"""
        decision = await self.allocate_capital(symbols, data)
        
        # Log decision for learning
        self._log_decision(decision, data)
        
        return decision
    
    def _log_decision(self, decision: AllocationDecision, data: Dict[str, pd.DataFrame]):
        """Log decision for future learning"""
        decision_record = {
            'timestamp': decision.timestamp,
            'strategy_weights': decision.strategy_weights.copy(),
            'regime': decision.regime.value,
            'overall_confidence': decision.overall_confidence,
            'market_conditions': self._extract_market_conditions(data),
            'symbols': list(data.keys()),
            'capital': decision.total_capital
        }
        
        self.decision_log.append(decision_record)
        
        # Keep only recent decisions to manage memory
        if len(self.decision_log) > 1000:
            self.decision_log = self.decision_log[-500:]  # Keep last 500 decisions
    
    def _extract_market_conditions(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """Extract market conditions for decision logging"""
        if not data:
            return {}
        
        # Use first symbol as representative
        symbol = list(data.keys())[0]
        df = data[symbol]
        
        if len(df) < 2:
            return {}
        
        returns = df['close'].pct_change().dropna()
        
        conditions = {
            'volatility': returns.std() if len(returns) > 0 else 0,
            'trend': df['close'].iloc[-1] / df['close'].rolling(20).mean().iloc[-1] - 1 if len(df) >= 20 else 0,
            'volume_regime': df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1] if 'volume' in df.columns and len(df) >= 20 else 1,
            'price_position': (df['close'].iloc[-1] - df['close'].rolling(20).min().iloc[-1]) / 
                             (df['close'].rolling(20).max().iloc[-1] - df['close'].rolling(20).min().iloc[-1]) if len(df) >= 20 else 0.5
        }
        
        return conditions
    
    def get_learning_insights(self) -> Dict:
        """Get insights from decision patterns"""
        if len(self.decision_log) < 10:
            return {"message": "Need more decisions to derive insights"}
        
        # Analyze decision patterns
        recent_decisions = self.decision_log[-50:]  # Last 50 decisions
        
        insights = {
            'most_allocated_strategies': self._get_top_strategies(recent_decisions),
            'regime_distribution': self._get_regime_distribution(recent_decisions),
            'confidence_trends': self._get_confidence_trends(recent_decisions)
        }
        
        return insights
    
    def _get_top_strategies(self, decisions: List[Dict]) -> Dict:
        """Get most allocated strategies in recent decisions"""
        strategy_totals = {}
        total_decisions = len(decisions)
        
        for decision in decisions:
            for strategy, weight in decision['strategy_weights'].items():
                strategy_totals[strategy] = strategy_totals.get(strategy, 0) + weight
        
        # Normalize by number of decisions
        for strategy in strategy_totals:
            strategy_totals[strategy] /= total_decisions
        
        # Sort by allocation
        sorted_strategies = sorted(strategy_totals.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_strategies[:3])  # Top 3
    
    def _get_regime_distribution(self, decisions: List[Dict]) -> Dict:
        """Get distribution of regimes in recent decisions"""
        regime_counts = {}
        for decision in decisions:
            regime = decision['regime']
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        # Convert to percentages
        total = len(decisions)
        return {regime: count/total for regime, count in regime_counts.items()}
    
    def _get_confidence_trends(self, decisions: List[Dict]) -> Dict:
        """Get trends in decision confidence"""
        confidences = [d['overall_confidence'] for d in decisions]
        
        return {
            'average_confidence': np.mean(confidences),
            'confidence_volatility': np.std(confidences),
            'latest_confidence': confidences[-1] if confidences else 0.5,
            'confidence_trend': 'increasing' if len(confidences) > 1 and confidences[-1] > confidences[-2] else 'decreasing'
        }