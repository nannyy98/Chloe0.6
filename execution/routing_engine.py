"""
Order Routing Engine
Implements smart order routing and execution optimization
"""
import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import time
import statistics
from dataclasses import dataclass

from execution.adapters.base_broker_adapter import BaseBrokerAdapter, BrokerAdapterManager
from execution.latency_monitor import LatencyMonitor

logger = logging.getLogger(__name__)


@dataclass
class RouteMetrics:
    """Metrics for a specific route/broker"""
    broker_name: str
    success_rate: float
    avg_execution_time: float
    avg_slippage: float
    min_notional: float
    fees: Dict[str, float]
    available_liquidity: float
    latency: float


@dataclass
class RoutingDecision:
    """Result of routing decision"""
    primary_broker: str
    secondary_brokers: List[str]
    allocation_percentage: Dict[str, float]  # For order splitting
    reason: str
    confidence: float


class OrderRoutingEngine:
    """Smart order routing engine for optimal execution"""
    
    def __init__(self):
        self.broker_manager = BrokerAdapterManager()
        self.latency_monitor = LatencyMonitor()
        self.route_metrics: Dict[str, RouteMetrics] = {}
        self.execution_history = []
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize the routing engine"""
        # Connect to all registered brokers
        await self.broker_manager.connect_all()
        
        # Start monitoring latencies
        await self.latency_monitor.start_monitoring()
        
        self.is_initialized = True
        logger.info("🚦 Order Routing Engine initialized")
    
    def register_broker(self, name: str, adapter: BaseBrokerAdapter, is_default: bool = False):
        """Register a broker with the routing engine"""
        self.broker_manager.register_adapter(name, adapter, is_default)
    
    async def get_route_metrics(self, symbol: str) -> Dict[str, RouteMetrics]:
        """Get current metrics for all available routes"""
        metrics = {}
        
        for broker_name, broker_adapter in self.broker_manager.get_all_adapters().items():
            try:
                # Get broker-specific metrics
                latency = self.latency_monitor.get_average_latency(broker_name)
                
                # Get account balance to determine available liquidity
                balance = await broker_adapter.get_account_balance()
                
                # Estimate available liquidity (simplified)
                available_liquidity = 0
                for currency, details in balance.items():
                    if isinstance(details, dict):
                        available_liquidity += details.get('free', 0)
                
                # Get current market price for slippage calculation
                market_price = await broker_adapter.get_market_price(symbol)
                
                # Calculate fees (simplified)
                fees = {'maker': 0.001, 'taker': 0.001}  # 0.1% for both maker/taker
                
                # Set min notional (simplified)
                min_notional = 10.0  # $10 minimum
                
                metrics[broker_name] = RouteMetrics(
                    broker_name=broker_name,
                    success_rate=0.95,  # Default assumption
                    avg_execution_time=0.2,  # 200ms default
                    avg_slippage=0.0005,  # 0.05% default
                    min_notional=min_notional,
                    fees=fees,
                    available_liquidity=available_liquidity,
                    latency=latency
                )
                
            except Exception as e:
                logger.error(f"❌ Error getting metrics for {broker_name}: {e}")
                # Still add with default values
                metrics[broker_name] = RouteMetrics(
                    broker_name=broker_name,
                    success_rate=0.80,
                    avg_execution_time=1.0,
                    avg_slippage=0.001,
                    min_notional=10.0,
                    fees={'maker': 0.0015, 'taker': 0.002},
                    available_liquidity=0,
                    latency=999.0
                )
        
        return metrics
    
    def calculate_routing_score(self, broker_name: str, metrics: RouteMetrics, 
                              symbol: str, side: str, quantity: float, price: Optional[float] = None) -> float:
        """Calculate routing score for a specific broker based on various factors"""
        score = 0.0
        
        # Factor 1: Success rate (higher is better) - weight 25%
        score += metrics.success_rate * 0.25
        
        # Factor 2: Low latency (lower is better) - weight 25%
        # Normalize latency to 0-1 scale (lower is better)
        normalized_latency = max(0, min(1, 1 - (metrics.latency / 1000)))  # Assume max 1s latency
        score += normalized_latency * 0.25
        
        # Factor 3: Fees (lower is better) - weight 20%
        fee_rate = metrics.fees.get('taker', 0.001)  # Use taker fee as default
        normalized_fees = max(0, min(1, 1 - fee_rate * 100))  # Invert so lower fees = higher score
        score += normalized_fees * 0.20
        
        # Factor 4: Available liquidity (higher is better) - weight 15%
        # Check if we have enough liquidity for this order
        order_value = quantity * (price or 1.0)  # Use price if available, otherwise assume $1
        if metrics.available_liquidity >= order_value:
            liquidity_score = min(1.0, order_value / metrics.available_liquidity)
        else:
            liquidity_score = 0.1  # Low score if not enough liquidity
        score += liquidity_score * 0.15
        
        # Factor 5: Min notional check (binary) - weight 5%
        if order_value >= metrics.min_notional:
            score += 0.05
        
        return score
    
    async def make_routing_decision(self, symbol: str, side: str, quantity: float, 
                                 order_type: str = 'MARKET', price: Optional[float] = None,
                                 strategy: str = 'best_fill') -> RoutingDecision:
        """Make routing decision based on current metrics and strategy"""
        if not self.is_initialized:
            await self.initialize()
        
        # Get metrics for all available brokers
        metrics = await self.get_route_metrics(symbol)
        
        if not metrics:
            return RoutingDecision(
                primary_broker='default',
                secondary_brokers=[],
                allocation_percentage={'default': 1.0},
                reason='No brokers available, using default',
                confidence=0.5
            )
        
        # Calculate scores for each broker
        broker_scores = {}
        for broker_name, metric in metrics.items():
            score = self.calculate_routing_score(broker_name, metric, symbol, side, quantity, price)
            broker_scores[broker_name] = score
        
        # Sort brokers by score
        sorted_brokers = sorted(broker_scores.items(), key=lambda x: x[1], reverse=True)
        
        if strategy == 'best_fill':
            # Use the broker with the highest score
            primary_broker = sorted_brokers[0][0]
            confidence = sorted_brokers[0][1] / sum(score for _, score in sorted_brokers)  # Normalize
            
            return RoutingDecision(
                primary_broker=primary_broker,
                secondary_brokers=[broker[0] for broker in sorted_brokers[1:3]],  # Top 3 minus primary
                allocation_percentage={primary_broker: 1.0},
                reason=f'Selected {primary_broker} with highest routing score ({sorted_brokers[0][1]:.3f})',
                confidence=confidence
            )
        
        elif strategy == 'diversify':
            # Split order across top brokers based on their scores
            total_score = sum(score for _, score in sorted_brokers)
            if total_score == 0:
                primary_broker = sorted_brokers[0][0]
                return RoutingDecision(
                    primary_broker=primary_broker,
                    secondary_brokers=[],
                    allocation_percentage={primary_broker: 1.0},
                    reason='All scores are zero, using top broker',
                    confidence=0.5
                )
            
            allocation = {}
            remaining = 1.0
            primary_set = False
            
            for i, (broker, score) in enumerate(sorted_brokers):
                if i == 0:
                    # Assign primary broker
                    primary_broker = broker
                    primary_set = True
                elif i < 3:  # Limit to top 3 for secondary
                    pass  # Just track in secondary list
                
                # Allocate based on normalized score
                normalized_score = score / total_score
                if i == 0:  # Primary gets largest portion
                    allocation[broker] = min(0.7, normalized_score * 2)  # Cap at 70%
                    remaining -= allocation[broker]
                elif i < 3 and remaining > 0:  # Distribute rest among top 3
                    remaining_portion = normalized_score * (remaining / sum(s / total_score for _, s in sorted_brokers[1:3]))
                    allocation[broker] = min(0.3, remaining_portion)  # Cap secondary at 30%
                    remaining -= allocation[broker]
            
            # Assign any remaining to primary
            if remaining > 0:
                allocation[primary_broker] = allocation.get(primary_broker, 0) + remaining
            
            return RoutingDecision(
                primary_broker=primary_broker,
                secondary_brokers=[broker for broker, _ in sorted_brokers[1:3]],
                allocation_percentage=allocation,
                reason=f'Diversified across {len(allocation)} brokers based on routing scores',
                confidence=sum(allocation.values()) / len(allocation) if allocation else 0.5
            )
        
        else:
            # Default to best fill
            primary_broker = sorted_brokers[0][0]
            confidence = sorted_brokers[0][1] / sum(score for _, score in sorted_brokers)
            
            return RoutingDecision(
                primary_broker=primary_broker,
                secondary_brokers=[broker[0] for broker in sorted_brokers[1:3]],
                allocation_percentage={primary_broker: 1.0},
                reason=f'Selected {primary_broker} with highest routing score ({sorted_brokers[0][1]:.3f})',
                confidence=confidence
            )
    
    async def execute_order_with_routing(self, symbol: str, side: str, quantity: float,
                                       order_type: str = 'MARKET', price: Optional[float] = None,
                                       routing_strategy: str = 'best_fill') -> Dict:
        """Execute an order using smart routing"""
        # Make routing decision
        routing_decision = await self.make_routing_decision(
            symbol, side, quantity, order_type, price, routing_strategy
        )
        
        logger.info(f"🎯 Routing decision: {routing_decision.reason}")
        
        results = {}
        
        # Execute on primary broker first
        primary_broker = self.broker_manager.get_adapter(routing_decision.primary_broker)
        if primary_broker:
            primary_qty = quantity * routing_decision.allocation_percentage[routing_decision.primary_broker]
            result = await primary_broker.place_order(
                symbol=symbol,
                side=side,
                quantity=primary_qty,
                order_type=order_type,
                price=price
            )
            results[routing_decision.primary_broker] = result
            logger.info(f"✅ Primary execution on {routing_decision.primary_broker}: {result}")
        
        # Execute on secondary brokers if using diversification
        for secondary_broker_name in routing_decision.secondary_brokers:
            if secondary_broker_name in routing_decision.allocation_percentage:
                secondary_broker = self.broker_manager.get_adapter(secondary_broker_name)
                if secondary_broker:
                    secondary_qty = quantity * routing_decision.allocation_percentage[secondary_broker_name]
                    result = await secondary_broker.place_order(
                        symbol=symbol,
                        side=side,
                        quantity=secondary_qty,
                        order_type=order_type,
                        price=price
                    )
                    results[secondary_broker_name] = result
                    logger.info(f"✅ Secondary execution on {secondary_broker_name}: {result}")
        
        # Record execution for analytics
        execution_record = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'order_type': order_type,
            'price': price,
            'routing_decision': routing_decision,
            'results': results
        }
        self.execution_history.append(execution_record)
        
        return {
            'routing_decision': routing_decision,
            'execution_results': results,
            'total_filled': quantity,  # Simplified
            'success': all('error' not in res for res in results.values())
        }
    
    def get_execution_statistics(self) -> Dict:
        """Get statistics about execution performance"""
        if not self.execution_history:
            return {'message': 'No execution history available'}
        
        total_executions = len(self.execution_history)
        successful_executions = sum(1 for exec in self.execution_history 
                                  if exec.get('results') and 
                                  all('error' not in res for res in exec['results'].values()))
        
        avg_latency = self.latency_monitor.get_all_latencies()
        
        return {
            'total_executions': total_executions,
            'successful_executions': successful_executions,
            'success_rate': successful_executions / total_executions if total_executions > 0 else 0,
            'average_latency_by_broker': avg_latency,
            'recent_executions': self.execution_history[-10:]  # Last 10 executions
        }
    
    async def close(self):
        """Clean up resources"""
        await self.latency_monitor.stop_monitoring()
        await self.broker_manager.disconnect_all()
        logger.info("🚦 Order Routing Engine closed")


# Global instance
routing_engine = None


def initialize_routing_engine():
    """Initialize the global routing engine"""
    global routing_engine
    routing_engine = OrderRoutingEngine()
    logger.info("🚦 Order Routing Engine created")
    return routing_engine