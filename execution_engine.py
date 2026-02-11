"""
Professional Execution Engine for Chloe AI 0.4
Implements smart order routing, market impact modeling, and execution optimization
Based on institutional trading practices
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import asyncio
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class Order:
    """Trading order representation"""
    order_id: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    price: float
    order_type: str = 'LIMIT'  # 'MARKET', 'LIMIT', 'STOP'
    time_in_force: str = 'GTC'  # 'GTC', 'IOC', 'FOK'
    status: str = 'PENDING'
    created_at: datetime = None
    executed_at: Optional[datetime] = None
    executed_price: Optional[float] = None
    executed_quantity: float = 0.0
    fees: float = 0.0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class ExecutionReport:
    """Execution performance report"""
    order_id: str
    symbol: str
    planned_quantity: float
    executed_quantity: float
    average_price: float
    slippage: float
    fees: float
    execution_time: float  # in seconds
    market_impact: float
    status: str

class MarketImpactModel:
    """Models market impact of large orders"""
    
    def __init__(self):
        self.impact_coefficients = {
            'BTC/USDT': 0.0001,  # 0.01% impact per 1% of market volume
            'ETH/USDT': 0.0002,  # 0.02% impact per 1% of market volume
            'SOL/USDT': 0.0005,  # 0.05% impact per 1% of market volume
            'default': 0.0003
        }
        logger.info("📈 Market Impact Model initialized")

    def estimate_impact(self, symbol: str, order_size: float, 
                       market_volume_24h: float, price: float) -> float:
        """
        Estimate market impact of an order
        Returns impact as percentage of current price
        """
        try:
            # Calculate participation rate
            participation_rate = abs(order_size * price) / market_volume_24h
            
            # Get symbol-specific coefficient
            coeff = self.impact_coefficients.get(symbol, self.impact_coefficients['default'])
            
            # Square root market impact model
            impact_percentage = coeff * np.sqrt(participation_rate * 100)  # Scale to percentage
            
            return impact_percentage
            
        except Exception as e:
            logger.warning(f"Impact estimation failed for {symbol}: {e}")
            return 0.01  # Default 1% impact

class SlippageEstimator:
    """Estimates execution slippage"""
    
    def __init__(self):
        self.slippage_models = {
            'BTC/USDT': {'linear': 0.0001, 'quadratic': 0.00001},
            'ETH/USDT': {'linear': 0.0002, 'quadratic': 0.00002},
            'SOL/USDT': {'linear': 0.0005, 'quadratic': 0.00005},
            'default': {'linear': 0.0003, 'quadratic': 0.00003}
        }
        logger.info("📉 Slippage Estimator initialized")

    def estimate_slippage(self, symbol: str, order_size: float, 
                         volatility: float, order_book_depth: float = 1000000) -> float:
        """
        Estimate slippage for an order
        Returns slippage as percentage of order value
        """
        try:
            # Get symbol model
            model = self.slippage_models.get(symbol, self.slippage_models['default'])
            
            # Size-based slippage
            size_component = model['linear'] * abs(order_size) + model['quadratic'] * (order_size ** 2)
            
            # Volatility component
            vol_component = volatility * 0.1  # 10% of current volatility
            
            # Order book depth adjustment
            depth_factor = min(1.0, 1000000 / order_book_depth)  # Normalize to 1M depth
            
            total_slippage = (size_component + vol_component) * depth_factor
            
            return min(total_slippage, 0.05)  # Cap at 5% slippage
            
        except Exception as e:
            logger.warning(f"Slippage estimation failed for {symbol}: {e}")
            return 0.02  # Default 2% slippage

class OrderRouter:
    """Smart order routing system"""
    
    def __init__(self):
        self.execution_venues = ['binance', 'bybit', 'kucoin', 'okx']
        self.venue_costs = {
            'binance': {'fee': 0.001, 'latency': 0.05},    # 0.1% fee, 50ms latency
            'bybit': {'fee': 0.001, 'latency': 0.08},      # 0.1% fee, 80ms latency
            'kucoin': {'fee': 0.001, 'latency': 0.12},     # 0.1% fee, 120ms latency
            'okx': {'fee': 0.0015, 'latency': 0.15}        # 0.15% fee, 150ms latency
        }
        self.best_venues = {}  # Cache for best venues per symbol
        logger.info("🔀 Order Router initialized")

    def select_best_venue(self, symbol: str, order_size: float, urgency: str = 'NORMAL') -> str:
        """
        Select optimal execution venue based on cost, latency, and liquidity
        """
        try:
            # For demo purposes, use simple selection logic
            # In production: would check real-time liquidity, fees, and market conditions
            
            if symbol in ['BTC/USDT', 'ETH/USDT']:
                # Major pairs - Binance typically best
                return 'binance'
            elif 'SOL' in symbol or 'AVAX' in symbol:
                # Altcoins - Bybit often better
                return 'bybit'
            else:
                # Others - Kucoin good balance
                return 'kucoin'
                
        except Exception as e:
            logger.warning(f"Venue selection failed for {symbol}: {e}")
            return 'binance'  # Default fallback

    def slice_large_order(self, order: Order, max_participation_rate: float = 0.02) -> List[Order]:
        """
        Slice large orders into smaller chunks to minimize market impact
        """
        try:
            # Calculate maximum chunk size (2% of market volume)
            max_chunk_size = order.quantity * max_participation_rate
            
            if order.quantity <= max_chunk_size:
                return [order]  # No slicing needed
            
            # Calculate number of chunks
            num_chunks = int(np.ceil(order.quantity / max_chunk_size))
            chunk_size = order.quantity / num_chunks
            
            chunks = []
            for i in range(num_chunks):
                chunk_order = Order(
                    order_id=f"{order.order_id}_chunk_{i+1}",
                    symbol=order.symbol,
                    side=order.side,
                    quantity=chunk_size,
                    price=order.price,
                    order_type=order.order_type,
                    time_in_force=order.time_in_force
                )
                chunks.append(chunk_order)
            
            logger.info(f"Sliced order {order.order_id} into {num_chunks} chunks of {chunk_size:.4f}")
            return chunks
            
        except Exception as e:
            logger.error(f"Order slicing failed: {e}")
            return [order]  # Return original order if slicing fails

class ExecutionEngine:
    """Main execution engine coordinating all execution components"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.pending_orders: Dict[str, Order] = {}
        self.completed_orders: Dict[str, Order] = {}
        self.execution_reports: List[ExecutionReport] = []
        
        # Initialize components
        self.impact_model = MarketImpactModel()
        self.slippage_estimator = SlippageEstimator()
        self.order_router = OrderRouter()
        
        # Mock market data (would be real in production)
        self.market_data = {
            'BTC/USDT': {'price': 48500, 'volume_24h': 20000000000, 'volatility': 0.03},
            'ETH/USDT': {'price': 3650, 'volume_24h': 10000000000, 'volatility': 0.04},
            'SOL/USDT': {'price': 47.5, 'volume_24h': 2000000000, 'volatility': 0.06}
        }
        
        logger.info("⚡ Execution Engine initialized")
        logger.info(f"   Initial Capital: ${self.initial_capital:,.2f}")

    async def execute_order(self, order: Order, execution_strategy: str = 'SMART') -> ExecutionReport:
        """
        Execute trading order with optimal strategy
        """
        try:
            logger.info(f"⚡ Executing order: {order.symbol} {order.side} {order.quantity}")
            
            # Add to pending orders
            self.pending_orders[order.order_id] = order
            
            # Analyze market conditions
            market_conditions = self._analyze_market_conditions(order.symbol)
            
            # Estimate execution costs
            impact_pct = self.impact_model.estimate_impact(
                order.symbol, order.quantity, 
                market_conditions['volume_24h'], market_conditions['price']
            )
            
            slippage_pct = self.slippage_estimator.estimate_slippage(
                order.symbol, order.quantity,
                market_conditions['volatility']
            )
            
            # Select execution strategy
            if execution_strategy == 'SMART':
                execution_plan = self._smart_execution_plan(order, market_conditions, impact_pct)
            elif execution_strategy == 'AGGRESSIVE':
                execution_plan = self._aggressive_execution_plan(order)
            else:  # PASSIVE
                execution_plan = self._passive_execution_plan(order)
            
            # Route order to optimal venue
            venue = self.order_router.select_best_venue(order.symbol, order.quantity)
            logger.info(f"   Selected venue: {venue}")
            
            # Simulate execution (would connect to real exchange in production)
            execution_result = await self._simulate_execution(order, execution_plan, venue)
            
            # Generate execution report
            report = self._generate_execution_report(order, execution_result, impact_pct, slippage_pct)
            
            # Update order status
            order.status = execution_result['status']
            order.executed_at = datetime.now()
            order.executed_price = execution_result['executed_price']
            order.executed_quantity = execution_result['executed_quantity']
            order.fees = execution_result['fees']
            
            # Move to completed orders
            self.completed_orders[order.order_id] = order
            del self.pending_orders[order.order_id]
            
            # Update capital
            self._update_capital(order, execution_result)
            
            self.execution_reports.append(report)
            
            logger.info(f"✅ Order executed: {order.symbol} filled {order.executed_quantity:.4f} at ${order.executed_price:.2f}")
            return report
            
        except Exception as e:
            logger.error(f"❌ Order execution failed: {e}")
            order.status = 'FAILED'
            raise

    def _analyze_market_conditions(self, symbol: str) -> Dict:
        """Analyze current market conditions for execution"""
        return self.market_data.get(symbol, {
            'price': 10000,
            'volume_24h': 1000000000,
            'volatility': 0.03
        })

    def _smart_execution_plan(self, order: Order, market_conditions: Dict, 
                            impact_pct: float) -> Dict:
        """Generate smart execution plan based on market impact"""
        plan = {
            'strategy': 'SMART',
            'should_slice': impact_pct > 0.01,  # Slice if impact > 1%
            'urgency': 'MODERATE',
            'timing': 'OPPORTUNISTIC'  # Execute when conditions improve
        }
        
        if plan['should_slice']:
            # Slice large orders
            chunks = self.order_router.slice_large_order(order)
            plan['chunks'] = chunks
            plan['interval_seconds'] = 30  # 30-second intervals between chunks
            
        return plan

    def _aggressive_execution_plan(self, order: Order) -> Dict:
        """Aggressive execution plan - immediate market order"""
        return {
            'strategy': 'AGGRESSIVE',
            'should_slice': False,
            'urgency': 'HIGH',
            'timing': 'IMMEDIATE'
        }

    def _passive_execution_plan(self, order: Order) -> Dict:
        """Passive execution plan - patient limit order placement"""
        return {
            'strategy': 'PASSIVE',
            'should_slice': False,
            'urgency': 'LOW',
            'timing': 'PATIENT',
            'price_adjustment': -0.001 if order.side == 'BUY' else 0.001  # 0.1% better price
        }

    async def _simulate_execution(self, order: Order, execution_plan: Dict, venue: str) -> Dict:
        """Simulate order execution (would be real exchange connection in production)"""
        # Simulate execution delay
        await asyncio.sleep(0.1)  # 100ms simulation delay
        
        market_price = self.market_data[order.symbol]['price']
        
        if execution_plan['strategy'] == 'AGGRESSIVE':
            # Market order - immediate execution at market price ± small spread
            executed_price = market_price * (1.001 if order.side == 'BUY' else 0.999)
            executed_quantity = order.quantity
            execution_time = 0.1  # 100ms
        elif execution_plan['strategy'] == 'PASSIVE':
            # Limit order - may not fill completely
            limit_price = order.price * (1 + execution_plan['price_adjustment'])
            if (order.side == 'BUY' and limit_price >= market_price) or \
               (order.side == 'SELL' and limit_price <= market_price):
                executed_price = limit_price
                executed_quantity = order.quantity
                execution_time = 2.0  # May take time to fill
            else:
                executed_price = 0
                executed_quantity = 0
                execution_time = 5.0  # Timeout
        else:  # SMART
            # Simulate partial fills
            executed_price = market_price * (1.0005 if order.side == 'BUY' else 0.9995)
            executed_quantity = order.quantity * 0.95  # 95% fill rate
            execution_time = 1.0
            
        # Calculate fees
        fee_rate = self.order_router.venue_costs[venue]['fee']
        fees = executed_price * executed_quantity * fee_rate
        
        status = 'FILLED' if executed_quantity > 0 else 'EXPIRED'
        
        return {
            'executed_price': executed_price,
            'executed_quantity': executed_quantity,
            'execution_time': execution_time,
            'fees': fees,
            'status': status
        }

    def _generate_execution_report(self, order: Order, execution_result: Dict,
                                 impact_pct: float, slippage_pct: float) -> ExecutionReport:
        """Generate detailed execution performance report"""
        
        planned_value = order.quantity * order.price
        executed_value = execution_result['executed_quantity'] * execution_result['executed_price']
        
        # Calculate slippage
        if execution_result['executed_quantity'] > 0:
            slippage = abs((execution_result['executed_price'] - order.price) / order.price)
        else:
            slippage = 0.0
            
        # Calculate market impact
        market_impact = impact_pct
        
        return ExecutionReport(
            order_id=order.order_id,
            symbol=order.symbol,
            planned_quantity=order.quantity,
            executed_quantity=execution_result['executed_quantity'],
            average_price=execution_result['executed_price'] if execution_result['executed_quantity'] > 0 else 0,
            slippage=slippage,
            fees=execution_result['fees'],
            execution_time=execution_result['execution_time'],
            market_impact=market_impact,
            status=execution_result['status']
        )

    def _update_capital(self, order: Order, execution_result: Dict):
        """Update capital based on execution results"""
        if execution_result['status'] == 'FILLED':
            cost = execution_result['executed_price'] * execution_result['executed_quantity']
            fees = execution_result['fees']
            
            if order.side == 'BUY':
                self.current_capital -= (cost + fees)
            else:  # SELL
                self.current_capital += (cost - fees)

    def get_performance_summary(self) -> Dict:
        """Get execution performance summary"""
        if not self.execution_reports:
            return {'message': 'No executions yet'}
            
        total_orders = len(self.execution_reports)
        filled_orders = len([r for r in self.execution_reports if r.status == 'FILLED'])
        fill_rate = filled_orders / total_orders if total_orders > 0 else 0
        
        avg_slippage = np.mean([r.slippage for r in self.execution_reports if r.status == 'FILLED'])
        avg_impact = np.mean([r.market_impact for r in self.execution_reports if r.status == 'FILLED'])
        total_fees = sum([r.fees for r in self.execution_reports])
        
        return {
            'total_orders': total_orders,
            'filled_orders': filled_orders,
            'fill_rate': fill_rate,
            'average_slippage': avg_slippage,
            'average_market_impact': avg_impact,
            'total_fees_paid': total_fees,
            'current_capital': self.current_capital,
            'capital_utilization': (self.initial_capital - self.current_capital) / self.initial_capital
        }

# Global execution engine instance
execution_engine = None

def get_execution_engine(initial_capital: float = 100000.0) -> ExecutionEngine:
    """Get singleton execution engine instance"""
    global execution_engine
    if execution_engine is None:
        execution_engine = ExecutionEngine(initial_capital)
    return execution_engine

def main():
    """Example usage"""
    print("⚡ Execution Engine ready for professional order execution")

if __name__ == "__main__":
    main()