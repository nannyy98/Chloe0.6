"""
Execution Engine - Professional Order Management
Handles order routing, execution, and broker integration
"""

import asyncio
import logging
from typing import Dict, List, Optional, Protocol
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import ccxt
import yfinance as yf

from execution.adapters.base_broker_adapter import BrokerAdapterManager
from execution.routing_engine import routing_engine, initialize_routing_engine
from execution.adapters.binance_adapter import BinanceBrokerAdapter

logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderStatus(Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

@dataclass
class Order:
    """Order object for execution"""
    symbol: str
    order_type: OrderType
    side: OrderSide
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    created_time: datetime = None
    filled_time: Optional[datetime] = None
    filled_price: Optional[float] = None
    filled_quantity: float = 0.0
    commission: float = 0.0
    exchange: Optional[str] = None
    
    def __post_init__(self):
        if self.created_time is None:
            self.created_time = datetime.now()

class OrderManager:
    """Main order management system with advanced routing capabilities"""
    
    def __init__(self):
        self.broker_manager = BrokerAdapterManager()
        self.active_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.commission_rate = 0.001  # 0.1% default commission
        self.routing_enabled = True
        
        # Initialize routing engine
        initialize_routing_engine()
        
    def add_broker_adapter(self, name: str, adapter, is_default: bool = False):
        """Add broker adapter to manager"""
        self.broker_manager.register_adapter(name, adapter, is_default)
        logger.info(f"✅ Added broker adapter: {name}")
        
    async def connect_all_brokers(self) -> Dict[str, bool]:
        """Connect to all registered brokers"""
        return await self.broker_manager.connect_all()
        
    async def place_order(self, symbol: str, side: str, quantity: float, 
                         order_type: str = 'MARKET', price: float = None,
                         broker: str = 'default', use_routing: bool = True) -> Optional[Order]:
        """Place order through broker with optional smart routing"""
        # Create order object
        order = Order(
            symbol=symbol,
            order_type=OrderType(order_type.upper()),
            side=OrderSide(side.upper()),
            quantity=quantity,
            price=price
        )
        
        # Use smart routing if enabled and requested
        if self.routing_enabled and use_routing:
            try:
                from execution.routing_engine import routing_engine
                if routing_engine:
                    result = await routing_engine.execute_order_with_routing(
                        symbol=symbol,
                        side=side,
                        quantity=quantity,
                        order_type=order_type,
                        price=price,
                        routing_strategy='best_fill'
                    )
                    
                    # Extract order ID from result
                    for broker_name, broker_result in result['execution_results'].items():
                        if 'orderId' in broker_result:
                            order.order_id = broker_result['orderId']
                            order.status = OrderStatus.SUBMITTED
                            order.exchange = broker_name.upper()
                            break
                    
                    if order.order_id:
                        self.active_orders[order.order_id] = order
                        self.order_history.append(order)
                        return order
            except Exception as e:
                logger.warning(f"⚠️ Smart routing failed, falling back to direct execution: {e}")
        
        # Fallback to direct broker execution
        broker_adapter = self.broker_manager.get_adapter(broker)
        if not broker_adapter:
            logger.error(f"❌ Broker adapter {broker} not found")
            return None
            
        # Place order using the adapter interface
        try:
            result = await broker_adapter.place_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                price=price
            )
            
            if result and not result.get('error'):
                order_id = result.get('orderId') or result.get('order_id') or str(hash(f"{symbol}_{datetime.now()}"))
                order.order_id = order_id
                order.status = OrderStatus.SUBMITTED
                order.exchange = broker.upper()
                
                self.active_orders[order.order_id] = order
                self.order_history.append(order)
                
                logger.info(f"✅ Order placed: {symbol} {side} {quantity} @ {price or 'MARKET'} on {broker}")
                return order
        except Exception as e:
            logger.error(f"❌ Order placement failed: {e}")
            order.status = OrderStatus.REJECTED
            
        return None
        
    async def get_market_price(self, symbol: str, broker: str = 'default') -> float:
        """Get market price from broker"""
        broker_adapter = self.broker_manager.get_adapter(broker)
        if broker_adapter:
            try:
                return await broker_adapter.get_market_price(symbol)
            except:
                # Fallback to yfinance for paper trading
                return await self._get_yfinance_price(symbol)
        return 0.0
    
    async def _get_yfinance_price(self, symbol: str) -> float:
        """Fallback to yfinance for price data"""
        try:
            # Convert symbol format (BTC/USDT -> BTC-USD)
            yf_symbol = symbol.replace('/', '-').replace('USDT', 'USD')
            ticker = yf.Ticker(yf_symbol)
            hist = ticker.history(period="1d", interval="1m")
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
            return 0.0
        except Exception as e:
            logger.error(f"❌ YFinance fallback failed for {symbol}: {e}")
            return 0.0
        
    async def update_order_status(self):
        """Update status of all active orders"""
        for order_id, order in list(self.active_orders.items()):
            broker_adapter = self.broker_manager.get_adapter(order.exchange.lower() if order.exchange else 'default')
            if broker_adapter:
                try:
                    status_info = await broker_adapter.get_order_status(order_id)
                    if status_info and 'standard_status' in status_info:
                        status_map = {
                            'SUBMITTED': OrderStatus.SUBMITTED,
                            'PARTIALLY_FILLED': OrderStatus.PARTIALLY_FILLED,
                            'FILLED': OrderStatus.FILLED,
                            'CANCELLED': OrderStatus.CANCELLED,
                            'REJECTED': OrderStatus.REJECTED
                        }
                        new_status = status_map.get(status_info['standard_status'], OrderStatus.PENDING)
                        order.status = new_status
                        
                        if new_status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                            del self.active_orders[order_id]
                    elif status_info and 'status' in status_info:
                        # Fallback to raw status
                        status_map = {
                            'NEW': OrderStatus.SUBMITTED,
                            'PARTIALLY_FILLED': OrderStatus.PARTIALLY_FILLED,
                            'FILLED': OrderStatus.FILLED,
                            'CANCELED': OrderStatus.CANCELLED,
                            'REJECTED': OrderStatus.REJECTED,
                            'EXPIRED': OrderStatus.REJECTED
                        }
                        raw_status = status_info['status']
                        new_status = status_map.get(raw_status, OrderStatus.PENDING)
                        order.status = new_status
                        
                        if new_status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                            del self.active_orders[order_id]
                except Exception as e:
                    logger.error(f"❌ Error getting order status for {order_id}: {e}")
                        
    def get_active_orders(self) -> Dict[str, Order]:
        """Get all active orders"""
        return self.active_orders.copy()
        
    def get_order_history(self, limit: int = 100) -> List[Order]:
        """Get recent order history"""
        return self.order_history[-limit:] if self.order_history else []

# Global order manager instance
order_manager = OrderManager()

def initialize_order_manager() -> OrderManager:
    """Initialize global order manager with default broker adapters"""
    global order_manager
    
    # Add paper trading broker adapter (using testnet)
    try:
        paper_broker = BinanceBrokerAdapter(api_key='', secret='', testnet=True)  # Empty keys for testnet
        order_manager.add_broker_adapter('paper', paper_broker, is_default=True)
        order_manager.add_broker_adapter('default', paper_broker)
        logger.info("📋 Order manager initialized with Binance testnet adapter")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Binance adapter: {e}")
        # Fallback to basic implementation if needed
        from execution.adapters.base_broker_adapter import BaseBrokerAdapter
        # This would require implementing a basic fallback adapter
        
    return order_manager