"""
Base Broker Adapter Interface
Defines the contract for all broker execution adapters
"""
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BaseBrokerAdapter(ABC):
    """Base interface for broker execution adapters"""
    
    def __init__(self, exchange_name: str, api_key: Optional[str] = None, secret: Optional[str] = None):
        self.exchange_name = exchange_name
        self.api_key = api_key
        self.secret = secret
        self.is_connected = False
        self.account_info = {}
        
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to broker/exchange"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from broker/exchange"""
        pass
    
    @abstractmethod
    async def place_order(self, symbol: str, side: str, quantity: float, order_type: str = 'LIMIT', 
                         price: Optional[float] = None, **kwargs) -> Dict:
        """Place an order on the exchange"""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str = None) -> bool:
        """Cancel an order on the exchange"""
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> Dict:
        """Get status of an order"""
        pass
    
    @abstractmethod
    async def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """Get all open orders"""
        pass
    
    @abstractmethod
    async def get_account_balance(self) -> Dict:
        """Get account balance information"""
        pass
    
    @abstractmethod
    async def get_positions(self) -> Dict:
        """Get current positions"""
        pass
    
    @abstractmethod
    async def get_trade_history(self, symbol: str = None, limit: int = 100) -> List[Dict]:
        """Get trade history"""
        pass
    
    @abstractmethod
    async def get_market_price(self, symbol: str) -> float:
        """Get current market price for a symbol"""
        pass


class BrokerAdapterManager:
    """Manages multiple broker adapters"""
    
    def __init__(self):
        self.adapters: Dict[str, BaseBrokerAdapter] = {}
        self.default_adapter = None
        
    def register_adapter(self, name: str, adapter: BaseBrokerAdapter, is_default: bool = False):
        """Register a broker adapter"""
        self.adapters[name] = adapter
        if is_default or self.default_adapter is None:
            self.default_adapter = name
        logger.info(f"✅ Registered broker adapter: {name}")
    
    def get_adapter(self, name: str = None) -> Optional[BaseBrokerAdapter]:
        """Get a specific adapter or the default one"""
        adapter_name = name or self.default_adapter
        return self.adapters.get(adapter_name)
    
    def get_all_adapters(self) -> Dict[str, BaseBrokerAdapter]:
        """Get all registered adapters"""
        return self.adapters.copy()
    
    async def connect_all(self) -> Dict[str, bool]:
        """Connect to all registered adapters"""
        results = {}
        for name, adapter in self.adapters.items():
            try:
                success = await adapter.connect()
                results[name] = success
                if success:
                    logger.info(f"✅ Connected to {name} broker")
                else:
                    logger.error(f"❌ Failed to connect to {name} broker")
            except Exception as e:
                logger.error(f"❌ Error connecting to {name} broker: {e}")
                results[name] = False
        return results
    
    async def disconnect_all(self) -> Dict[str, bool]:
        """Disconnect from all registered adapters"""
        results = {}
        for name, adapter in self.adapters.items():
            try:
                success = await adapter.disconnect()
                results[name] = success
                if success:
                    logger.info(f"✅ Disconnected from {name} broker")
                else:
                    logger.error(f"❌ Failed to disconnect from {name} broker")
            except Exception as e:
                logger.error(f"❌ Error disconnecting from {name} broker: {e}")
                results[name] = False
        return results