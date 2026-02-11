"""
Base Market Data Adapter Interface
Defines the contract for all market data adapters
"""
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Callable, Optional
from datetime import datetime
import logging

from market_data.models import TradeData, QuoteData, OrderBookData, OHLCVData

logger = logging.getLogger(__name__)


class BaseMarketDataAdapter(ABC):
    """Base interface for market data adapters"""
    
    def __init__(self, exchange_name: str, api_key: Optional[str] = None, secret: Optional[str] = None):
        self.exchange_name = exchange_name
        self.api_key = api_key
        self.secret = secret
        self.is_connected = False
        self.connection_attempts = 0
        
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to exchange"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from exchange"""
        pass
    
    @abstractmethod
    async def subscribe_to_trades(self, symbol: str, callback: Callable[[TradeData], None]):
        """Subscribe to trade data for a symbol"""
        pass
    
    @abstractmethod
    async def subscribe_to_quotes(self, symbol: str, callback: Callable[[QuoteData], None]):
        """Subscribe to quote data for a symbol"""
        pass
    
    @abstractmethod
    async def subscribe_to_order_book(self, symbol: str, callback: Callable[[OrderBookData], None]):
        """Subscribe to order book data for a symbol"""
        pass
    
    @abstractmethod
    async def get_historical_data(self, symbol: str, start_time: datetime, end_time: datetime, 
                                 interval: str = '1m') -> List[OHLCVData]:
        """Get historical OHLCV data"""
        pass
    
    @abstractmethod
    async def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        pass
    
    @abstractmethod
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported trading pairs"""
        pass
    
    @abstractmethod
    def get_exchange_info(self) -> Dict:
        """Get exchange information"""
        pass


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