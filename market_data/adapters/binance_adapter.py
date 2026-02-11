"""
Binance Market Data Adapter
Implements real-time market data streaming from Binance
"""
import asyncio
import json
import logging
from typing import Dict, List, Callable, Optional
from datetime import datetime
import websockets
import aiohttp
from urllib.parse import urlencode

from market_data.adapters.base_adapter import BaseMarketDataAdapter
from market_data.models import TradeData, QuoteData, OrderBookData, OHLCVData, MarketDataAdapter

logger = logging.getLogger(__name__)


class BinanceMarketDataAdapter(BaseMarketDataAdapter):
    """Binance market data adapter implementation"""
    
    def __init__(self, api_key: Optional[str] = None, secret: Optional[str] = None):
        super().__init__("binance", api_key, secret)
        self.ws_url = "wss://stream.binance.com:9443/ws"
        self.rest_url = "https://api.binance.com/api/v3"
        self.websocket = None
        self.session = None
        self.subscribed_streams = set()
        self.stream_callbacks = {}
        
    async def connect(self) -> bool:
        """Establish connection to Binance"""
        try:
            self.session = aiohttp.ClientSession()
            logger.info("✅ Connected to Binance REST API")
            self.is_connected = True
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to Binance: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from Binance"""
        try:
            if self.websocket:
                await self.websocket.close()
            if self.session:
                await self.session.close()
            self.is_connected = False
            logger.info("✅ Disconnected from Binance")
            return True
        except Exception as e:
            logger.error(f"❌ Error disconnecting from Binance: {e}")
            return False
    
    async def _ensure_websocket_connection(self):
        """Ensure WebSocket connection is active"""
        if not self.websocket or self.websocket.closed:
            self.websocket = await websockets.connect(self.ws_url)
            # Re-subscribe to streams if we had any
            if self.subscribed_streams:
                stream_list = '/'.join(list(self.subscribed_streams))
                await self.websocket.send(json.dumps({
                    "method": "SUBSCRIBE",
                    "params": [stream_list],
                    "id": 1
                }))
    
    async def subscribe_to_trades(self, symbol: str, callback: Callable[[TradeData], None]):
        """Subscribe to trade data for a symbol"""
        stream_name = f"{symbol.lower()}@trade"
        self.stream_callbacks[stream_name] = callback
        
        if stream_name not in self.subscribed_streams:
            self.subscribed_streams.add(stream_name)
            
            if self.websocket and not self.websocket.closed:
                await self.websocket.send(json.dumps({
                    "method": "SUBSCRIBE",
                    "params": [stream_name],
                    "id": len(self.subscribed_streams)
                }))
            else:
                await self._ensure_websocket_connection()
                await self.websocket.send(json.dumps({
                    "method": "SUBSCRIBE",
                    "params": [stream_name],
                    "id": len(self.subscribed_streams)
                }))
        
        logger.info(f"✅ Subscribed to trades for {symbol}")
    
    async def subscribe_to_quotes(self, symbol: str, callback: Callable[[QuoteData], None]):
        """Subscribe to quote data for a symbol"""
        stream_name = f"{symbol.lower()}@ticker"
        self.stream_callbacks[stream_name] = callback
        
        if stream_name not in self.subscribed_streams:
            self.subscribed_streams.add(stream_name)
            
            if self.websocket and not self.websocket.closed:
                await self.websocket.send(json.dumps({
                    "method": "SUBSCRIBE",
                    "params": [stream_name],
                    "id": len(self.subscribed_streams)
                }))
            else:
                await self._ensure_websocket_connection()
                await self.websocket.send(json.dumps({
                    "method": "SUBSCRIBE",
                    "params": [stream_name],
                    "id": len(self.subscribed_streams)
                }))
        
        logger.info(f"✅ Subscribed to quotes for {symbol}")
    
    async def subscribe_to_order_book(self, symbol: str, callback: Callable[[OrderBookData], None]):
        """Subscribe to order book data for a symbol"""
        stream_name = f"{symbol.lower()}@depth"
        self.stream_callbacks[stream_name] = callback
        
        if stream_name not in self.subscribed_streams:
            self.subscribed_streams.add(stream_name)
            
            if self.websocket and not self.websocket.closed:
                await self.websocket.send(json.dumps({
                    "method": "SUBSCRIBE",
                    "params": [stream_name],
                    "id": len(self.subscribed_streams)
                }))
            else:
                await self._ensure_websocket_connection()
                await self.websocket.send(json.dumps({
                    "method": "SUBSCRIBE",
                    "params": [stream_name],
                    "id": len(self.subscribed_streams)
                }))
        
        logger.info(f"✅ Subscribed to order book for {symbol}")
    
    async def start_streaming(self):
        """Start the WebSocket streaming loop"""
        if not self.websocket:
            await self._ensure_websocket_connection()
        
        try:
            async for message in self.websocket:
                data = json.loads(message)
                
                # Handle different stream types
                stream = data.get('stream', '')
                
                if '@trade' in stream:
                    # Trade stream
                    normalized_data = MarketDataAdapter.normalize_trade(self.exchange_name, data['data'])
                    callback = self.stream_callbacks.get(stream)
                    if callback:
                        callback(normalized_data)
                
                elif '@ticker' in stream:
                    # Quote stream
                    normalized_data = MarketDataAdapter.normalize_quote(self.exchange_name, data['data'])
                    callback = self.stream_callbacks.get(stream)
                    if callback:
                        callback(normalized_data)
                
                elif '@depth' in stream:
                    # Order book stream
                    raw_data = data['data']
                    order_book = OrderBookData(
                        symbol=raw_data['s'],
                        exchange=self.exchange_name,
                        bids=[(float(bid[0]), float(bid[1])) for bid in raw_data['b']],
                        asks=[(float(ask[0]), float(ask[1])) for ask in raw_data['a']],
                        timestamp=datetime.fromtimestamp(raw_data['E'] / 1000.0)
                    )
                    callback = self.stream_callbacks.get(stream)
                    if callback:
                        callback(order_book)
        
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed, attempting to reconnect...")
            await self._ensure_websocket_connection()
            await self.start_streaming()
        except Exception as e:
            logger.error(f"Error in streaming loop: {e}")
    
    async def get_historical_data(self, symbol: str, start_time: datetime, end_time: datetime, 
                                 interval: str = '1m') -> List[OHLCVData]:
        """Get historical OHLCV data from Binance"""
        params = {
            'symbol': symbol.upper(),
            'interval': interval,
            'startTime': int(start_time.timestamp() * 1000),
            'endTime': int(end_time.timestamp() * 1000),
            'limit': 1000  # Max allowed by Binance
        }
        
        url = f"{self.rest_url}/klines?{urlencode(params)}"
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    ohlcv_list = []
                    for item in data:
                        ohlcv = OHLCVData(
                            symbol=symbol.upper(),
                            exchange=self.exchange_name,
                            open=float(item[1]),
                            high=float(item[2]),
                            low=float(item[3]),
                            close=float(item[4]),
                            volume=float(item[5]),
                            timestamp=datetime.fromtimestamp(item[6] / 1000.0),
                            interval=interval
                        )
                        ohlcv_list.append(ohlcv)
                    return ohlcv_list
                else:
                    logger.error(f"Failed to get historical data: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return []
    
    async def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol from Binance"""
        url = f"{self.rest_url}/ticker/price?symbol={symbol.upper()}"
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return float(data['price'])
                else:
                    logger.error(f"Failed to get current price: {response.status}")
                    return 0.0
        except Exception as e:
            logger.error(f"Error fetching current price: {e}")
            return 0.0
    
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported trading pairs from Binance"""
        # This would typically involve calling the exchange info endpoint
        # For now, returning a placeholder list
        return ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT']
    
    def get_exchange_info(self) -> Dict:
        """Get Binance exchange information"""
        return {
            'name': 'Binance',
            'rest_url': self.rest_url,
            'websocket_url': self.ws_url,
            'rate_limit': '1200 requests per minute',
            'supported_intervals': ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
        }


class BinanceBrokerAdapter:
    """Binance broker execution adapter"""
    
    def __init__(self, api_key: str, secret: str):
        self.api_key = api_key
        self.secret = secret
        self.exchange_name = "binance"
        self.rest_url = "https://api.binance.com/api/v3"
        self.session = None
        self.is_connected = False
        
    async def connect(self) -> bool:
        """Establish connection to Binance"""
        try:
            self.session = aiohttp.ClientSession()
            # Test connection by getting account info
            headers = {'X-MBX-APIKEY': self.api_key}
            async with self.session.get(f"{self.rest_url}/account", headers=headers) as response:
                if response.status == 200:
                    self.is_connected = True
                    logger.info("✅ Connected to Binance Broker API")
                    return True
                else:
                    logger.error(f"❌ Failed to authenticate with Binance: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"❌ Failed to connect to Binance: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from Binance"""
        try:
            if self.session:
                await self.session.close()
            self.is_connected = False
            logger.info("✅ Disconnected from Binance Broker API")
            return True
        except Exception as e:
            logger.error(f"❌ Error disconnecting from Binance: {e}")
            return False
    
    async def place_order(self, symbol: str, side: str, quantity: float, order_type: str = 'LIMIT', 
                         price: Optional[float] = None, **kwargs) -> Dict:
        """Place an order on Binance"""
        if not self.is_connected:
            raise Exception("Not connected to Binance")
        
        # Prepare order parameters
        params = {
            'symbol': symbol.upper(),
            'side': side.upper(),
            'type': order_type.upper(),
            'quantity': f'{quantity:.8f}',
        }
        
        if order_type.upper() in ['LIMIT', 'STOP_LOSS_LIMIT', 'TAKE_PROFIT_LIMIT']:
            if price is None:
                raise ValueError("Price is required for limit orders")
            params['price'] = f'{price:.8f}'
            params['timeInForce'] = kwargs.get('timeInForce', 'GTC')
        
        # Add signature and timestamp for authentication
        import hmac
        import hashlib
        timestamp = int(datetime.now().timestamp() * 1000)
        params['timestamp'] = timestamp
        
        # Create query string
        query_string = '&'.join([f"{key}={value}" for key, value in params.items()])
        
        # Create signature
        signature = hmac.new(
            self.secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        params['signature'] = signature
        
        headers = {'X-MBX-APIKEY': self.api_key}
        url = f"{self.rest_url}/order"
        
        try:
            async with self.session.post(url, headers=headers, params=params) as response:
                if response.status == 200:
                    order_data = await response.json()
                    return order_data
                else:
                    error_data = await response.json()
                    logger.error(f"Failed to place order: {error_data}")
                    return {'error': error_data}
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {'error': str(e)}
    
    async def cancel_order(self, order_id: str, symbol: str = None) -> bool:
        """Cancel an order on Binance"""
        if not self.is_connected:
            raise Exception("Not connected to Binance")
        
        params = {'orderId': order_id}
        if symbol:
            params['symbol'] = symbol.upper()
        
        import hmac
        import hashlib
        timestamp = int(datetime.now().timestamp() * 1000)
        params['timestamp'] = timestamp
        
        query_string = '&'.join([f"{key}={value}" for key, value in params.items()])
        signature = hmac.new(
            self.secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        params['signature'] = signature
        
        headers = {'X-MBX-APIKEY': self.api_key}
        url = f"{self.rest_url}/order"
        
        try:
            async with self.session.delete(url, headers=headers, params=params) as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Error canceling order: {e}")
            return False
    
    async def get_order_status(self, order_id: str) -> Dict:
        """Get status of an order on Binance"""
        if not self.is_connected:
            raise Exception("Not connected to Binance")
        
        params = {'orderId': order_id}
        
        import hmac
        import hashlib
        timestamp = int(datetime.now().timestamp() * 1000)
        params['timestamp'] = timestamp
        
        query_string = '&'.join([f"{key}={value}" for key, value in params.items()])
        signature = hmac.new(
            self.secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        params['signature'] = signature
        
        headers = {'X-MBX-APIKEY': self.api_key}
        url = f"{self.rest_url}/order"
        
        try:
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_data = await response.json()
                    return {'error': error_data}
        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            return {'error': str(e)}
    
    async def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """Get all open orders on Binance"""
        if not self.is_connected:
            raise Exception("Not connected to Binance")
        
        params = {}
        if symbol:
            params['symbol'] = symbol.upper()
        
        import hmac
        import hashlib
        timestamp = int(datetime.now().timestamp() * 1000)
        params['timestamp'] = timestamp
        
        query_string = '&'.join([f"{key}={value}" for key, value in params.items()])
        signature = hmac.new(
            self.secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        params['signature'] = signature
        
        headers = {'X-MBX-APIKEY': self.api_key}
        url = f"{self.rest_url}/openOrders"
        
        try:
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_data = await response.json()
                    return [{'error': error_data}]
        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            return [{'error': str(e)}]
    
    async def get_account_balance(self) -> Dict:
        """Get account balance information from Binance"""
        if not self.is_connected:
            raise Exception("Not connected to Binance")
        
        params = {}
        
        import hmac
        import hashlib
        timestamp = int(datetime.now().timestamp() * 1000)
        params['timestamp'] = timestamp
        
        query_string = '&'.join([f"{key}={value}" for key, value in params.items()])
        signature = hmac.new(
            self.secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        params['signature'] = signature
        
        headers = {'X-MBX-APIKEY': self.api_key}
        url = f"{self.rest_url}/account"
        
        try:
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_data = await response.json()
                    return {'error': error_data}
        except Exception as e:
            logger.error(f"Error getting account balance: {e}")
            return {'error': str(e)}
    
    async def get_positions(self) -> Dict:
        """Get current positions from Binance (for futures if applicable)"""
        # For spot trading, positions are just holdings
        # For futures, we would use a different endpoint
        balance = await self.get_account_balance()
        return balance
    
    async def get_trade_history(self, symbol: str = None, limit: int = 100) -> List[Dict]:
        """Get trade history from Binance"""
        if not self.is_connected:
            raise Exception("Not connected to Binance")
        
        params = {'limit': limit}
        if symbol:
            params['symbol'] = symbol.upper()
        
        import hmac
        import hashlib
        timestamp = int(datetime.now().timestamp() * 1000)
        params['timestamp'] = timestamp
        
        query_string = '&'.join([f"{key}={value}" for key, value in params.items()])
        signature = hmac.new(
            self.secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        params['signature'] = signature
        
        headers = {'X-MBX-APIKEY': self.api_key}
        url = f"{self.rest_url}/myTrades"
        
        try:
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_data = await response.json()
                    return [{'error': error_data}]
        except Exception as e:
            logger.error(f"Error getting trade history: {e}")
            return [{'error': str(e)}]