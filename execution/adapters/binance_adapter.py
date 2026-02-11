"""
Binance Broker Adapter
Implements execution functionality for Binance exchange
"""
import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime
import aiohttp
import hmac
import hashlib
from urllib.parse import urlencode

from execution.adapters.base_broker_adapter import BaseBrokerAdapter

logger = logging.getLogger(__name__)


class BinanceBrokerAdapter(BaseBrokerAdapter):
    """Binance broker execution adapter"""
    
    def __init__(self, api_key: str, secret: str, testnet: bool = True):
        super().__init__("binance", api_key, secret)
        self.testnet = testnet
        self.session = None
        self.is_connected = False
        self.account_info = {}
        
        # Set URLs based on testnet/prod
        if testnet:
            self.rest_url = "https://testnet.binance.vision/api/v3"
            logger.info("🏦 Binance testnet adapter initialized")
        else:
            self.rest_url = "https://api.binance.com/api/v3"
            logger.info("🏦 Binance production adapter initialized")
    
    async def connect(self) -> bool:
        """Establish connection to Binance"""
        try:
            self.session = aiohttp.ClientSession()
            # Test connection by getting account info
            headers = {'X-MBX-APIKEY': self.api_key}
            url = f"{self.rest_url}/account"
            
            # Add timestamp for API call
            timestamp = int(datetime.now().timestamp() * 1000)
            params = {'timestamp': timestamp}
            query_string = '&'.join([f"{key}={value}" for key, value in params.items()])
            signature = hmac.new(
                self.secret.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            params['signature'] = signature
            
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    self.account_info = await response.json()
                    self.is_connected = True
                    logger.info("✅ Connected to Binance Broker API")
                    return True
                else:
                    error_data = await response.json()
                    logger.error(f"❌ Failed to authenticate with Binance: {error_data}")
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
        elif order_type.upper() == 'MARKET':
            # For market orders, we might need to handle quoteOrderQty differently
            pass
        
        # Add timestamp for authentication
        timestamp = int(datetime.now().timestamp() * 1000)
        params['timestamp'] = timestamp
        
        # Create signature
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
            async with self.session.post(url, headers=headers, params=params) as response:
                if response.status == 200:
                    order_data = await response.json()
                    logger.info(f"✅ Order placed: {order_data.get('symbol')} {order_data.get('side')} {order_data.get('origQty')} @ {order_data.get('price', 'MARKET')}")
                    return order_data
                else:
                    error_data = await response.json()
                    logger.error(f"❌ Failed to place order: {error_data}")
                    return {'error': error_data, 'success': False}
        except Exception as e:
            logger.error(f"❌ Error placing order: {e}")
            return {'error': str(e), 'success': False}
    
    async def cancel_order(self, order_id: str, symbol: str = None) -> bool:
        """Cancel an order on Binance"""
        if not self.is_connected:
            raise Exception("Not connected to Binance")
        
        params = {'orderId': order_id}
        if symbol:
            params['symbol'] = symbol.upper()
        
        # Add timestamp for authentication
        timestamp = int(datetime.now().timestamp() * 1000)
        params['timestamp'] = timestamp
        
        # Create signature
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
                success = response.status == 200
                if success:
                    logger.info(f"✅ Order cancelled: {order_id}")
                else:
                    error_data = await response.json()
                    logger.error(f"❌ Failed to cancel order {order_id}: {error_data}")
                return success
        except Exception as e:
            logger.error(f"❌ Error canceling order {order_id}: {e}")
            return False
    
    async def get_order_status(self, order_id: str) -> Dict:
        """Get status of an order on Binance"""
        if not self.is_connected:
            raise Exception("Not connected to Binance")
        
        params = {'orderId': order_id}
        
        # Add timestamp for authentication
        timestamp = int(datetime.now().timestamp() * 1000)
        params['timestamp'] = timestamp
        
        # Create signature
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
                    order_data = await response.json()
                    # Map Binance status to standard status
                    status_mapping = {
                        'NEW': 'SUBMITTED',
                        'PARTIALLY_FILLED': 'PARTIALLY_FILLED',
                        'FILLED': 'FILLED',
                        'CANCELED': 'CANCELLED',
                        'PENDING_CANCEL': 'PENDING_CANCEL',
                        'REJECTED': 'REJECTED',
                        'EXPIRED': 'EXPIRED'
                    }
                    
                    mapped_status = status_mapping.get(order_data.get('status', ''), order_data.get('status'))
                    order_data['standard_status'] = mapped_status
                    return order_data
                else:
                    error_data = await response.json()
                    return {'error': error_data, 'status': 'UNKNOWN'}
        except Exception as e:
            logger.error(f"❌ Error getting order status: {e}")
            return {'error': str(e), 'status': 'UNKNOWN'}
    
    async def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """Get all open orders on Binance"""
        if not self.is_connected:
            raise Exception("Not connected to Binance")
        
        params = {}
        if symbol:
            params['symbol'] = symbol.upper()
        
        # Add timestamp for authentication
        timestamp = int(datetime.now().timestamp() * 1000)
        params['timestamp'] = timestamp
        
        # Create signature
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
                    orders = await response.json()
                    # Add standard status mapping to each order
                    for order in orders:
                        status_mapping = {
                            'NEW': 'SUBMITTED',
                            'PARTIALLY_FILLED': 'PARTIALLY_FILLED',
                            'FILLED': 'FILLED',
                            'CANCELED': 'CANCELLED',
                            'PENDING_CANCEL': 'PENDING_CANCEL',
                            'REJECTED': 'REJECTED',
                            'EXPIRED': 'EXPIRED'
                        }
                        order['standard_status'] = status_mapping.get(order.get('status', ''), order.get('status'))
                    return orders
                else:
                    error_data = await response.json()
                    return [{'error': error_data}]
        except Exception as e:
            logger.error(f"❌ Error getting open orders: {e}")
            return [{'error': str(e)}]
    
    async def get_account_balance(self) -> Dict:
        """Get account balance information from Binance"""
        if not self.is_connected:
            raise Exception("Not connected to Binance")
        
        params = {}
        
        # Add timestamp for authentication
        timestamp = int(datetime.now().timestamp() * 1000)
        params['timestamp'] = timestamp
        
        # Create signature
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
                    account_data = await response.json()
                    # Format as simple balance dict
                    balance_dict = {}
                    for asset in account_data.get('balances', []):
                        if float(asset['free']) > 0 or float(asset['locked']) > 0:
                            total = float(asset['free']) + float(asset['locked'])
                            balance_dict[asset['asset']] = {
                                'free': float(asset['free']),
                                'locked': float(asset['locked']),
                                'total': total
                            }
                    return balance_dict
                else:
                    error_data = await response.json()
                    return {'error': error_data}
        except Exception as e:
            logger.error(f"❌ Error getting account balance: {e}")
            return {'error': str(e)}
    
    async def get_positions(self) -> Dict:
        """Get current positions from Binance (for spot trading)"""
        # For spot trading, positions are just non-zero balances
        balance = await self.get_account_balance()
        positions = {}
        
        for asset, details in balance.items():
            if isinstance(details, dict) and details.get('total', 0) > 0:
                positions[asset] = {
                    'symbol': asset,
                    'quantity': details['total'],
                    'value': None  # Would need price to calculate value
                }
        
        return positions
    
    async def get_trade_history(self, symbol: str = None, limit: int = 100) -> List[Dict]:
        """Get trade history from Binance"""
        if not self.is_connected:
            raise Exception("Not connected to Binance")
        
        params = {'limit': min(limit, 1000)}  # Binance max is 1000
        if symbol:
            params['symbol'] = symbol.upper()
        
        # Add timestamp for authentication
        timestamp = int(datetime.now().timestamp() * 1000)
        params['timestamp'] = timestamp
        
        # Create signature
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
                    trades = await response.json()
                    # Format trades to standard format
                    formatted_trades = []
                    for trade in trades:
                        formatted_trade = {
                            'id': trade['id'],
                            'symbol': trade['symbol'],
                            'orderId': trade['orderId'],
                            'side': trade['isBuyer'] and 'BUY' or 'SELL',
                            'price': float(trade['price']),
                            'quantity': float(trade['qty']),
                            'fee': float(trade['commission']),
                            'feeAsset': trade['commissionAsset'],
                            'timestamp': datetime.fromtimestamp(trade['time'] / 1000.0),
                            'isMaker': trade['isMaker']
                        }
                        formatted_trades.append(formatted_trade)
                    
                    return formatted_trades
                else:
                    error_data = await response.json()
                    return [{'error': error_data}]
        except Exception as e:
            logger.error(f"❌ Error getting trade history: {e}")
            return [{'error': str(e)}]
    
    async def get_market_price(self, symbol: str) -> float:
        """Get current market price for a symbol"""
        url = f"{self.rest_url}/ticker/price?symbol={symbol.upper()}"
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return float(data['price'])
                else:
                    logger.error(f"❌ Failed to get market price for {symbol}: {response.status}")
                    return 0.0
        except Exception as e:
            logger.error(f"❌ Error fetching market price for {symbol}: {e}")
            return 0.0