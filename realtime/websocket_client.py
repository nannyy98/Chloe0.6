"""
Real-time WebSocket Client for Chloe AI
Handles streaming market data from cryptocurrency exchanges
"""

import asyncio
import json
import logging
import websockets
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
import pandas as pd
import numpy as np
from dataclasses import dataclass
from collections import deque
import time

logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Real-time market data structure"""
    symbol: str
    price: float
    timestamp: datetime
    volume: float
    buy_volume: float
    sell_volume: float
    price_change: float
    price_change_percent: float

@dataclass
class OrderBook:
    """Order book data structure"""
    symbol: str
    bids: List[tuple]  # (price, quantity)
    asks: List[tuple]  # (price, quantity)
    timestamp: datetime

class WebSocketClient:
    """
    WebSocket client for real-time market data streaming
    """
    
    def __init__(self, exchange: str = 'binance'):
        self.exchange = exchange
        self.ws_url = self._get_websocket_url(exchange)
        self.connection = None
        self.is_connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 5  # seconds
        
        # Data storage
        self.price_buffer = {}
        self.orderbook_buffer = {}
        self.trade_buffer = {}
        
        # Callbacks
        self.price_callbacks = []
        self.orderbook_callbacks = []
        self.trade_callbacks = []
        
        # Buffer settings
        self.buffer_size = 1000
        self.data_quality_threshold = 0.95  # 95% data quality required
        
        logger.info(f"📡 WebSocket Client initialized for {exchange}")
    
    def _get_websocket_url(self, exchange: str) -> str:
        """Get WebSocket URL for the exchange"""
        urls = {
            'binance': 'wss://stream.binance.com:9443/ws',
            'binance_futures': 'wss://fstream.binance.com/ws',
            'coinbase': 'wss://ws-feed.exchange.coinbase.com',
            'kraken': 'wss://ws.kraken.com',
            'kucoin': 'wss://ws-api.kucoin.com/endpoint'
        }
        return urls.get(exchange, urls['binance'])
    
    def add_price_callback(self, callback: Callable[[MarketData], None]):
        """Add callback for price updates"""
        self.price_callbacks.append(callback)
    
    def add_orderbook_callback(self, callback: Callable[[OrderBook], None]):
        """Add callback for order book updates"""
        self.orderbook_callbacks.append(callback)
    
    def add_trade_callback(self, callback: Callable[[Dict], None]):
        """Add callback for trade updates"""
        self.trade_callbacks.append(callback)
    
    async def connect(self):
        """Establish WebSocket connection"""
        logger.info(f"🔗 Connecting to {self.exchange} WebSocket...")
        
        try:
            self.connection = await websockets.connect(self.ws_url)
            self.is_connected = True
            self.reconnect_attempts = 0
            logger.info("✅ WebSocket connection established")
            return True
        except Exception as e:
            logger.error(f"❌ WebSocket connection failed: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """Close WebSocket connection"""
        if self.connection:
            await self.connection.close()
            self.connection = None
            self.is_connected = False
            logger.info("🔌 WebSocket connection closed")
    
    async def reconnect(self):
        """Attempt to reconnect with exponential backoff"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error("❌ Max reconnection attempts reached")
            return False
        
        self.reconnect_attempts += 1
        delay = self.reconnect_delay * (2 ** (self.reconnect_attempts - 1))
        
        logger.info(f"🔄 Reconnecting in {delay} seconds (attempt {self.reconnect_attempts})")
        await asyncio.sleep(delay)
        
        return await self.connect()
    
    async def subscribe_to_tickers(self, symbols: List[str]):
        """Subscribe to ticker data for given symbols"""
        if not self.is_connected:
            await self.connect()
        
        try:
            if self.exchange == 'binance':
                # Binance format: <symbol>@ticker
                streams = [f"{symbol.lower()}@ticker" for symbol in symbols]
                subscribe_msg = {
                    "method": "SUBSCRIBE",
                    "params": streams,
                    "id": 1
                }
            elif self.exchange == 'coinbase':
                subscribe_msg = {
                    "type": "subscribe",
                    "channels": [{"name": "ticker", "product_ids": symbols}]
                }
            else:
                # Generic format
                subscribe_msg = {
                    "op": "subscribe",
                    "args": [f"ticker.{symbol}" for symbol in symbols]
                }
            
            await self.connection.send(json.dumps(subscribe_msg))
            logger.info(f"✅ Subscribed to ticker data for {len(symbols)} symbols")
            
        except Exception as e:
            logger.error(f"❌ Subscription failed: {e}")
    
    async def subscribe_to_orderbook(self, symbols: List[str], depth: int = 20):
        """Subscribe to order book data"""
        if not self.is_connected:
            await self.connect()
        
        try:
            if self.exchange == 'binance':
                streams = [f"{symbol.lower()}@depth{depth}@100ms" for symbol in symbols]
                subscribe_msg = {
                    "method": "SUBSCRIBE",
                    "params": streams,
                    "id": 2
                }
            elif self.exchange == 'coinbase':
                subscribe_msg = {
                    "type": "subscribe",
                    "channels": [{"name": "level2", "product_ids": symbols}]
                }
            else:
                subscribe_msg = {
                    "op": "subscribe",
                    "args": [f"orderbook.{depth}.{symbol}" for symbol in symbols]
                }
            
            await self.connection.send(json.dumps(subscribe_msg))
            logger.info(f"✅ Subscribed to order book for {len(symbols)} symbols")
            
        except Exception as e:
            logger.error(f"❌ Order book subscription failed: {e}")
    
    async def subscribe_to_trades(self, symbols: List[str]):
        """Subscribe to trade data"""
        if not self.is_connected:
            await self.connect()
        
        try:
            if self.exchange == 'binance':
                streams = [f"{symbol.lower()}@aggTrade" for symbol in symbols]
                subscribe_msg = {
                    "method": "SUBSCRIBE",
                    "params": streams,
                    "id": 3
                }
            elif self.exchange == 'coinbase':
                subscribe_msg = {
                    "type": "subscribe",
                    "channels": [{"name": "matches", "product_ids": symbols}]
                }
            else:
                subscribe_msg = {
                    "op": "subscribe",
                    "args": [f"trade.{symbol}" for symbol in symbols]
                }
            
            await self.connection.send(json.dumps(subscribe_msg))
            logger.info(f"✅ Subscribed to trade data for {len(symbols)} symbols")
            
        except Exception as e:
            logger.error(f"❌ Trade subscription failed: {e}")
    
    async def listen(self):
        """Main listening loop for incoming WebSocket messages"""
        if not self.is_connected:
            logger.error("❌ Not connected to WebSocket")
            return
        
        logger.info("👂 Starting WebSocket message listener...")
        
        try:
            async for message in self.connection:
                try:
                    data = json.loads(message)
                    await self._handle_message(data)
                except json.JSONDecodeError as e:
                    logger.warning(f"⚠️ Invalid JSON message: {e}")
                except Exception as e:
                    logger.error(f"❌ Error handling message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.warning("⚠️ WebSocket connection closed")
            self.is_connected = False
            # Attempt reconnection
            if await self.reconnect():
                await self.listen()
        except Exception as e:
            logger.error(f"❌ WebSocket listener error: {e}")
            self.is_connected = False
    
    async def _handle_message(self, data: Dict):
        """Handle incoming WebSocket message"""
        try:
            if self.exchange == 'binance':
                await self._handle_binance_message(data)
            elif self.exchange == 'coinbase':
                await self._handle_coinbase_message(data)
            else:
                await self._handle_generic_message(data)
        except Exception as e:
            logger.error(f"❌ Error processing message: {e}")
    
    async def _handle_binance_message(self, data: Dict):
        """Handle Binance WebSocket messages"""
        if 'e' in data:
            event_type = data['e']
            
            if event_type == '24hrTicker':
                # Ticker update
                market_data = self._parse_binance_ticker(data)
                self._update_price_buffer(market_data)
                for callback in self.price_callbacks:
                    try:
                        callback(market_data)
                    except Exception as e:
                        logger.error(f"❌ Error in price callback: {e}")
                        
            elif event_type == 'depthUpdate':
                # Order book update
                orderbook = self._parse_binance_orderbook(data)
                self._update_orderbook_buffer(orderbook)
                for callback in self.orderbook_callbacks:
                    try:
                        callback(orderbook)
                    except Exception as e:
                        logger.error(f"❌ Error in orderbook callback: {e}")
                        
            elif event_type == 'aggTrade':
                # Aggregate trade
                trade_data = self._parse_binance_trade(data)
                self._update_trade_buffer(trade_data)
                for callback in self.trade_callbacks:
                    try:
                        callback(trade_data)
                    except Exception as e:
                        logger.error(f"❌ Error in trade callback: {e}")
    
    async def _handle_coinbase_message(self, data: Dict):
        """Handle Coinbase WebSocket messages"""
        message_type = data.get('type')
        
        if message_type == 'ticker':
            market_data = self._parse_coinbase_ticker(data)
            self._update_price_buffer(market_data)
            for callback in self.price_callbacks:
                try:
                    callback(market_data)
                except Exception as e:
                    logger.error(f"❌ Error in price callback: {e}")
                    
        elif message_type == 'l2update':
            orderbook = self._parse_coinbase_orderbook(data)
            self._update_orderbook_buffer(orderbook)
            for callback in self.orderbook_callbacks:
                try:
                    callback(orderbook)
                except Exception as e:
                    logger.error(f"❌ Error in orderbook callback: {e}")
    
    async def _handle_generic_message(self, data: Dict):
        """Handle generic exchange messages"""
        # Generic handler - can be extended for other exchanges
        pass
    
    def _parse_binance_ticker(self, data: Dict) -> MarketData:
        """Parse Binance ticker data"""
        return MarketData(
            symbol=data['s'],
            price=float(data['c']),
            timestamp=datetime.fromtimestamp(data['E'] / 1000),
            volume=float(data['v']),
            buy_volume=float(data['V']),  # Taker buy volume
            sell_volume=float(data['v']) - float(data['V']),
            price_change=float(data['p']),
            price_change_percent=float(data['P'])
        )
    
    def _parse_coinbase_ticker(self, data: Dict) -> MarketData:
        """Parse Coinbase ticker data"""
        return MarketData(
            symbol=data['product_id'],
            price=float(data['price']),
            timestamp=datetime.fromisoformat(data['time'].rstrip('Z')),
            volume=float(data.get('volume', 0)),
            buy_volume=0,  # Not provided in Coinbase ticker
            sell_volume=0,  # Not provided in Coinbase ticker
            price_change=float(data.get('price_change', 0)),
            price_change_percent=float(data.get('price_change_percent', 0))
        )
    
    def _parse_binance_orderbook(self, data: Dict) -> OrderBook:
        """Parse Binance order book data"""
        return OrderBook(
            symbol=data['s'],
            bids=[(float(price), float(qty)) for price, qty in data['bids'][:20]],
            asks=[(float(price), float(qty)) for price, qty in data['asks'][:20]],
            timestamp=datetime.fromtimestamp(data['E'] / 1000)
        )
    
    def _parse_coinbase_orderbook(self, data: Dict) -> OrderBook:
        """Parse Coinbase order book data"""
        bids = []
        asks = []
        
        for change in data.get('changes', []):
            side, price, qty = change
            price, qty = float(price), float(qty)
            if side == 'buy':
                bids.append((price, qty))
            else:
                asks.append((price, qty))
        
        return OrderBook(
            symbol=data['product_id'],
            bids=bids[:20],
            asks=asks[:20],
            timestamp=datetime.now()
        )
    
    def _parse_binance_trade(self, data: Dict) -> Dict:
        """Parse Binance trade data"""
        return {
            'symbol': data['s'],
            'price': float(data['p']),
            'quantity': float(data['q']),
            'timestamp': datetime.fromtimestamp(data['T'] / 1000),
            'is_buyer_maker': data['m'],
            'trade_id': data['a']
        }
    
    def _update_price_buffer(self, market_data: MarketData):
        """Update price data buffer"""
        if market_data.symbol not in self.price_buffer:
            self.price_buffer[market_data.symbol] = deque(maxlen=self.buffer_size)
        
        self.price_buffer[market_data.symbol].append(market_data)
    
    def _update_orderbook_buffer(self, orderbook: OrderBook):
        """Update order book buffer"""
        self.orderbook_buffer[orderbook.symbol] = orderbook
    
    def _update_trade_buffer(self, trade_data: Dict):
        """Update trade buffer"""
        symbol = trade_data['symbol']
        if symbol not in self.trade_buffer:
            self.trade_buffer[symbol] = deque(maxlen=self.buffer_size)
        
        self.trade_buffer[symbol].append(trade_data)
    
    def get_latest_price(self, symbol: str) -> Optional[MarketData]:
        """Get latest price for a symbol"""
        if symbol in self.price_buffer and self.price_buffer[symbol]:
            return self.price_buffer[symbol][-1]
        return None
    
    def get_price_history(self, symbol: str, count: int = 100) -> List[MarketData]:
        """Get price history for a symbol"""
        if symbol in self.price_buffer:
            return list(self.price_buffer[symbol])[-count:]
        return []
    
    def get_orderbook(self, symbol: str) -> Optional[OrderBook]:
        """Get current order book for a symbol"""
        return self.orderbook_buffer.get(symbol)
    
    def get_data_quality(self, symbol: str) -> float:
        """Get data quality score for a symbol"""
        if symbol not in self.price_buffer:
            return 0.0
        
        buffer = self.price_buffer[symbol]
        if len(buffer) < 10:  # Need minimum data
            return 0.0
        
        # Check for data gaps and consistency
        timestamps = [data.timestamp for data in buffer]
        time_diffs = [(timestamps[i] - timestamps[i-1]).total_seconds() 
                     for i in range(1, len(timestamps))]
        
        # Expected interval (assuming 1 second for ticker data)
        expected_interval = 1.0
        consistency_score = sum(abs(diff - expected_interval) < 0.5 
                              for diff in time_diffs) / len(time_diffs)
        
        completeness_score = len(buffer) / self.buffer_size
        
        return (consistency_score + completeness_score) / 2

# Example usage
async def main():
    """Example usage of WebSocket client"""
    # Initialize client
    client = WebSocketClient('binance')
    
    # Add callbacks
    def price_callback(data: MarketData):
        print(f"📈 {data.symbol}: ${data.price:.2f} ({data.price_change_percent:+.2f}%)")
    
    def orderbook_callback(data: OrderBook):
        print(f"📋 {data.symbol} Order Book - Best Bid: {data.bids[0][0]} Best Ask: {data.asks[0][0]}")
    
    client.add_price_callback(price_callback)
    client.add_orderbook_callback(orderbook_callback)
    
    # Connect and subscribe
    if await client.connect():
        await client.subscribe_to_tickers(['BTCUSDT', 'ETHUSDT'])
        await client.subscribe_to_orderbook(['BTCUSDT'], depth=10)
        
        # Start listening
        await client.listen()

if __name__ == "__main__":
    asyncio.run(main())