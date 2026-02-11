"""
Market Data Gateway Service
Main entry point for market data connectivity
"""
import asyncio
import logging
from typing import Dict, List, Callable, Optional
from datetime import datetime
import signal
import sys

from core.event_bus import event_bus, EventType, Event
from market_data.models import TradeData, QuoteData, OrderBookData, OHLCVData, MarketData
from market_data.connection_manager import MarketDataGateway as DataGateway
from market_data.retry_handler import create_connection_retry_handler
from market_data.adapters.binance_adapter import BinanceMarketDataAdapter

logger = logging.getLogger(__name__)


class MarketDataGatewayService:
    """Main market data gateway service that integrates with Chloe's event bus"""
    
    def __init__(self):
        self.data_gateway = DataGateway()
        self.retry_handler = create_connection_retry_handler()
        self.is_running = False
        self.subscribed_symbols = set()
        self.normalized_data_handlers = {}
        
        # Register data handlers for different data types
        self._register_data_handlers()
    
    def _register_data_handlers(self):
        """Register handlers for different market data types"""
        self.normalized_data_handlers = {
            'trade': self._handle_normalized_trade,
            'quote': self._handle_normalized_quote,
            'book': self._handle_normalized_book,
            'ohlcv': self._handle_normalized_ohlcv
        }
    
    def subscribe_to_symbol(self, symbol: str, exchange: str = 'binance'):
        """Subscribe to market data for a symbol"""
        if symbol not in self.subscribed_symbols:
            self.subscribed_symbols.add(symbol)
            logger.info(f"🔔 Subscribed to {symbol} on {exchange}")
    
    async def _handle_normalized_trade(self, trade_data: TradeData):
        """Handle normalized trade data and publish to event bus"""
        try:
            # Create market event for the event bus
            market_event = Event(
                event_type=EventType.MARKET,
                timestamp=trade_data.timestamp,
                data={
                    'type': 'trade',
                    'symbol': trade_data.symbol,
                    'price': trade_data.price,
                    'quantity': trade_data.quantity,
                    'side': trade_data.side,
                    'exchange': trade_data.exchange,
                    'trade_id': trade_data.trade_id
                }
            )
            
            # Publish to event bus
            await event_bus.publish(market_event)
            
            # Also publish as a specific trade event
            trade_event = Event(
                event_type=EventType.MARKET,
                timestamp=trade_data.timestamp,
                data={
                    'type': 'normalized_trade',
                    'symbol': trade_data.symbol,
                    'data': trade_data.__dict__
                }
            )
            
            await event_bus.publish(trade_event)
            
            logger.debug(f"📊 Published trade data for {trade_data.symbol}: {trade_data.price}")
            
        except Exception as e:
            logger.error(f"❌ Error handling trade data: {e}")
    
    async def _handle_normalized_quote(self, quote_data: QuoteData):
        """Handle normalized quote data and publish to event bus"""
        try:
            # Create market event for the event bus
            market_event = Event(
                event_type=EventType.MARKET,
                timestamp=quote_data.timestamp,
                data={
                    'type': 'quote',
                    'symbol': quote_data.symbol,
                    'bid_price': quote_data.bid_price,
                    'bid_quantity': quote_data.bid_quantity,
                    'ask_price': quote_data.ask_price,
                    'ask_quantity': quote_data.ask_quantity,
                    'exchange': quote_data.exchange
                }
            )
            
            # Publish to event bus
            await event_bus.publish(market_event)
            
            # Also publish as a specific quote event
            quote_event = Event(
                event_type=EventType.MARKET,
                timestamp=quote_data.timestamp,
                data={
                    'type': 'normalized_quote',
                    'symbol': quote_data.symbol,
                    'data': quote_data.__dict__
                }
            )
            
            await event_bus.publish(quote_event)
            
            logger.debug(f"📊 Published quote data for {quote_data.symbol}: B={quote_data.bid_price}, A={quote_data.ask_price}")
            
        except Exception as e:
            logger.error(f"❌ Error handling quote data: {e}")
    
    async def _handle_normalized_book(self, book_data: OrderBookData):
        """Handle normalized order book data and publish to event bus"""
        try:
            # Create market event for the event bus
            market_event = Event(
                event_type=EventType.MARKET,
                timestamp=book_data.timestamp,
                data={
                    'type': 'orderbook',
                    'symbol': book_data.symbol,
                    'bids': book_data.bids,
                    'asks': book_data.asks,
                    'exchange': book_data.exchange,
                    'checksum': book_data.checksum
                }
            )
            
            # Publish to event bus
            await event_bus.publish(market_event)
            
            # Also publish as a specific order book event
            book_event = Event(
                event_type=EventType.MARKET,
                timestamp=book_data.timestamp,
                data={
                    'type': 'normalized_orderbook',
                    'symbol': book_data.symbol,
                    'data': book_data.__dict__
                }
            )
            
            await event_bus.publish(book_event)
            
            logger.debug(f"📊 Published order book data for {book_data.symbol}: {len(book_data.bids)} bids, {len(book_data.asks)} asks")
            
        except Exception as e:
            logger.error(f"❌ Error handling order book data: {e}")
    
    async def _handle_normalized_ohlcv(self, ohlcv_data: OHLCVData):
        """Handle normalized OHLCV data and publish to event bus"""
        try:
            # Create market event for the event bus
            market_event = Event(
                event_type=EventType.MARKET,
                timestamp=ohlcv_data.timestamp,
                data={
                    'type': 'ohlcv',
                    'symbol': ohlcv_data.symbol,
                    'open': ohlcv_data.open,
                    'high': ohlcv_data.high,
                    'low': ohlcv_data.low,
                    'close': ohlcv_data.close,
                    'volume': ohlcv_data.volume,
                    'interval': ohlcv_data.interval,
                    'exchange': ohlcv_data.exchange
                }
            )
            
            # Publish to event bus
            await event_bus.publish(market_event)
            
            # Also publish as a specific OHLCV event
            ohlcv_event = Event(
                event_type=EventType.MARKET,
                timestamp=ohlcv_data.timestamp,
                data={
                    'type': 'normalized_ohlcv',
                    'symbol': ohlcv_data.symbol,
                    'data': ohlcv_data.__dict__
                }
            )
            
            await event_bus.publish(ohlcv_event)
            
            logger.debug(f"📊 Published OHLCV data for {ohlcv_data.symbol}: O={ohlcv_data.open}, C={ohlcv_data.close}, V={ohlcv_data.volume}")
            
        except Exception as e:
            logger.error(f"❌ Error handling OHLCV data: {e}")
    
    async def setup_binance_adapter(self, api_key: Optional[str] = None, secret: Optional[str] = None):
        """Setup Binance adapter with the gateway"""
        try:
            adapter = BinanceMarketDataAdapter(api_key=api_key, secret=secret)
            
            # Register the adapter
            self.data_gateway.register_adapter(adapter, 'binance')
            
            # Setup subscriptions for each symbol
            for symbol in self.subscribed_symbols:
                # Subscribe to trades
                await adapter.subscribe_to_trades(
                    symbol, 
                    lambda data: asyncio.create_task(self._handle_normalized_trade(data))
                )
                
                # Subscribe to quotes
                await adapter.subscribe_to_quotes(
                    symbol, 
                    lambda data: asyncio.create_task(self._handle_normalized_quote(data))
                )
                
                # Subscribe to order book
                await adapter.subscribe_to_order_book(
                    symbol, 
                    lambda data: asyncio.create_task(self._handle_normalized_book(data))
                )
            
            logger.info(f"✅ Binance adapter setup complete for {len(self.subscribed_symbols)} symbols")
            
        except Exception as e:
            logger.error(f"❌ Error setting up Binance adapter: {e}")
            raise
    
    async def start(self, symbols: List[str], api_key: Optional[str] = None, secret: Optional[str] = None):
        """Start the market data gateway service"""
        logger.info("🚀 Starting Market Data Gateway Service")
        
        # Subscribe to symbols
        for symbol in symbols:
            self.subscribe_to_symbol(symbol)
        
        # Setup Binance adapter
        await self.setup_binance_adapter(api_key, secret)
        
        # Start the data gateway
        self.is_running = True
        
        # Run the gateway
        try:
            await self.data_gateway.start()
        except KeyboardInterrupt:
            logger.info("🛑 Received shutdown signal")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the market data gateway service"""
        logger.info("🛑 Stopping Market Data Gateway Service")
        
        self.is_running = False
        
        # Stop the data gateway
        await self.data_gateway.stop()
        
        logger.info("✅ Market Data Gateway Service stopped")
    
    def get_status(self) -> Dict:
        """Get current service status"""
        return {
            'is_running': self.is_running,
            'subscribed_symbols': list(self.subscribed_symbols),
            'gateway_status': self.data_gateway.get_status(),
            'handlers_registered': list(self.normalized_data_handlers.keys())
        }


# Global instance for use in the institutional main
market_data_service = None


def initialize_market_data_service(symbols: List[str], api_key: Optional[str] = None, secret: Optional[str] = None):
    """Initialize the market data service with symbols and credentials"""
    global market_data_service
    market_data_service = MarketDataGatewayService()
    
    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        if market_data_service and market_data_service.is_running:
            asyncio.create_task(market_data_service.stop())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info(f"📡 Market Data Service initialized for symbols: {symbols}")
    return market_data_service


# Example usage function
async def run_market_data_service():
    """Example function to run the market data service"""
    # Example symbols to subscribe to
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    
    # Initialize service (in real usage, this would use actual API keys)
    service = initialize_market_data_service(symbols)
    
    # Start the service
    await service.start(symbols)


if __name__ == "__main__":
    # Run the market data service
    asyncio.run(run_market_data_service())