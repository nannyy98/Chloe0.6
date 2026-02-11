"""
Normalized Market Data Models for Chloe AI
Standardized data structures for market data across different exchanges
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd


@dataclass
class MarketData:
    """Normalized market data structure"""
    symbol: str
    timestamp: datetime
    exchange: str
    data_type: str  # 'trade', 'quote', 'book', 'ohlc'
    data: Dict


@dataclass
class TradeData:
    """Normalized trade data structure"""
    symbol: str
    exchange: str
    price: float
    quantity: float
    side: str  # 'buy' or 'sell'
    trade_id: str
    timestamp: datetime
    fee: Optional[float] = None
    fee_currency: Optional[str] = None


@dataclass
class QuoteData:
    """Normalized quote data structure"""
    symbol: str
    exchange: str
    bid_price: float
    bid_quantity: float
    ask_price: float
    ask_quantity: float
    timestamp: datetime


@dataclass
class OrderBookData:
    """Normalized order book data structure"""
    symbol: str
    exchange: str
    bids: List[tuple]  # List of (price, quantity) tuples
    asks: List[tuple]  # List of (price, quantity) tuples
    timestamp: datetime
    checksum: Optional[str] = None


@dataclass
class OHLCVData:
    """Normalized OHLCV data structure"""
    symbol: str
    exchange: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: datetime
    interval: str  # '1m', '5m', '1h', '1d', etc.


@dataclass
class MarketDepth:
    """Market depth information"""
    symbol: str
    exchange: str
    levels: int
    bids: List[Dict[str, float]]  # [{'price': float, 'quantity': float}, ...]
    asks: List[Dict[str, float]]  # [{'price': float, 'quantity': float}, ...]
    timestamp: datetime


class MarketDataAdapter:
    """Adapter to normalize market data from different exchanges"""
    
    @staticmethod
    def normalize_trade(exchange: str, raw_data: Dict) -> TradeData:
        """Normalize trade data from any exchange to standard format"""
        if exchange.lower() == 'binance':
            return TradeData(
                symbol=raw_data['s'],
                exchange=exchange,
                price=float(raw_data['p']),
                quantity=float(raw_data['q']),
                side='buy' if raw_data['m'] else 'sell',  # m is True for buyer
                trade_id=str(raw_data['t']),
                timestamp=datetime.fromtimestamp(raw_data['T'] / 1000.0)
            )
        elif exchange.lower() == 'kraken':
            # Kraken trade format: [price, volume, time, buy/sell, market/limit, misc]
            return TradeData(
                symbol=raw_data[3],  # Pair name
                exchange=exchange,
                price=float(raw_data[0]),
                quantity=float(raw_data[1]),
                side='buy' if raw_data[2] == 'b' else 'sell',
                trade_id=f"{raw_data[0]}_{raw_data[1]}_{raw_data[2]}",  # Kraken doesn't provide unique IDs
                timestamp=datetime.fromtimestamp(raw_data[2])
            )
        else:
            # Generic mapping - assumes common field names
            return TradeData(
                symbol=raw_data.get('symbol', raw_data.get('s', '')),
                exchange=exchange,
                price=raw_data.get('price', raw_data.get('p', 0.0)),
                quantity=raw_data.get('quantity', raw_data.get('q', 0.0)),
                side=raw_data.get('side', raw_data.get('S', 'buy')).lower(),
                trade_id=str(raw_data.get('id', raw_data.get('trade_id', ''))),
                timestamp=datetime.fromtimestamp(raw_data.get('timestamp', raw_data.get('T', 0)))
            )
    
    @staticmethod
    def normalize_quote(exchange: str, raw_data: Dict) -> QuoteData:
        """Normalize quote data from any exchange to standard format"""
        if exchange.lower() == 'binance':
            return QuoteData(
                symbol=raw_data['s'],
                exchange=exchange,
                bid_price=float(raw_data['b']),
                bid_quantity=float(raw_data['B']),
                ask_price=float(raw_data['a']),
                ask_quantity=float(raw_data['A']),
                timestamp=datetime.fromtimestamp(raw_data['E'] / 1000.0)
            )
        elif exchange.lower() == 'kraken':
            return QuoteData(
                symbol=raw_data['pair'],  # Assuming pair is in raw_data
                exchange=exchange,
                bid_price=float(raw_data['bid'][0]),  # [price, whole_lot_volume, lot_volume]
                bid_quantity=float(raw_data['bid'][2]),
                ask_price=float(raw_data['ask'][0]),
                ask_quantity=float(raw_data['ask'][2]),
                timestamp=datetime.fromtimestamp(raw_data['time'])
            )
        else:
            return QuoteData(
                symbol=raw_data.get('symbol', ''),
                exchange=exchange,
                bid_price=raw_data.get('bid', raw_data.get('bid_price', 0.0)),
                bid_quantity=raw_data.get('bid_qty', raw_data.get('bid_quantity', 0.0)),
                ask_price=raw_data.get('ask', raw_data.get('ask_price', 0.0)),
                ask_quantity=raw_data.get('ask_qty', raw_data.get('ask_quantity', 0.0)),
                timestamp=datetime.fromtimestamp(raw_data.get('timestamp', 0))
            )
    
    @staticmethod
    def normalize_ohlcv(exchange: str, raw_data: Dict) -> OHLCVData:
        """Normalize OHLCV data from any exchange to standard format"""
        if exchange.lower() == 'binance':
            return OHLCVData(
                symbol=raw_data[0],
                exchange=exchange,
                open=float(raw_data[1]),
                high=float(raw_data[2]),
                low=float(raw_data[3]),
                close=float(raw_data[4]),
                volume=float(raw_data[5]),
                timestamp=datetime.fromtimestamp(raw_data[6] / 1000.0),
                interval=raw_data[11]  # Interval in binance format
            )
        elif exchange.lower() == 'kraken':
            return OHLCVData(
                symbol=raw_data['pair'],
                exchange=exchange,
                open=float(raw_data[1]),
                high=float(raw_data[2]),
                low=float(raw_data[3]),
                close=float(raw_data[4]),
                volume=float(raw_data[6]),
                timestamp=datetime.fromtimestamp(raw_data[0]),
                interval=raw_data.get('interval', '1m')
            )
        else:
            return OHLCVData(
                symbol=raw_data.get('symbol', ''),
                exchange=exchange,
                open=raw_data.get('open', 0.0),
                high=raw_data.get('high', 0.0),
                low=raw_data.get('low', 0.0),
                close=raw_data.get('close', 0.0),
                volume=raw_data.get('volume', 0.0),
                timestamp=datetime.fromtimestamp(raw_data.get('timestamp', 0)),
                interval=raw_data.get('interval', '1m')
            )