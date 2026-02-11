"""
Event Bus - Core Engine for Institutional Trading System
Event-driven architecture like hedge funds use
"""

import asyncio
import logging
from typing import Dict, List, Callable, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)

class EventType(Enum):
    MARKET = "MARKET"
    SIGNAL = "SIGNAL" 
    ORDER = "ORDER"
    FILL = "FILL"
    RISK = "RISK"
    PORTFOLIO = "PORTFOLIO"
    FORECAST = "FORECAST"

@dataclass
class Event:
    """Base event class"""
    event_type: EventType
    timestamp: datetime
    data: Dict[str, Any]

@dataclass
class MarketEvent(Event):
    """Market data update event"""
    symbol: str
    price: float
    volume: float
    exchange: str

@dataclass
class SignalEvent(Event):
    """Trading signal event"""
    symbol: str
    signal: str  # BUY, SELL, HOLD
    confidence: float
    strategy_id: str

@dataclass
class OrderEvent(Event):
    """Order execution request event"""
    symbol: str
    order_type: str  # MARKET, LIMIT
    side: str  # BUY, SELL
    quantity: float
    price: float = None

@dataclass
class FillEvent(Event):
    """Order fill confirmation event"""
    symbol: str
    side: str
    quantity: float
    price: float
    commission: float
    fill_time: datetime

@dataclass
class ForecastEvent(Event):
    """Market forecast event"""
    symbol: str
    horizon: int  # forecast horizon in days
    expected_return: float
    volatility: float
    confidence: float
    p10: float  # 10th percentile
    p50: float  # 50th percentile (median)
    p90: float  # 90th percentile

class EventBus:
    """Central event bus for routing events between components"""
    
    def __init__(self):
        self.handlers: Dict[EventType, List[Callable]] = {}
        self.queue = asyncio.Queue()
        self.running = False
        
    def subscribe(self, event_type: EventType, handler: Callable):
        """Subscribe handler to specific event type"""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
        logger.info(f"✅ Subscribed handler to {event_type.value}")
        
    def unsubscribe(self, event_type: EventType, handler: Callable):
        """Unsubscribe handler from event type"""
        if event_type in self.handlers:
            self.handlers[event_type].remove(handler)
            
    async def publish(self, event: Event):
        """Publish event to all subscribers"""
        await self.queue.put(event)
        logger.debug(f"📢 Published {event.event_type.value} event")
        
    async def process_events(self):
        """Process events from queue"""
        self.running = True
        logger.info("🚀 Event bus started")
        
        while self.running:
            try:
                event = await self.queue.get()
                await self._handle_event(event)
                self.queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Error processing event: {e}")
                
    async def _handle_event(self, event: Event):
        """Route event to appropriate handlers"""
        if event.event_type in self.handlers:
            for handler in self.handlers[event.event_type]:
                try:
                    await handler(event)
                except Exception as e:
                    logger.error(f"❌ Error in handler for {event.event_type.value}: {e}")
                    
    def stop(self):
        """Stop event processing"""
        self.running = False
        logger.info("⏹️ Event bus stopped")

# Global event bus instance
event_bus = EventBus()