"""
Real-time API Endpoints for Chloe AI
WebSocket and REST endpoints for real-time data streaming
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from pydantic import BaseModel
from typing import List, Dict, Optional
import json
import logging
from datetime import datetime
import asyncio

# Import real-time components
from realtime.data_pipeline import RealTimeDataPipeline, RealTimeSignal
from realtime.websocket_client import MarketData, OrderBook

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/realtime", tags=["realtime"])

# Global pipeline instance (in production, use proper dependency injection)
pipeline = None
connected_clients = []

class SymbolRequest(BaseModel):
    symbols: List[str]

class RealTimeSignalResponse(BaseModel):
    symbol: str
    signal: str
    confidence: float
    price: float
    timestamp: str
    risk_level: str
    explanation: str

class MarketDataResponse(BaseModel):
    symbol: str
    price: float
    change_percent: float
    volume: float
    timestamp: str
    data_quality: float

def get_pipeline():
    """Get or create real-time pipeline instance"""
    global pipeline
    if pipeline is None:
        pipeline = RealTimeDataPipeline()
    return pipeline

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time data streaming"""
    await websocket.accept()
    connected_clients.append(websocket)
    logger.info(f"🔌 New WebSocket client connected. Total clients: {len(connected_clients)}")
    
    try:
        # Send initial connection confirmation
        await websocket.send_text(json.dumps({
            "type": "connection",
            "status": "connected",
            "timestamp": datetime.now().isoformat(),
            "supported_symbols": ["BTCUSDT", "ETHUSDT"]
        }))
        
        # Keep connection alive
        while True:
            try:
                # Wait for client messages (optional)
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                # Handle client commands if needed
                await _handle_client_message(websocket, data)
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_text(json.dumps({
                    "type": "heartbeat",
                    "timestamp": datetime.now().isoformat()
                }))
            except Exception as e:
                logger.error(f"❌ WebSocket client error: {e}")
                break
                
    except WebSocketDisconnect:
        logger.info("🔌 WebSocket client disconnected")
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
    finally:
        if websocket in connected_clients:
            connected_clients.remove(websocket)
        logger.info(f"🧹 Cleaned up WebSocket connection. Remaining clients: {len(connected_clients)}")

async def _handle_client_message(websocket: WebSocket, message: str):
    """Handle incoming WebSocket messages from clients"""
    try:
        data = json.loads(message)
        message_type = data.get('type')
        
        if message_type == 'subscribe':
            symbols = data.get('symbols', [])
            # Handle subscription logic
            await websocket.send_text(json.dumps({
                "type": "subscription_confirmed",
                "symbols": symbols,
                "timestamp": datetime.now().isoformat()
            }))
            
        elif message_type == 'unsubscribe':
            # Handle unsubscription logic
            await websocket.send_text(json.dumps({
                "type": "unsubscription_confirmed",
                "timestamp": datetime.now().isoformat()
            }))
            
    except json.JSONDecodeError:
        logger.warning("⚠️ Invalid JSON from WebSocket client")
    except Exception as e:
        logger.error(f"❌ Error handling client message: {e}")

async def broadcast_signal(signal: RealTimeSignal):
    """Broadcast trading signal to all connected clients"""
    if not connected_clients:
        return
    
    message = {
        "type": "signal",
        "data": {
            "symbol": signal.symbol,
            "signal": signal.signal,
            "confidence": signal.confidence,
            "price": signal.price,
            "timestamp": signal.timestamp.isoformat(),
            "risk_level": signal.risk_level,
            "explanation": signal.explanation
        }
    }
    
    # Send to all connected clients
    disconnected_clients = []
    for client in connected_clients:
        try:
            await client.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"❌ Error broadcasting to client: {e}")
            disconnected_clients.append(client)
    
    # Clean up disconnected clients
    for client in disconnected_clients:
        if client in connected_clients:
            connected_clients.remove(client)

async def broadcast_market_data(market_data: MarketData):
    """Broadcast market data to all connected clients"""
    if not connected_clients:
        return
    
    message = {
        "type": "market_data",
        "data": {
            "symbol": market_data.symbol,
            "price": market_data.price,
            "change_percent": market_data.price_change_percent,
            "volume": market_data.volume,
            "timestamp": market_data.timestamp.isoformat()
        }
    }
    
    # Send to all connected clients
    disconnected_clients = []
    for client in connected_clients:
        try:
            await client.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"❌ Error broadcasting market data: {e}")
            disconnected_clients.append(client)
    
    # Clean up disconnected clients
    for client in disconnected_clients:
        if client in connected_clients:
            connected_clients.remove(client)

@router.post("/start", response_model=Dict)
async def start_realtime_pipeline(request: SymbolRequest):
    """Start real-time data processing pipeline"""
    try:
        pipeline = get_pipeline()
        pipeline.symbols = request.symbols
        
        # Start pipeline in background task
        task = asyncio.create_task(pipeline.start_pipeline())
        
        # Set up callbacks to broadcast data
        def signal_callback(signal: RealTimeSignal):
            asyncio.create_task(broadcast_signal(signal))
        
        def alert_callback(alert_type: str, symbol: str, data: Dict):
            message = {
                "type": "alert",
                "alert_type": alert_type,
                "symbol": symbol,
                "data": data,
                "timestamp": datetime.now().isoformat()
            }
            # Broadcast to connected clients
            for client in connected_clients:
                try:
                    asyncio.create_task(client.send_text(json.dumps(message)))
                except Exception:
                    pass  # Ignore send errors
        
        pipeline.add_signal_callback(signal_callback)
        pipeline.add_alert_callback(alert_callback)
        
        logger.info(f"🚀 Real-time pipeline started for symbols: {request.symbols}")
        
        return {
            "status": "started",
            "symbols": request.symbols,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error starting real-time pipeline: {e}")
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }

@router.get("/status", response_model=Dict)
async def get_realtime_status():
    """Get real-time pipeline status"""
    try:
        pipeline = get_pipeline()
        status = {
            "is_running": pipeline.is_processing,
            "monitored_symbols": pipeline.symbols,
            "connected_clients": len(connected_clients),
            "timestamp": datetime.now().isoformat()
        }
        
        # Add symbol-specific status
        symbol_status = {}
        for symbol in pipeline.symbols:
            symbol_status[symbol] = pipeline.get_current_state(symbol)
        status["symbol_status"] = symbol_status
        
        return status
        
    except Exception as e:
        logger.error(f"❌ Error getting real-time status: {e}")
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }

@router.get("/signals", response_model=List[RealTimeSignalResponse])
async def get_current_signals():
    """Get current trading signals for all symbols"""
    try:
        pipeline = get_pipeline()
        signals = pipeline.get_portfolio_signals()
        
        return [
            RealTimeSignalResponse(
                symbol=signal.symbol,
                signal=signal.signal,
                confidence=signal.confidence,
                price=signal.price,
                timestamp=signal.timestamp.isoformat(),
                risk_level=signal.risk_level,
                explanation=signal.explanation
            )
            for signal in signals
        ]
        
    except Exception as e:
        logger.error(f"❌ Error getting current signals: {e}")
        return []

@router.get("/data/{symbol}", response_model=MarketDataResponse)
async def get_symbol_data(symbol: str):
    """Get current market data for a symbol"""
    try:
        pipeline = get_pipeline()
        state = pipeline.get_current_state(symbol)
        
        if state['latest_price'] is None:
            return {"error": "No data available for symbol"}
        
        return MarketDataResponse(
            symbol=symbol,
            price=state['latest_price'],
            change_percent=0.0,  # Would be calculated from price history
            volume=0.0,  # Would be calculated from recent volume
            timestamp=datetime.now().isoformat(),
            data_quality=state['data_quality']
        )
        
    except Exception as e:
        logger.error(f"❌ Error getting symbol data: {e}")
        return {"error": str(e)}

@router.get("/alerts", response_model=Dict)
async def get_recent_alerts():
    """Get recent system alerts (mock implementation)"""
    # In a real implementation, this would return recent alerts from a database
    return {
        "alerts": [
            {
                "type": "PRICE_MOVEMENT",
                "symbol": "BTCUSDT",
                "message": "Significant price movement detected",
                "timestamp": datetime.now().isoformat()
            }
        ],
        "timestamp": datetime.now().isoformat()
    }

# Register the WebSocket endpoint separately in main API
# The router approach above has limitations for WebSocket routing in FastAPI

# Example usage documentation:
"""
Real-time API Usage Examples:

WebSocket Connection:
1. Connect to: ws://localhost:8000/realtime/ws
2. Subscribe to symbols: {"type": "subscribe", "symbols": ["BTCUSDT", "ETHUSDT"]}

REST API Endpoints:
GET /realtime/status - Get pipeline status
POST /realtime/start - Start pipeline with symbols
GET /realtime/signals - Get current trading signals
GET /realtime/data/{symbol} - Get market data for symbol
GET /realtime/alerts - Get recent alerts
"""

if __name__ == "__main__":
    # This file is meant to be imported as router module
    pass