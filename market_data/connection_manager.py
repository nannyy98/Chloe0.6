"""
Connection Manager for Market Data Adapters
Handles connection lifecycle, reconnections, and health monitoring
"""
import asyncio
import logging
from typing import Dict, List, Optional, Callable
from datetime import datetime
import time

from market_data.adapters.base_adapter import BaseMarketDataAdapter

logger = logging.getLogger(__name__)


class ConnectionHealthMonitor:
    """Monitors connection health and performance"""
    
    def __init__(self, adapter_name: str):
        self.adapter_name = adapter_name
        self.last_message_time = None
        self.message_count = 0
        self.error_count = 0
        self.connection_start_time = None
        self.latency_samples = []
        
    def record_message(self):
        """Record receipt of a market data message"""
        self.message_count += 1
        self.last_message_time = datetime.now()
    
    def record_error(self):
        """Record an error"""
        self.error_count += 1
    
    def record_latency(self, latency_ms: float):
        """Record latency measurement"""
        self.latency_samples.append(latency_ms)
        # Keep only last 100 samples
        if len(self.latency_samples) > 100:
            self.latency_samples = self.latency_samples[-100:]
    
    def get_health_status(self) -> Dict:
        """Get current health status"""
        now = datetime.now()
        time_since_last_msg = None
        if self.last_message_time:
            time_since_last_msg = (now - self.last_message_time).total_seconds()
        
        avg_latency = sum(self.latency_samples) / len(self.latency_samples) if self.latency_samples else 0
        
        return {
            'adapter_name': self.adapter_name,
            'last_message_time': self.last_message_time,
            'time_since_last_message': time_since_last_msg,
            'message_count': self.message_count,
            'error_count': self.error_count,
            'average_latency_ms': avg_latency,
            'uptime_seconds': (now - self.connection_start_time).total_seconds() if self.connection_start_time else 0
        }


class ConnectionManager:
    """Manages connections to multiple market data adapters"""
    
    def __init__(self):
        self.adapters: Dict[str, BaseMarketDataAdapter] = {}
        self.health_monitors: Dict[str, ConnectionHealthMonitor] = {}
        self.is_running = False
        self.reconnection_tasks = {}
        
    def register_adapter(self, adapter: BaseMarketDataAdapter, adapter_id: str = None):
        """Register a market data adapter"""
        adapter_id = adapter_id or adapter.exchange_name
        self.adapters[adapter_id] = adapter
        self.health_monitors[adapter_id] = ConnectionHealthMonitor(adapter_id)
        logger.info(f"✅ Registered adapter: {adapter_id}")
    
    async def connect_all(self) -> Dict[str, bool]:
        """Connect to all registered adapters"""
        results = {}
        for adapter_id, adapter in self.adapters.items():
            try:
                success = await adapter.connect()
                results[adapter_id] = success
                if success:
                    self.health_monitors[adapter_id].connection_start_time = datetime.now()
                    logger.info(f"✅ Connected to {adapter_id}")
                else:
                    logger.error(f"❌ Failed to connect to {adapter_id}")
            except Exception as e:
                logger.error(f"❌ Error connecting to {adapter_id}: {e}")
                results[adapter_id] = False
        
        return results
    
    async def disconnect_all(self) -> Dict[str, bool]:
        """Disconnect from all adapters"""
        results = {}
        for adapter_id, adapter in self.adapters.items():
            try:
                success = await adapter.disconnect()
                results[adapter_id] = success
                if success:
                    logger.info(f"✅ Disconnected from {adapter_id}")
                else:
                    logger.error(f"❌ Failed to disconnect from {adapter_id}")
            except Exception as e:
                logger.error(f"❌ Error disconnecting from {adapter_id}: {e}")
                results[adapter_id] = False
        
        return results
    
    async def start_monitoring(self):
        """Start monitoring all connections"""
        self.is_running = True
        logger.info("🔍 Started connection monitoring")
        
        while self.is_running:
            for adapter_id, monitor in self.health_monitors.items():
                health = monitor.get_health_status()
                
                # Check for potential issues
                if health['time_since_last_message'] and health['time_since_last_message'] > 30:
                    logger.warning(f"⚠️ Potential disconnection for {adapter_id} - no messages for {health['time_since_last_message']}s")
                    
                    # Attempt reconnection if needed
                    if not self.adapters[adapter_id].is_connected:
                        logger.info(f"🔄 Attempting reconnection for {adapter_id}")
                        await self._attempt_reconnection(adapter_id)
                
                # Log health status periodically
                if health['message_count'] % 100 == 0:  # Every 100 messages
                    logger.debug(f"📊 {adapter_id} health: {health}")
            
            await asyncio.sleep(5)  # Check every 5 seconds
    
    async def _attempt_reconnection(self, adapter_id: str):
        """Attempt to reconnect a specific adapter"""
        if adapter_id in self.reconnection_tasks and not self.reconnection_tasks[adapter_id].done():
            logger.debug(f"🔄 Reconnection already in progress for {adapter_id}")
            return
        
        async def reconnect_task():
            max_attempts = 5
            base_delay = 1  # seconds
            
            for attempt in range(max_attempts):
                try:
                    logger.info(f"🔄 Reconnection attempt {attempt + 1}/{max_attempts} for {adapter_id}")
                    
                    # Disconnect first if somehow partially connected
                    try:
                        await self.adapters[adapter_id].disconnect()
                    except:
                        pass  # Ignore errors during disconnect
                    
                    # Attempt reconnection
                    success = await self.adapters[adapter_id].connect()
                    if success:
                        logger.info(f"✅ Successfully reconnected to {adapter_id}")
                        self.health_monitors[adapter_id].connection_start_time = datetime.now()
                        return True
                    
                except Exception as e:
                    logger.error(f"❌ Reconnection attempt {attempt + 1} failed for {adapter_id}: {e}")
                
                # Wait before next attempt (exponential backoff)
                if attempt < max_attempts - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.info(f"⏳ Waiting {delay}s before next reconnection attempt for {adapter_id}")
                    await asyncio.sleep(delay)
            
            logger.error(f"❌ Failed to reconnect to {adapter_id} after {max_attempts} attempts")
            return False
        
        self.reconnection_tasks[adapter_id] = asyncio.create_task(reconnect_task())
        
        try:
            result = await self.reconnection_tasks[adapter_id]
            return result
        except Exception as e:
            logger.error(f"❌ Reconnection task failed for {adapter_id}: {e}")
            return False
    
    def get_health_report(self) -> Dict[str, Dict]:
        """Get comprehensive health report for all connections"""
        report = {}
        for adapter_id, monitor in self.health_monitors.items():
            report[adapter_id] = monitor.get_health_status()
        return report
    
    def get_active_connections(self) -> List[str]:
        """Get list of currently active connections"""
        active = []
        for adapter_id, adapter in self.adapters.items():
            if adapter.is_connected:
                active.append(adapter_id)
        return active
    
    def get_failed_connections(self) -> List[str]:
        """Get list of failed/disconnected connections"""
        failed = []
        for adapter_id, adapter in self.adapters.items():
            if not adapter.is_connected:
                failed.append(adapter_id)
        return failed
    
    async def stop_monitoring(self):
        """Stop monitoring all connections"""
        self.is_running = False
        logger.info("🔍 Stopped connection monitoring")


class MarketDataGateway:
    """Main market data gateway that manages all connections and routes data"""
    
    def __init__(self):
        self.connection_manager = ConnectionManager()
        self.data_handlers = {}
        self.is_running = False
        
    def register_adapter(self, adapter, adapter_id: str = None):
        """Register a market data adapter with the gateway"""
        self.connection_manager.register_adapter(adapter, adapter_id)
    
    def subscribe_to_data(self, data_type: str, handler: Callable):
        """Subscribe to specific data type with a handler"""
        if data_type not in self.data_handlers:
            self.data_handlers[data_type] = []
        self.data_handlers[data_type].append(handler)
    
    async def start(self):
        """Start the market data gateway"""
        logger.info("🚀 Starting Market Data Gateway")
        
        # Connect to all adapters
        connection_results = await self.connection_manager.connect_all()
        successful_connections = [k for k, v in connection_results.items() if v]
        
        if not successful_connections:
            logger.error("❌ No adapters connected successfully")
            return False
        
        logger.info(f"✅ Successfully connected to {len(successful_connections)} adapters: {successful_connections}")
        
        # Start monitoring
        monitor_task = asyncio.create_task(self.connection_manager.start_monitoring())
        
        self.is_running = True
        
        try:
            # Keep the gateway running
            while self.is_running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("🛑 Received shutdown signal")
        finally:
            await self.stop()
            await monitor_task
    
    async def stop(self):
        """Stop the market data gateway"""
        logger.info("🛑 Stopping Market Data Gateway")
        
        self.is_running = False
        
        # Disconnect from all adapters
        await self.connection_manager.disconnect_all()
        
        # Stop monitoring
        await self.connection_manager.stop_monitoring()
        
        logger.info("✅ Market Data Gateway stopped")
    
    def get_status(self) -> Dict:
        """Get current gateway status"""
        return {
            'is_running': self.is_running,
            'active_connections': self.connection_manager.get_active_connections(),
            'failed_connections': self.connection_manager.get_failed_connections(),
            'health_report': self.connection_manager.get_health_report(),
            'registered_handlers': list(self.data_handlers.keys())
        }