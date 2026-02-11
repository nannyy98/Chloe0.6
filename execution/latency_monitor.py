"""
Latency Monitor for Execution System
Monitors round-trip times to different brokers/exchanges
"""
import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime
import time
from collections import deque

logger = logging.getLogger(__name__)


class LatencyMonitor:
    """Monitors latency to various brokers/exchanges"""
    
    def __init__(self, max_samples: int = 100):
        self.latencies: Dict[str, deque] = {}  # broker_name -> list of latency samples
        self.is_monitoring = False
        self.max_samples = max_samples
        self.monitor_tasks = {}
        
    def record_latency(self, broker_name: str, latency_ms: float):
        """Record a latency sample for a specific broker"""
        if broker_name not in self.latencies:
            self.latencies[broker_name] = deque(maxlen=self.max_samples)
        
        self.latencies[broker_name].append(latency_ms)
        logger.debug(f"⏱️ Latency recorded for {broker_name}: {latency_ms:.2f}ms")
    
    def get_average_latency(self, broker_name: str) -> float:
        """Get average latency for a specific broker"""
        if broker_name not in self.latencies or len(self.latencies[broker_name]) == 0:
            return 999.0  # Return high latency if no data
        
        return sum(self.latencies[broker_name]) / len(self.latencies[broker_name])
    
    def get_recent_latency(self, broker_name: str, num_samples: int = 5) -> float:
        """Get average of recent latency samples for a broker"""
        if broker_name not in self.latencies or len(self.latencies[broker_name]) == 0:
            return 999.0
        
        recent_samples = list(self.latencies[broker_name])[-num_samples:]
        return sum(recent_samples) / len(recent_samples)
    
    def get_all_latencies(self) -> Dict[str, float]:
        """Get average latencies for all monitored brokers"""
        return {name: self.get_average_latency(name) for name in self.latencies}
    
    def get_latency_stats(self, broker_name: str) -> Dict:
        """Get detailed latency statistics for a broker"""
        if broker_name not in self.latencies or len(self.latencies[broker_name]) == 0:
            return {
                'average': 999.0,
                'min': 999.0,
                'max': 999.0,
                'count': 0,
                'percentile_95': 999.0
            }
        
        samples = list(self.latencies[broker_name])
        sorted_samples = sorted(samples)
        
        return {
            'average': sum(samples) / len(samples),
            'min': min(samples),
            'max': max(samples),
            'count': len(samples),
            'percentile_95': sorted_samples[int(len(sorted_samples) * 0.95)] if samples else 0
        }
    
    async def ping_broker(self, broker_name: str, broker_adapter) -> float:
        """Ping a broker to measure latency"""
        start_time = time.time()
        
        try:
            # Try to get a simple piece of data to measure round-trip time
            # This could be a ticker, ping endpoint, or other lightweight request
            if hasattr(broker_adapter, 'get_market_price'):
                # Try to get a price for a common symbol
                await broker_adapter.get_market_price('BTCUSDT')
            elif hasattr(broker_adapter, 'get_account_balance'):
                # Otherwise, try to get account balance
                await broker_adapter.get_account_balance()
            else:
                # Fallback: just try connecting
                if hasattr(broker_adapter, 'connect'):
                    await broker_adapter.connect()
            
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            self.record_latency(broker_name, latency_ms)
            return latency_ms
            
        except Exception as e:
            logger.warning(f"⚠️ Ping failed for {broker_name}: {e}")
            # Record a high latency to indicate issues
            self.record_latency(broker_name, 9999.0)
            return 9999.0
    
    async def start_monitoring(self):
        """Start monitoring latencies continuously"""
        self.is_monitoring = True
        logger.info("📡 Starting latency monitoring")
    
    async def stop_monitoring(self):
        """Stop monitoring latencies"""
        self.is_monitoring = False
        logger.info("📡 Stopped latency monitoring")
    
    def add_broker_for_monitoring(self, broker_name: str, broker_adapter, ping_interval: int = 30):
        """Add a broker to be monitored with a specific ping interval"""
        if self.is_monitoring:
            # Start monitoring this broker
            task = asyncio.create_task(self._monitor_broker(broker_name, broker_adapter, ping_interval))
            self.monitor_tasks[broker_name] = task
            logger.info(f"🔍 Started monitoring {broker_name} every {ping_interval}s")
    
    async def _monitor_broker(self, broker_name: str, broker_adapter, interval: int):
        """Monitor a single broker continuously"""
        while self.is_monitoring:
            try:
                await self.ping_broker(broker_name, broker_adapter)
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"❌ Error monitoring {broker_name}: {e}")
                await asyncio.sleep(interval)  # Continue monitoring despite errors


class ExecutionTimeTracker:
    """Tracks execution times for different order types and brokers"""
    
    def __init__(self, max_samples: int = 100):
        self.execution_times: Dict[str, deque] = {}  # "broker_type_side" -> list of execution times
        self.max_samples = max_samples
    
    def start_timer(self, broker_name: str, order_type: str, side: str) -> float:
        """Start timer for an execution"""
        key = f"{broker_name}_{order_type}_{side}"
        start_time = time.time()
        return start_time
    
    def record_execution_time(self, start_time: float, broker_name: str, order_type: str, side: str):
        """Record execution time for an order"""
        key = f"{broker_name}_{order_type}_{side}"
        if key not in self.execution_times:
            self.execution_times[key] = deque(maxlen=self.max_samples)
        
        execution_time_ms = (time.time() - start_time) * 1000
        self.execution_times[key].append(execution_time_ms)
        logger.debug(f"⏱️ Execution time for {key}: {execution_time_ms:.2f}ms")
    
    def get_average_execution_time(self, broker_name: str, order_type: str, side: str) -> float:
        """Get average execution time for specific combination"""
        key = f"{broker_name}_{order_type}_{side}"
        if key not in self.execution_times or len(self.execution_times[key]) == 0:
            return 999.0
        
        return sum(self.execution_times[key]) / len(self.execution_times[key])
    
    def get_execution_stats(self, broker_name: str, order_type: str, side: str) -> Dict:
        """Get detailed execution time statistics"""
        key = f"{broker_name}_{order_type}_{side}"
        if key not in self.execution_times or len(self.execution_times[key]) == 0:
            return {
                'average': 999.0,
                'min': 999.0,
                'max': 999.0,
                'count': 0
            }
        
        samples = list(self.execution_times[key])
        return {
            'average': sum(samples) / len(samples),
            'min': min(samples),
            'max': max(samples),
            'count': len(samples)
        }


class SlippageTracker:
    """Tracks slippage for different order types and market conditions"""
    
    def __init__(self, max_samples: int = 100):
        self.slippage_data: Dict[str, deque] = {}  # "broker_symbol" -> list of slippage values
        self.max_samples = max_samples
    
    def record_slippage(self, broker_name: str, symbol: str, expected_price: float, actual_price: float):
        """Record slippage for an executed order"""
        key = f"{broker_name}_{symbol}"
        if key not in self.slippage_data:
            self.slippage_data[key] = deque(maxlen=self.max_samples)
        
        if expected_price != 0:
            slippage = abs(actual_price - expected_price) / expected_price
            self.slippage_data[key].append(slippage)
            logger.debug(f"📉 Slippage for {key}: {slippage:.4f} ({slippage*100:.2f}%)")
        else:
            logger.warning(f"⚠️ Cannot calculate slippage, expected price is 0 for {key}")
    
    def get_average_slippage(self, broker_name: str, symbol: str) -> float:
        """Get average slippage for a specific broker and symbol"""
        key = f"{broker_name}_{symbol}"
        if key not in self.slippage_data or len(self.slippage_data[key]) == 0:
            return 0.001  # Default 0.1% slippage
        
        return sum(self.slippage_data[key]) / len(self.slippage_data[key])
    
    def get_slippage_stats(self, broker_name: str, symbol: str) -> Dict:
        """Get detailed slippage statistics"""
        key = f"{broker_name}_{symbol}"
        if key not in self.slippage_data or len(self.slippage_data[key]) == 0:
            return {
                'average': 0.001,
                'min': 0.0,
                'max': 0.01,
                'count': 0
            }
        
        samples = list(self.slippage_data[key])
        return {
            'average': sum(samples) / len(samples),
            'min': min(samples),
            'max': max(samples),
            'count': len(samples)
        }


# Global instances
latency_monitor = LatencyMonitor()
execution_tracker = ExecutionTimeTracker()
slippage_tracker = SlippageTracker()


def initialize_latency_monitor():
    """Initialize global latency monitoring components"""
    global latency_monitor, execution_tracker, slippage_tracker
    latency_monitor = LatencyMonitor()
    execution_tracker = ExecutionTimeTracker()
    slippage_tracker = SlippageTracker()
    logger.info("📡 Latency monitoring components initialized")
    return latency_monitor, execution_tracker, slippage_tracker