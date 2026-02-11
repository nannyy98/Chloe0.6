"""
Performance Optimization Module for Chloe AI
Implementation of algorithm optimization, parallel processing, and calculation acceleration
"""

import numpy as np
import pandas as pd
from numba import jit, prange
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from typing import Dict, List, Optional, Tuple, Union
import time
import logging
from functools import lru_cache
import psutil
import gc

logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    """
    Comprehensive performance optimization for Chloe AI
    """
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(32, (mp.cpu_count() or 1) + 4)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=max_workers or mp.cpu_count())
        
        # Cache for frequently used calculations
        self._cache = {}
        self._cache_stats = {'hits': 0, 'misses': 0}
        
        logger.info(f"⚡ Performance Optimizer initialized with {self.max_workers} workers")
    
    def optimize_indicators_calculation(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Optimized technical indicators calculation using vectorization and JIT compilation
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary of optimized indicator arrays
        """
        if data.empty:
            return {}
        
        # Convert to numpy arrays for faster processing
        close_prices = data['close'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        volume = data['volume'].values if 'volume' in data.columns else np.ones(len(data))
        
        indicators = {}
        
        # Calculate indicators in parallel
        futures = {
            'rsi': self.thread_pool.submit(self._calculate_rsi_optimized, close_prices),
            'macd': self.thread_pool.submit(self._calculate_macd_optimized, close_prices),
            'ema': self.thread_pool.submit(self._calculate_ema_optimized, close_prices),
            'bollinger': self.thread_pool.submit(self._calculate_bollinger_optimized, close_prices),
            'stochastic': self.thread_pool.submit(self._calculate_stochastic_optimized, 
                                               high_prices, low_prices, close_prices),
            'volatility': self.thread_pool.submit(self._calculate_volatility_optimized, close_prices)
        }
        
        # Collect results
        for name, future in futures.items():
            try:
                indicators[name] = future.result(timeout=30)
            except Exception as e:
                logger.warning(f"Indicator {name} calculation failed: {e}")
                indicators[name] = np.zeros(len(data))
        
        return indicators
    
    @staticmethod
    @jit(nopython=True, fastmath=True)
    def _calculate_rsi_optimized(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Optimized RSI calculation using Numba JIT compilation"""
        if len(prices) < period + 1:
            return np.zeros(len(prices))
        
        # Calculate price changes
        deltas = np.diff(prices)
        
        # Separate gains and losses
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        
        # Calculate initial averages
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        rsi = np.zeros(len(prices))
        
        # Calculate RSI for each period
        for i in range(period, len(prices)):
            avg_gain = (avg_gain * (period - 1) + gains[i-1]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i-1]) / period
            
            if avg_loss == 0:
                rsi[i] = 100
            else:
                rs = avg_gain / avg_loss
                rsi[i] = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    @jit(nopython=True, fastmath=True)
    def _calculate_macd_optimized(prices: np.ndarray, 
                                fast_period: int = 12,
                                slow_period: int = 26,
                                signal_period: int = 9) -> np.ndarray:
        """Optimized MACD calculation"""
        if len(prices) < slow_period:
            return np.zeros(len(prices))
        
        # Calculate EMAs
        fast_ema = PerformanceOptimizer._calculate_ema_numba(prices, fast_period)
        slow_ema = PerformanceOptimizer._calculate_ema_numba(prices, slow_period)
        
        # Calculate MACD line
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line
        signal_line = PerformanceOptimizer._calculate_ema_numba(macd_line, signal_period)
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return histogram
    
    @staticmethod
    @jit(nopython=True, fastmath=True)
    def _calculate_ema_numba(prices: np.ndarray, period: int) -> np.ndarray:
        """Optimized EMA calculation using Numba"""
        if len(prices) < period:
            return np.zeros(len(prices))
        
        ema = np.zeros(len(prices))
        multiplier = 2.0 / (period + 1)
        
        # Initialize with simple average
        ema[period-1] = np.mean(prices[:period])
        
        # Calculate EMA
        for i in range(period, len(prices)):
            ema[i] = (prices[i] * multiplier) + (ema[i-1] * (1 - multiplier))
        
        return ema
    
    @staticmethod
    def _calculate_ema_optimized(prices: np.ndarray, period: int = 20) -> np.ndarray:
        """Wrapper for EMA calculation"""
        return PerformanceOptimizer._calculate_ema_numba(prices, period)
    
    @staticmethod
    @jit(nopython=True, fastmath=True)
    def _calculate_bollinger_optimized(prices: np.ndarray, 
                                     period: int = 20,
                                     std_dev: float = 2.0) -> np.ndarray:
        """Optimized Bollinger Bands calculation"""
        if len(prices) < period:
            return np.zeros(len(prices))
        
        bb_width = np.zeros(len(prices))
        
        # Calculate rolling standard deviation and mean
        for i in range(period-1, len(prices)):
            window = prices[i-period+1:i+1]
            mean = np.mean(window)
            std = np.std(window)
            bb_width[i] = (std * std_dev * 2) / mean if mean != 0 else 0
        
        return bb_width
    
    @staticmethod
    @jit(nopython=True, fastmath=True)
    def _calculate_stochastic_optimized(high: np.ndarray, low: np.ndarray, 
                                      close: np.ndarray, 
                                      k_period: int = 14) -> np.ndarray:
        """Optimized Stochastic Oscillator calculation"""
        if len(close) < k_period:
            return np.zeros(len(close))
        
        stoch_k = np.zeros(len(close))
        
        for i in range(k_period-1, len(close)):
            highest_high = np.max(high[i-k_period+1:i+1])
            lowest_low = np.min(low[i-k_period+1:i+1])
            
            if highest_high != lowest_low:
                stoch_k[i] = ((close[i] - lowest_low) / (highest_high - lowest_low)) * 100
            else:
                stoch_k[i] = 50  # Neutral value
        
        return stoch_k
    
    @staticmethod
    @jit(nopython=True, fastmath=True)
    def _calculate_volatility_optimized(prices: np.ndarray, period: int = 20) -> np.ndarray:
        """Optimized volatility calculation"""
        if len(prices) < period + 1:
            return np.zeros(len(prices))
        
        returns = np.diff(prices) / prices[:-1]
        volatility = np.zeros(len(prices))
        
        for i in range(period, len(prices)):
            window_returns = returns[i-period:i]
            volatility[i] = np.std(window_returns) * np.sqrt(252)  # Annualized
        
        return volatility
    
    def parallel_data_processing(self, data_chunks: List[pd.DataFrame]) -> List[Dict]:
        """
        Process multiple data chunks in parallel
        
        Args:
            data_chunks: List of DataFrame chunks to process
            
        Returns:
            List of processed results
        """
        if not data_chunks:
            return []
        
        # Submit processing tasks
        futures = [
            self.process_pool.submit(self._process_data_chunk, chunk)
            for chunk in data_chunks
        ]
        
        # Collect results
        results = []
        for future in futures:
            try:
                result = future.result(timeout=60)
                results.append(result)
            except Exception as e:
                logger.error(f"Data chunk processing failed: {e}")
                results.append({})
        
        return results
    
    @staticmethod
    def _process_data_chunk(data_chunk: pd.DataFrame) -> Dict:
        """Process individual data chunk"""
        if data_chunk.empty:
            return {}
        
        # Perform calculations on chunk
        optimizer = PerformanceOptimizer()
        indicators = optimizer.optimize_indicators_calculation(data_chunk)
        
        # Add basic statistics
        stats = {
            'data_points': len(data_chunk),
            'price_range': data_chunk['close'].max() - data_chunk['close'].min(),
            'avg_volume': data_chunk['volume'].mean() if 'volume' in data_chunk.columns else 0,
            'indicators': indicators
        }
        
        return stats
    
    @lru_cache(maxsize=128)
    def cached_calculation(self, calculation_key: str, *args) -> any:
        """
        Cached calculation with automatic cache management
        
        Args:
            calculation_key: Unique key for the calculation
            *args: Arguments for the calculation
            
        Returns:
            Cached or calculated result
        """
        # Check if result is in cache
        cache_key = f"{calculation_key}_{hash(str(args))}"
        
        if cache_key in self._cache:
            self._cache_stats['hits'] += 1
            return self._cache[cache_key]
        
        self._cache_stats['misses'] += 1
        
        # Perform calculation (this would be the actual calculation logic)
        # For demonstration, we'll just return a placeholder
        result = f"calculated_{calculation_key}"
        
        # Store in cache
        self._cache[cache_key] = result
        
        # Clean cache if it gets too large
        if len(self._cache) > 1000:
            # Remove oldest entries
            keys_to_remove = list(self._cache.keys())[:100]
            for key in keys_to_remove:
                del self._cache[key]
        
        return result
    
    def get_performance_stats(self) -> Dict:
        """Get performance optimization statistics"""
        memory_info = psutil.virtual_memory()
        
        return {
            'cache_hits': self._cache_stats['hits'],
            'cache_misses': self._cache_stats['misses'],
            'cache_hit_rate': self._cache_stats['hits'] / max(1, self._cache_stats['hits'] + self._cache_stats['misses']),
            'cache_size': len(self._cache),
            'memory_usage_percent': memory_info.percent,
            'available_memory_gb': memory_info.available / (1024**3),
            'thread_workers': self.max_workers,
            'process_workers': self.process_pool._max_workers if hasattr(self.process_pool, '_max_workers') else 0
        }
    
    def cleanup(self):
        """Cleanup resources"""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        self._cache.clear()
        gc.collect()
        logger.info("⚡ Performance optimizer cleaned up")

# Example usage and benchmarking
def benchmark_optimization():
    """Benchmark performance improvements"""
    import time
    
    # Create sample data
    np.random.seed(42)
    data_size = 10000
    sample_data = pd.DataFrame({
        'close': np.random.randn(data_size).cumsum() + 100,
        'high': np.random.randn(data_size).cumsum() + 105,
        'low': np.random.randn(data_size).cumsum() + 95,
        'volume': np.random.randint(1000, 10000, data_size)
    })
    
    optimizer = PerformanceOptimizer()
    
    print("⚡ Performance Optimization Benchmark")
    print("=" * 50)
    
    # Test optimized calculation
    start_time = time.time()
    indicators = optimizer.optimize_indicators_calculation(sample_data)
    optimized_time = time.time() - start_time
    
    print(f"Optimized calculation time: {optimized_time:.4f} seconds")
    print(f"Data points processed: {len(sample_data)}")
    print(f"Indicators calculated: {len(indicators)}")
    
    # Show performance stats
    stats = optimizer.get_performance_stats()
    print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
    print(f"Memory usage: {stats['memory_usage_percent']:.1f}%")
    
    optimizer.cleanup()

if __name__ == "__main__":
    benchmark_optimization()