"""
Retry Handler for Market Data Connections
Implements exponential backoff and jitter for resilient connections
"""
import asyncio
import logging
import random
from typing import Callable, Any, Optional, Dict
from datetime import datetime

logger = logging.getLogger(__name__)


class RetryConfig:
    """Configuration for retry behavior"""
    
    def __init__(self,
                 max_attempts: int = 10,
                 base_delay: float = 1.0,  # seconds
                 max_delay: float = 60.0,  # seconds
                 backoff_multiplier: float = 2.0,
                 jitter: bool = True,
                 retryable_exceptions: tuple = (Exception,)):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions


class RetryHandler:
    """Handles retry logic with exponential backoff and jitter"""
    
    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()
        self.stats = {
            'total_retries': 0,
            'successful_after_retry': 0,
            'failed_after_max_attempts': 0,
            'functions_called': {}
        }
    
    async def execute_with_retry(self, 
                                func: Callable, 
                                *args, 
                                **kwargs) -> Any:
        """
        Execute a function with retry logic
        
        Args:
            func: Function to execute
            *args: Arguments to pass to function
            **kwargs: Keyword arguments to pass to function
            
        Returns:
            Result of function execution
            
        Raises:
            Exception: If max retries exceeded or non-retryable exception occurs
        """
        last_exception = None
        attempt = 0
        
        for attempt in range(self.config.max_attempts):
            try:
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                
                # Update stats on success
                if attempt > 0:
                    self.stats['successful_after_retry'] += 1
                
                function_name = getattr(func, '__name__', str(func))
                if function_name not in self.stats['functions_called']:
                    self.stats['functions_called'][function_name] = {'attempts': 0, 'successes': 0}
                self.stats['functions_called'][function_name]['successes'] += 1
                
                return result
                
            except self.config.retryable_exceptions as e:
                last_exception = e
                self.stats['total_retries'] += 1
                
                if attempt == self.config.max_attempts - 1:
                    # Last attempt - update failure stats
                    self.stats['failed_after_max_attempts'] += 1
                    function_name = getattr(func, '__name__', str(func))
                    if function_name not in self.stats['functions_called']:
                        self.stats['functions_called'][function_name] = {'attempts': 0, 'failures': 0}
                    if 'failures' not in self.stats['functions_called'][function_name]:
                        self.stats['functions_called'][function_name]['failures'] = 0
                    self.stats['functions_called'][function_name]['failures'] += 1
                    
                    logger.error(f"❌ Function {func.__name__} failed after {self.config.max_attempts} attempts: {e}")
                    raise e
                
                # Calculate delay with exponential backoff
                delay = min(
                    self.config.base_delay * (self.config.backoff_multiplier ** attempt),
                    self.config.max_delay
                )
                
                # Add jitter if enabled
                if self.config.jitter:
                    jitter_factor = random.uniform(0.5, 1.0)
                    delay *= jitter_factor
                
                logger.warning(
                    f"⚠️ Function {func.__name__} failed on attempt {attempt + 1}/{self.config.max_attempts}, "
                    f"retrying in {delay:.2f}s: {e}"
                )
                
                await asyncio.sleep(delay)
        
        # This shouldn't be reached, but just in case
        raise last_exception
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for a specific attempt"""
        delay = min(
            self.config.base_delay * (self.config.backoff_multiplier ** attempt),
            self.config.max_delay
        )
        
        if self.config.jitter:
            jitter_factor = random.uniform(0.5, 1.0)
            delay *= jitter_factor
        
        return delay
    
    def get_stats(self) -> Dict:
        """Get retry statistics"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset retry statistics"""
        self.stats = {
            'total_retries': 0,
            'successful_after_retry': 0,
            'failed_after_max_attempts': 0,
            'functions_called': {}
        }


class ConnectionRetryHandler(RetryHandler):
    """Specialized retry handler for connection operations"""
    
    def __init__(self, config: RetryConfig = None):
        # Specific exceptions for connection retries
        conn_config = config or RetryConfig(
            retryable_exceptions=(
                ConnectionError, 
                TimeoutError, 
                asyncio.TimeoutError,
                OSError
            )
        )
        super().__init__(conn_config)
    
    async def connect_with_retry(self, connect_func: Callable, *args, **kwargs) -> bool:
        """Execute connection function with retry logic"""
        try:
            result = await self.execute_with_retry(connect_func, *args, **kwargs)
            return result if isinstance(result, bool) else True
        except Exception as e:
            logger.error(f"❌ Connection failed permanently: {e}")
            return False
    
    async def reconnect_with_retry(self, 
                                  connect_func: Callable, 
                                  disconnect_func: Callable = None, 
                                  *args, 
                                  **kwargs) -> bool:
        """Execute reconnection with proper disconnect/connect sequence"""
        # Optionally disconnect first
        if disconnect_func:
            try:
                await disconnect_func()
            except Exception as e:
                logger.warning(f"Warning during disconnect: {e}")
        
        # Connect with retry
        return await self.connect_with_retry(connect_func, *args, **kwargs)


class DataStreamRetryHandler(RetryHandler):
    """Specialized retry handler for data stream operations"""
    
    def __init__(self, config: RetryConfig = None):
        # Specific exceptions for data stream retries
        stream_config = config or RetryConfig(
            retryable_exceptions=(
                ConnectionError,
                TimeoutError,
                asyncio.TimeoutError,
                BrokenPipeError,
                EOFError
            ),
            max_attempts=5  # Fewer attempts for streaming since we want to fail fast sometimes
        )
        super().__init__(stream_config)
    
    async def handle_stream_interrupt(self, 
                                   stream_func: Callable, 
                                   restart_func: Callable,
                                   *args, 
                                   **kwargs) -> Any:
        """Handle interruption and restart of data stream"""
        try:
            return await self.execute_with_retry(stream_func, *args, **kwargs)
        except Exception as e:
            logger.error(f"Stream failed permanently: {e}")
            # Attempt to restart the stream
            try:
                await restart_func(*args, **kwargs)
                logger.info("✅ Stream restarted successfully")
            except Exception as restart_e:
                logger.error(f"❌ Stream restart failed: {restart_e}")
                raise


def create_default_retry_handler() -> RetryHandler:
    """Create a default retry handler with sensible defaults for market data"""
    config = RetryConfig(
        max_attempts=5,
        base_delay=1.0,
        max_delay=30.0,
        backoff_multiplier=2.0,
        jitter=True,
        retryable_exceptions=(
            ConnectionError,
            TimeoutError,
            asyncio.TimeoutError,
            OSError,
            RuntimeError
        )
    )
    return RetryHandler(config)


def create_connection_retry_handler() -> ConnectionRetryHandler:
    """Create a retry handler optimized for connection operations"""
    config = RetryConfig(
        max_attempts=10,
        base_delay=0.5,
        max_delay=60.0,
        backoff_multiplier=2.0,
        jitter=True,
        retryable_exceptions=(
            ConnectionError,
            TimeoutError,
            asyncio.TimeoutError,
            OSError
        )
    )
    return ConnectionRetryHandler(config)


def create_stream_retry_handler() -> DataStreamRetryHandler:
    """Create a retry handler optimized for data streaming"""
    config = RetryConfig(
        max_attempts=3,
        base_delay=1.0,
        max_delay=10.0,
        backoff_multiplier=2.0,
        jitter=True,
        retryable_exceptions=(
            ConnectionError,
            TimeoutError,
            asyncio.TimeoutError,
            BrokenPipeError,
            EOFError
        )
    )
    return DataStreamRetryHandler(config)


# Decorator for easy retry functionality
def retry_on_failure(config: RetryConfig = None):
    """Decorator to add retry functionality to functions"""
    def decorator(func):
        retry_handler = RetryHandler(config)
        
        async def async_wrapper(*args, **kwargs):
            return await retry_handler.execute_with_retry(func, *args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            return retry_handler.execute_with_retry(func, *args, **kwargs)
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Example usage
if __name__ == "__main__":
    # Example configuration
    config = RetryConfig(
        max_attempts=3,
        base_delay=1.0,
        max_delay=10.0,
        backoff_multiplier=2.0,
        jitter=True
    )
    
    retry_handler = RetryHandler(config)
    
    # Example async function that might fail
    async def flaky_function():
        import random
        if random.random() < 0.7:  # 70% chance of failure
            raise ConnectionError("Random connection error")
        return "Success!"
    
    # Execute with retry
    async def test_retry():
        try:
            result = await retry_handler.execute_with_retry(flaky_function)
            print(f"Function succeeded: {result}")
        except Exception as e:
            print(f"Function failed permanently: {e}")
        
        print(f"Retry stats: {retry_handler.get_stats()}")
    
    # Run the test
    asyncio.run(test_retry())