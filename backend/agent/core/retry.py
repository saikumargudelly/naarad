import time
import random
import asyncio
from typing import Callable, TypeVar, Type, Any
from functools import wraps

T = TypeVar('T')

def retry(
    exceptions: tuple[Type[Exception], ...] = (Exception,),
    max_retries: int = 3,
    initial_delay: float = 0.1,
    max_delay: float = 1.0,
    backoff: float = 2.0,
    jitter: float = 0.1
) -> Callable:
    """
    Retry decorator with exponential backoff and jitter.
    
    Args:
        exceptions: Exceptions to catch and retry on
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff: Backoff multiplier
        jitter: Jitter factor (0 to 1)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            delay = initial_delay
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        raise
                    jitter_amount = random.uniform(0, jitter * delay)
                    actual_delay = min(delay + jitter_amount, max_delay)
                    await asyncio.sleep(actual_delay)
                    delay *= backoff
                    
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            delay = initial_delay
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        raise
                    jitter_amount = random.uniform(0, jitter * delay)
                    actual_delay = min(delay + jitter_amount, max_delay)
                    time.sleep(actual_delay)
                    delay *= backoff
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator
