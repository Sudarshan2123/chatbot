"""
Performance monitoring for SQL agent
"""
import time
import logging
from functools import wraps
from typing import Dict, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor and log SQL agent performance metrics"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.query_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
    
    def track_query(self, query_type: str, duration: float):
        """Track query execution time"""
        self.metrics[query_type].append(duration)
        self.query_count += 1
    
    def record_cache_hit(self):
        """Record cache hit"""
        self.cache_hits += 1
    
    def record_cache_miss(self):
        """Record cache miss"""
        self.cache_misses += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {
            "total_queries": self.query_count,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": (
                self.cache_hits / (self.cache_hits + self.cache_misses) * 100
                if (self.cache_hits + self.cache_misses) > 0 else 0
            ),
            "avg_query_times": {}
        }
        
        for query_type, durations in self.metrics.items():
            if durations:
                stats["avg_query_times"][query_type] = {
                    "avg": sum(durations) / len(durations),
                    "min": min(durations),
                    "max": max(durations),
                    "count": len(durations)
                }
        
        return stats
    
    def reset(self):
        """Reset all metrics"""
        self.metrics.clear()
        self.query_count = 0
        self.cache_hits = 0
        self.cache_misses = 0


def time_execution(func):
    """Decorator to time function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        logger.info(f"{func.__name__} took {duration:.2f}s")
        return result
    return wrapper


# Global monitor instance
performance_monitor = PerformanceMonitor()
