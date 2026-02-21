"""
Query optimization utilities for SQL agent performance
"""
import logging
from typing import List, Dict, Any
from functools import lru_cache

logger = logging.getLogger(__name__)


class QueryOptimizer:
    """Optimize SQL queries for better performance"""
    
    @staticmethod
    @lru_cache(maxsize=128)
    def get_optimized_query(table_name: str, columns: tuple, limit: int = None) -> str:
        """Generate optimized SELECT query with caching"""
        cols = ", ".join(columns) if columns else "*"
        query = f"SELECT {cols} FROM {table_name}"
        if limit:
            query += f" WHERE ROWNUM <= {limit}"
        return query
    
    @staticmethod
    def add_query_hints(query: str, hint: str = "/*+ FIRST_ROWS(100) */") -> str:
        """Add Oracle optimizer hints for faster results"""
        if "SELECT" in query.upper() and "/*+" not in query:
            return query.replace("SELECT", f"SELECT {hint}", 1)
        return query
    
    @staticmethod
    def batch_queries(queries: List[str]) -> str:
        """Combine multiple queries into single batch"""
        return ";\n".join(queries)


class TablePreloader:
    """Preload frequently accessed tables on startup"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.preload_config = {
            # Add your most frequently queried tables here
            "hot_tables": [],  # Tables to always keep in cache
            "warm_tables": []  # Tables to preload on startup
        }
    
    def preload_hot_tables(self):
        """Preload frequently accessed tables to Redis"""
        if not self.preload_config["hot_tables"]:
            logger.info("No hot tables configured for preloading")
            return
        
        logger.info(f"Preloading {len(self.preload_config['hot_tables'])} hot tables...")
        results = self.db_manager.preload_tables_to_cache(
            self.preload_config["hot_tables"]
        )
        
        success = sum(1 for v in results.values() if v)
        logger.info(f"Preloaded {success}/{len(results)} hot tables")
        return results
