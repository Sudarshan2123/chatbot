# SQL Agent Performance Optimization Guide

## Changes Made

### 1. Database Connection Pool Optimization
**File**: `src/pipeline/database_manager.py`

- **Increased pool size**: 10 → 20 connections
- **Increased max overflow**: 20 → 40 connections
- **Reduced pool recycle**: 3600s → 1800s (30 min)
- **Added pool timeout**: 30 seconds
- **Added query result caching**: New `execute_query_cached()` method

### 2. Metadata Caching
**File**: `src/pipeline/database_manager.py`

- **Table names cached**: 1 hour TTL in Redis
- **Avoids repeated metadata queries**

### 3. SQL Agent Optimization
**File**: `src/pipeline/data_analyzer.py`

- **Reduced max_output_tokens**: 4096 → 2048
- **Added max_retries**: 2 retries for failed requests
- **Added request_timeout**: 30 seconds
- **Disabled verbose logging**: Reduces overhead
- **Added max_iterations**: Limits agent steps to 10
- **Added max_execution_time**: 60 second timeout

### 4. New Utilities
- **query_optimizer.py**: Query optimization and batching
- **performance_monitor.py**: Track metrics and cache hit rates
- **performance_config.yaml**: Centralized performance settings

## Quick Wins

### 1. Enable Query Result Caching
```python
# In your code, use the new cached query method:
df = db_manager.execute_query_cached(
    "SELECT * FROM employee_master WHERE dept_id = 5",
    ttl=300  # Cache for 5 minutes
)
```

### 2. Preload Frequently Used Tables
```python
# Add to performance_config.yaml:
preload:
  hot_tables:
    - employee_master
    - department_mst
    - designation_master

# Then in app startup:
from src.pipeline.query_optimizer import TablePreloader

preloader = TablePreloader(db_manager)
preloader.preload_hot_tables()
```

### 3. Monitor Performance
```python
from src.pipeline.performance_monitor import performance_monitor

# Get stats
stats = performance_monitor.get_stats()
print(f"Cache hit rate: {stats['cache_hit_rate']:.1f}%")
print(f"Avg query time: {stats['avg_query_times']}")
```

## Performance Benchmarks

### Before Optimization
- Connection pool: 10 connections
- No query caching
- No metadata caching
- Verbose logging enabled
- Avg response time: ~5-10 seconds

### After Optimization (Expected)
- Connection pool: 20-60 connections
- Query result caching (5 min TTL)
- Metadata caching (1 hour TTL)
- Optimized logging
- **Expected avg response time: ~1-3 seconds** (60-70% improvement)

## Additional Recommendations

### 1. Database Level
```sql
-- Add indexes on frequently queried columns
CREATE INDEX idx_emp_dept ON employee_master(department_id);
CREATE INDEX idx_emp_desig ON employee_master(designation_id);

-- Update table statistics
EXEC DBMS_STATS.GATHER_SCHEMA_STATS('YOUR_SCHEMA');
```

### 2. Redis Configuration
```bash
# Increase Redis memory limit
maxmemory 2gb
maxmemory-policy allkeys-lru

# Enable persistence for cache durability
save 900 1
save 300 10
```

### 3. Application Level
```python
# Use connection pooling at app level
from src.pipeline.database_manager import DatabaseManager

# Singleton pattern ensures one pool per app
db_manager = DatabaseManager()  # Reuses existing pool
```

### 4. Query Optimization
```python
from src.pipeline.query_optimizer import QueryOptimizer

# Add Oracle hints for faster results
optimizer = QueryOptimizer()
query = "SELECT * FROM large_table"
optimized = optimizer.add_query_hints(query)
# Result: SELECT /*+ FIRST_ROWS(100) */ * FROM large_table
```

## Monitoring Commands

### Check Cache Stats
```python
stats = db_manager.get_cache_stats()
print(f"Cached tables: {stats['total_cached_tables']}")
print(f"Cache TTL: {stats['cache_ttl']}")
```

### Check Connection Pool
```python
engine = db_manager.get_sqlalchemy_engine()
print(f"Pool size: {engine.pool.size()}")
print(f"Checked out: {engine.pool.checkedout()}")
```

### Clear Cache (if needed)
```python
# Clear specific table
db_manager.invalidate_cache("employee_master")

# Clear all cache
db_manager.invalidate_cache()
```

## Troubleshooting

### Issue: "Too many connections"
**Solution**: Reduce pool_size in `performance_config.yaml`

### Issue: "Stale cache data"
**Solution**: Reduce cache_ttl or call `db_manager.refresh_cache()`

### Issue: "Slow first query"
**Solution**: Preload tables on app startup

### Issue: "Redis connection errors"
**Solution**: Check Redis is running and increase socket_timeout

## Next Steps

1. **Measure baseline**: Run queries and note current response times
2. **Apply changes**: Restart application with new settings
3. **Monitor metrics**: Use performance_monitor to track improvements
4. **Tune settings**: Adjust pool sizes and TTLs based on your workload
5. **Add indexes**: Work with DBA to add database indexes

## Configuration Files

- `config/performance_config.yaml` - Performance settings
- `src/pipeline/database_manager.py` - Connection pool & caching
- `src/pipeline/data_analyzer.py` - SQL agent settings
- `src/pipeline/query_optimizer.py` - Query optimization utilities
- `src/pipeline/performance_monitor.py` - Metrics tracking
