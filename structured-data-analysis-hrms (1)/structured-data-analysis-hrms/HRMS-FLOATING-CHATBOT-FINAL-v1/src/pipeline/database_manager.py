import os
import redis
import pandas as pd
import pickle
from typing import List, Dict, Any, Optional, Tuple
import logging 
from datetime import datetime
from src.config.configuration import ConfigurationManager
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool

logger = logging.getLogger(__name__)

class Config:
    def __init__(self):
        self.config_obj = ConfigurationManager()
        self.config = self.config_obj.get_base_config()
        self.POSTGRES_USER = os.environ.get("POSTGRES_USER", self.config.POSTGRES_USER)
        self.POSTGRES_HOST = os.environ.get("POSTGRES_HOST", self.config.POSTGRES_HOST)
        self.POSTGRES_PORT = os.environ.get("POSTGRES_PORT", self.config.POSTGRES_PORT)
        self.POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", self.config.POSTGRES_PASSWORD)
        self.POSTGRES_DB = os.environ.get("POSTGRES_DB", self.config.POSTGRES_DB)
        
        # Redis Configuration
        self.REDIS_USERNAME = os.environ.get("REDIS_USERNAME", getattr(self.config, 'REDIS_USERNAME', 'redis'))
        self.REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", getattr(self.config, 'REDIS_PASSWORD', 'None'))
        self.REDIS_HOST = os.environ.get("REDIS_HOST", getattr(self.config, 'REDIS_HOST', 'localhost'))
        self.REDIS_PORT = int(os.environ.get("REDIS_PORT", getattr(self.config, 'REDIS_PORT', 6379)))
        self.REDIS_DB = int(os.environ.get("REDIS_DB", getattr(self.config, 'REDIS_DB', 0)))
        self.CACHE_TTL = os.environ.get("CACHE_TTL", getattr(self.config, 'CACHE_TTL', None))

class DatabaseManager:
    """Singleton DatabaseManager with connection pooling for Oracle and Redis caching."""
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self.config = Config()
        self.engine: Optional[Engine] = None
        self.connection = None
        self.is_connected = False
        self.schema = None

        # Initialize Redis connection
        try:
            self.redis_client = redis.Redis(
                host=self.config.REDIS_HOST,
                port=self.config.REDIS_PORT,
                username=self.config.REDIS_USERNAME,
                password=self.config.REDIS_PASSWORD,
                db=self.config.REDIS_DB,
                decode_responses=False,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            self.redis_client.ping()
            self.redis_enabled = True
            logging.info(f"Redis cache connected at {self.config.REDIS_HOST}:{self.config.REDIS_PORT}")
        except Exception as e:
            logging.warning(f"Redis unavailable: {e}. Using direct DB access.")
            self.redis_enabled = False
            self.redis_client = None
        
        # Cache expiration time
        self.cache_ttl = self.config.CACHE_TTL
        
        # Query result cache (in-memory for fast repeated queries)
        self._query_cache = {}
        self._query_cache_max_size = 100
        self._initialized = True
    
    def _get_cache_key(self, table_name: str) -> str:
        """Generate unique cache key for a table"""
        return f"postgres_table:{self.config.POSTGRES_DB}:{table_name}"
    
    def _get_metadata_key(self, table_name: str) -> str:
        """Generate cache key for metadata timestamp"""
        return f"postgres_meta:{self.config.POSTGRES_DB}:{table_name}"
    
    def _check_cache_exists(self, table_name: str) -> bool:
        """Check if data exists in Redis cache"""
        if not self.redis_enabled:
            return False
        
        try:
            cache_key = self._get_cache_key(table_name)
            return self.redis_client.exists(cache_key) > 0
        except Exception as e:
            logging.error(f"Error checking cache for '{table_name}': {e}")
            return False
    
    def _get_from_cache(self, table_name: str) -> Optional[pd.DataFrame]:
        """Retrieve DataFrame from Redis cache"""
        if not self.redis_enabled:
            return None
        
        try:
            cache_key = self._get_cache_key(table_name)
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                df = pickle.loads(cached_data)
                meta_key = self._get_metadata_key(table_name)
                cached_time = self.redis_client.get(meta_key)
                if cached_time:
                    cached_time = cached_time.decode('utf-8')
                    logging.info(f"Cache HIT: '{table_name}' (cached at {cached_time})")
                else:
                    logging.info(f"Cache HIT: '{table_name}'")
                return df
            else:
                logging.info(f"Cache MISS: '{table_name}'")
                return None
        except Exception as e:
            logging.error(f"Error reading cache for '{table_name}': {e}")
            return None
    
    def _set_to_cache(self, table_name: str, df: pd.DataFrame) -> bool:
        """Store DataFrame in Redis cache"""
        if not self.redis_enabled:
            return False
        
        try:
            cache_key = self._get_cache_key(table_name)
            pickled_df = pickle.dumps(df)
            
            # Store data with optional TTL
            if self.cache_ttl:
                self.redis_client.setex(cache_key, int(self.cache_ttl), pickled_df)
            else:
                self.redis_client.set(cache_key, pickled_df)
            
            # Store metadata timestamp
            meta_key = self._get_metadata_key(table_name)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if self.cache_ttl:
                self.redis_client.setex(meta_key, int(self.cache_ttl), timestamp)
            else:
                self.redis_client.set(meta_key, timestamp)
            
            ttl_msg = f"for {self.cache_ttl}s" if self.cache_ttl else "permanently"
            logging.info(f"Cached '{table_name}' {ttl_msg} ({len(df)} rows, {len(df.columns)} cols)")
            return True
        except Exception as e:
            logging.error(f"Error writing cache for '{table_name}': {e}")
            return False
    
    def connect(self) -> bool:
        try:
            if not all([self.config.POSTGRES_USER, self.config.POSTGRES_PASSWORD, 
                       self.config.POSTGRES_HOST, self.config.POSTGRES_PORT, self.config.POSTGRES_DB]):
                logging.error("Database credentials missing")
                self.is_connected = False
                return False
            
            from urllib.parse import quote_plus
            encoded_password = quote_plus(self.config.POSTGRES_PASSWORD)
            engine_url = (
                f"postgresql+psycopg2://{self.config.POSTGRES_USER}:{encoded_password}@"
                f"{self.config.POSTGRES_HOST}:{self.config.POSTGRES_PORT}/{self.config.POSTGRES_DB}"
            )
            # Create the SQLAlchemy engine with connection pooling
            self.engine = create_engine(
                engine_url,
                poolclass=QueuePool,
                pool_size=20,
                max_overflow=40,
                pool_pre_ping=True,
                pool_recycle=1800,
                pool_timeout=30,
                echo_pool=False
            )
            
            # Test connection
            with self.engine.connect() as conn:
                pass

            self.is_connected = True
            self.schema = 'hrms'
            logging.info(f"Connected to PostgreSQL '{self.config.POSTGRES_DB}' as '{self.config.POSTGRES_USER}'")
            return True
    
        except Exception as e:
            logging.error(f"Failed to connect to PostgreSQL: {e}")
            self.is_connected = False
            return False
    
    def dispose_pool(self) -> None:
        """Dispose connection pool (call at app shutdown)"""
        try:
            if self.engine:
                self.engine.dispose()
                logging.info("Connection pool disposed")
        except Exception as e:
            logging.error(f"Error disposing pool: {e}")

    def get_sqlalchemy_engine(self) -> Optional[Engine]:
        """Returns the active SQLAlchemy engine."""
        if not self.is_connected or not self.engine:
            logging.error("No active database engine. Call connect() first.")
            return None
        return self.engine
    
    def clear_table_names_cache(self):
        """Clear cached table names to force fresh lookup"""
        if not self.redis_enabled:
            logging.info("Redis not enabled, no cache to clear")
            return
        
        try:
            cache_key = f"table_names:{self.config.POSTGRES_DB}"
            deleted = self.redis_client.delete(cache_key)
            if deleted:
                logging.info("Cleared table names cache")
            else:
                logging.info("No table names cache found to clear")
        except Exception as e:
            logging.error(f"Error clearing table names cache: {e}")

    def get_table_names(self) -> List[str]:
        """Get all table names from schema with caching"""
        if not self.engine:
            logging.error("No active database engine")
            return []
        
        # Check cache first
        cache_key = f"table_names:{self.config.POSTGRES_DB}"
        if self.redis_enabled:
            try:
                cached = self.redis_client.get(cache_key)
                if cached:
                    table_names = pickle.loads(cached)
                    logging.info(f"Table names from cache: {len(table_names)} tables")
                    # If cache has 0 tables, clear it and fetch from DB
                    if len(table_names) == 0:
                        logging.warning("Cache contains 0 tables, clearing cache and fetching from DB")
                        self.redis_client.delete(cache_key)
                    else:
                        return table_names
                else:
                    logging.info("No cached table names found, fetching from database")
            except Exception as e:
                logging.warning(f"Cache read failed: {e}")
        
        try:
            query = (
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'hrms' AND table_type = 'BASE TABLE'"
            )
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                table_names = [row[0].lower() for row in result.fetchall()]
                logging.info(f"Retrieved {len(table_names)} table names from DB: {table_names}")
            
            # Cache for 1 hour
            if self.redis_enabled:
                try:
                    self.redis_client.setex(cache_key, 3600, pickle.dumps(table_names))
                except Exception as e:
                    logging.warning(f"Cache write failed: {e}")
            
            return table_names

        except Exception as e:
            logging.error(f"Error retrieving table names: {e}")
            return []


    def get_table_metadata(self, table_name: str) -> Dict[str, Any]:
        """Get metadata for a specific table"""
        metadata = {
            'table_name': table_name,
            'columns': [],
            'data_types': [],
            'row_count': 0,
            'sample_data': {}
        }
        if not self.engine:
            logging.error("No active database connection")
            return metadata

        # Parameters to be used for all queries
        params = {
            'table_name': table_name.lower(),
            'schema': 'hrms'
        }
        
        try:
            with self.engine.connect() as conn:
                # 1. Get columns and data types
                col_query = text("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = :table_name AND table_schema = :schema 
                    ORDER BY ordinal_position
                """)
                result1 = conn.execute(col_query, params)
                columns_info = result1.fetchall()
                
                metadata['columns'] = [col[0] for col in columns_info]
                metadata['data_types'] = [col[1] for col in columns_info]

                # 2. Get row count
                count_query = text(f"SELECT COUNT(*) FROM {params['schema']}.{params['table_name']}")
                result2 = conn.execute(count_query)
                row_count = result2.fetchone()
                metadata['row_count'] = row_count[0] if row_count and row_count[0] is not None else 0

                # 3. Get sample data
                if metadata['columns']:
                    sample_query = text(f"SELECT * FROM {params['schema']}.{params['table_name']} LIMIT 1")
                    result3 = conn.execute(sample_query) 
                    sample_row = result3.fetchone()
                    
                    if sample_row:
                        metadata['sample_data'] = dict(zip(metadata['columns'], sample_row))
            
            return metadata
    
        except Exception as e:
            # Note: If you want to use the ALL_TABLES query for row count, it must also use text():
            # count_query = text("SELECT num_rows FROM all_tables WHERE table_name = :table_name AND owner = :schema")
            logging.error(f"Error getting metadata for '{table_name}': {e}")
            return metadata
    
    def execute_query_cached(self, query: str, ttl: int = 300) -> Optional[pd.DataFrame]:
        """Execute SQL query with result caching (5 min default TTL)"""
        if not self.redis_enabled:
            return pd.read_sql(query, self.engine)
        
        try:
            cache_key = f"query_result:{hash(query)}"
            cached_result = self.redis_client.get(cache_key)
            
            if cached_result:
                logger.info(f"Query cache HIT")
                return pickle.loads(cached_result)
            
            logger.info(f"Query cache MISS - executing query")
            df = pd.read_sql(query, self.engine)
            
            # Cache result
            pickled_df = pickle.dumps(df)
            self.redis_client.setex(cache_key, ttl, pickled_df)
            
            return df
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            return None
    
    def load_table_data(self, table_name: str) -> Optional[pd.DataFrame]:
        """
        Load table data - ALWAYS tries Redis cache first, then falls back to database.
        Automatically caches data after loading from database.
        This ensures FAST responses for chatbot queries.
        """
        # STEP 1: Try Redis cache first (FAST PATH)
        if self.redis_enabled:
            cached_df = self._get_from_cache(table_name)
            if cached_df is not None:
                return cached_df
        
        # STEP 2: Cache miss - load from database (SLOW PATH)
        if not self.engine:
            logging.error("No active database engine")
            return None
        
        try:
            # Security: validate table name (allow alphanumeric and underscores)
            if not table_name.replace('_', '').isalnum():
                logging.error(f"Invalid table name: {table_name}")
                return None
            
            logging.info(f"Loading '{table_name}' from database...")
            query = f"SELECT * FROM hrms.{table_name}"
            df = pd.read_sql(query, self.engine)
            
            logging.info(f"Loaded '{table_name}' from database: {len(df)} rows")
            
            # STEP 3: Store in Redis for next time (automatic caching)
            if self.redis_enabled:
                self._set_to_cache(table_name, df)
            
            return df
        except Exception as e:
            logging.error(f"Error loading '{table_name}': {e}")
            return None

    def load_multiple_tables(self, table_names: List[str]) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
        """
        Load multiple tables - tries Redis first for each table.
        Perfect for chatbot queries that need multiple tables.
        Tracks cache hits vs database loads for monitoring.
        """
        loaded_data = {}
        messages = []
        
        if not self.is_connected or not self.connection:
            messages.append("No database connection available")
            return loaded_data, messages
        
        messages.append(f"Connected to PostgreSQL as '{self.config.POSTGRES_USER}'")
        
        if self.redis_enabled:
            ttl_display = self.cache_ttl if self.cache_ttl else 'permanent'
            messages.append(f"Redis cache enabled (TTL: {ttl_display})")
        else:
            messages.append("Redis unavailable - using direct DB access")
        
        cache_hits = 0
        db_loads = 0
        
        for table_name in table_names:
            # Check if already cached
            from_cache = self._check_cache_exists(table_name)
            
            # Load table (will use cache if available)
            df = self.load_table_data(table_name)
            
            if df is not None:
                loaded_data[table_name] = df
                if from_cache:
                    cache_hits += 1
                    source = "Redis"
                else:
                    db_loads += 1
                    source = "Database"
                messages.append(f"'{table_name}': {len(df)} rows, {len(df.columns)} cols ({source})")
            else:
                messages.append(f"Failed to load '{table_name}'")
        
        # Summary for monitoring
        if self.redis_enabled and table_names:
            messages.append(f"\nSummary: {cache_hits} from cache, {db_loads} from database")
        
        return loaded_data, messages
    
    def preload_tables_to_cache(self, table_names: List[str]) -> Dict[str, bool]:
        """
        Pre-populate Redis cache with tables from database.
        Call this on application startup to ensure first chatbot query is FAST.
        
        Args:
            table_names: List of table names to preload
            
        Returns:
            Dict mapping table names to success status
        """
        results = {}
        
        if not self.redis_enabled:
            logging.warning("Redis not enabled. Cannot preload cache.")
            return results
        
        if not self.is_connected or not self.connection:
            logging.error("No database connection. Cannot preload cache.")
            return results
        
        logging.info(f"Preloading {len(table_names)} tables to Redis cache...")
        
        for table_name in table_names:
            try:
                # Check if already cached
                if self._check_cache_exists(table_name):
                    logging.info(f"✓ '{table_name}' already cached, skipping")
                    results[table_name] = True
                    continue
                
                # Validate table name
                if not table_name.replace('_', '').isalnum():
                    logging.error(f"Invalid table name: {table_name}")
                    results[table_name] = False
                    continue
                
                # Load from database
                query = f"SELECT * FROM hrms.{table_name}"
                df = pd.read_sql(query, self.engine)
                
                # Cache it
                success = self._set_to_cache(table_name, df)
                results[table_name] = success
                
                if success:
                    logging.info(f"Preloaded '{table_name}': {len(df)} rows")
                else:
                    logging.error(f"Failed to cache '{table_name}'")
                    
            except Exception as e:
                logging.error(f"Error preloading '{table_name}': {e}")
                results[table_name] = False
        
        success_count = sum(1 for v in results.values() if v)
        logging.info(f"Preload complete: {success_count}/{len(table_names)} tables cached")
        
        return results
    
    def refresh_cache(self, table_names: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Refresh cache by reloading data from database.
        Use when you know data has been updated.
        
        Args:
            table_names: Tables to refresh, or None for all cached tables
        """
        if not self.redis_enabled:
            logging.warning("⚠ Redis not enabled")
            return {}
        
        if table_names is None:
            # Get all cached tables
            pattern = f"postgres_table:{self.config.POSTGRES_DB}:*"
            keys = self.redis_client.keys(pattern)
            table_names = [k.decode('utf-8').split(':')[-1] for k in keys]
        
        if not table_names:
            logging.info("No tables to refresh")
            return {}
        
        logging.info(f"Refreshing cache for {len(table_names)} tables...")
        
        # Delete existing cache
        for table_name in table_names:
            cache_key = self._get_cache_key(table_name)
            meta_key = self._get_metadata_key(table_name)
            self.redis_client.delete(cache_key, meta_key)
        
        # Reload from database
        return self.preload_tables_to_cache(table_names)
    
    def invalidate_cache(self, table_name: Optional[str] = None):
        """
        Invalidate cache for specific table or all tables.
        Use when data is updated in the database.
        """
        if not self.redis_enabled:
            logging.warning("Redis not enabled")
            return
        
        try:
            if table_name:
                cache_key = self._get_cache_key(table_name)
                meta_key = self._get_metadata_key(table_name)
                self.redis_client.delete(cache_key, meta_key)
                logging.info(f"Invalidated cache for '{table_name}'")
            else:
                pattern = f"postgres_table:{self.config.POSTGRES_DB}:*"
                keys = self.redis_client.keys(pattern)
                meta_pattern = f"postgres_meta:{self.config.POSTGRES_DB}:*"
                meta_keys = self.redis_client.keys(meta_pattern)
                all_keys = keys + meta_keys
                if all_keys:
                    self.redis_client.delete(*all_keys)
                    logging.info(f"Invalidated cache for {len(keys)} tables")
        except Exception as e:
            logging.error(f"Error invalidating cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about cached tables - useful for monitoring"""
        if not self.redis_enabled:
            return {"status": "Redis not enabled"}
        
        try:
            pattern = f"postgres_table:{self.config.POSTGRES_DB}:*"
            keys = self.redis_client.keys(pattern)
            
            stats = {
                "status": "Redis enabled",
                "redis_host": f"{self.config.REDIS_HOST}:{self.config.REDIS_PORT}",
                "total_cached_tables": len(keys),
                "cache_ttl": self.cache_ttl if self.cache_ttl else "permanent",
                "tables": []
            }
            
            for key in keys:
                table_name = key.decode('utf-8').split(':')[-1]
                meta_key = self._get_metadata_key(table_name)
                cached_time = self.redis_client.get(meta_key)
                
                stats["tables"].append({
                    "name": table_name,
                    "cached_at": cached_time.decode('utf-8') if cached_time else "Unknown"
                })
            
            return stats
        except Exception as e:
            logging.error(f"Error getting cache stats: {e}")
            return {"status": "Error", "error": str(e)}
    
    def get_all_table_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all tables in schema"""
        table_names = self.get_table_names()
        metadata_dict = {}
        for table_name in table_names:
            metadata_dict[table_name] = self.get_table_metadata(table_name)
        return metadata_dict