import os
import redis
import cx_Oracle
import pandas as pd
import pickle
from typing import List, Dict, Any, Optional, Tuple
import logging 
from datetime import datetime
from src.config.configuration import ConfigurationManager

class Config:
    def __init__(self):
        self.config_obj = ConfigurationManager()
        self.config = self.config_obj.get_base_config()
        self.ORACLE_USER = os.environ.get("ORACLE_USER", self.config.ORACLE_USER)
        self.ORACLE_HOST = os.environ.get("ORACLE_HOST", self.config.ORACLE_HOST)
        self.ORACLE_PORT = os.environ.get("ORACLE_PORT", self.config.ORACLE_PORT)
        self.ORACLE_PASSWORD = os.environ.get("ORACLE_PASSWORD", self.config.ORACLE_PASSWORD)
        self.ORACLE_DBNAME = os.environ.get("ORACLE_DBNAME", self.config.ORACLE_DBNAME)
        self.ORACLE_SCHEMA = os.environ.get("ORACLE_SCHEMA", self.config.ORACLE_SCHEMA)
        
        # Redis Configuration
        self.REDIS_USERNAME = os.environ.get("REDIS_USERNAME", getattr(self.config, 'REDIS_USERNAME', 'redis'))
        self.REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", getattr(self.config, 'REDIS_PASSWORD', 'None'))
        self.REDIS_HOST = os.environ.get("REDIS_HOST", getattr(self.config, 'REDIS_HOST', 'localhost'))
        self.REDIS_PORT = int(os.environ.get("REDIS_PORT", getattr(self.config, 'REDIS_PORT', 6379)))
        self.REDIS_DB = int(os.environ.get("REDIS_DB", getattr(self.config, 'REDIS_DB', 0)))
        self.CACHE_TTL = os.environ.get("CACHE_TTL", getattr(self.config, 'CACHE_TTL', None))

class DatabaseManager:
    """Handles all database operations using cx_Oracle with Redis caching for faster chatbot responses."""
    
    def __init__(self):
        self.config = Config()
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
    
    def _get_cache_key(self, table_name: str) -> str:
        """Generate unique cache key for a table"""
        return f"oracle_table:{self.config.ORACLE_SCHEMA.upper()}:{table_name}"
    
    def _get_metadata_key(self, table_name: str) -> str:
        """Generate cache key for metadata timestamp"""
        return f"oracle_meta:{self.config.ORACLE_SCHEMA.upper()}:{table_name}"
    
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
            if not all([self.config.ORACLE_USER, self.config.ORACLE_PASSWORD, 
                       self.config.ORACLE_HOST, self.config.ORACLE_PORT, self.config.ORACLE_DBNAME]):
                logging.error("Database credentials missing")
                self.is_connected = False
                return False
            
            dsn = cx_Oracle.makedsn(
                host=self.config.ORACLE_HOST,
                port=self.config.ORACLE_PORT,
                service_name=self.config.ORACLE_DBNAME
            )

            self.connection = cx_Oracle.connect(
                user=self.config.ORACLE_USER,
                password=self.config.ORACLE_PASSWORD,
                dsn=dsn
            )
            
            self.is_connected = True
            self.schema = self.config.ORACLE_SCHEMA.upper()
            logging.info(f"Connected to Oracle '{self.config.ORACLE_DBNAME}' as '{self.config.ORACLE_USER}'")
            return True
    
        except cx_Oracle.Error as e:
            logging.error(f"Failed to connect to Oracle: {e}")
            self.is_connected = False
            return False
    
    def disconnect(self) -> None:
        """Safely closes database connection"""
        if self.connection:
            try:
                self.connection.close()
                self.connection = None
                self.is_connected = False
                logging.info("Database connection closed")
            except cx_Oracle.Error as e:
                logging.error(f"Error closing connection: {e}")
    
    def get_table_names(self) -> List[str]:
        """Get all table names from schema"""
        table_names = []
        if not self.is_connected or not self.connection:
            logging.error("No active database connection")
            return table_names
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT table_name FROM all_tables WHERE owner = :schema", 
                          schema=self.config.ORACLE_SCHEMA.upper())
            table_names = [row[0] for row in cursor.fetchall()]
            cursor.close()
        except cx_Oracle.Error as e:
            logging.error(f"Error retrieving table names: {e}")
        return table_names

    def get_table_metadata(self, table_name: str) -> Dict[str, Any]:
        """Get metadata for a specific table"""
        metadata = {
            'table_name': table_name,
            'columns': [],
            'data_types': [],
            'row_count': 0,
            'sample_data': {}
        }
        if not self.is_connected or not self.connection:
            logging.error("No active database connection")
            return metadata

        try:
            # Validate table name (alphanumeric and underscores only)
            if not table_name.replace('_', '').isalnum():
                logging.error(f"Invalid table name: {table_name}")
                return metadata
            
            # Get columns and data types using bind variables
            with self.connection.cursor() as cursor1:
                cursor1.execute("""
                    SELECT column_name, data_type 
                    FROM all_tab_columns 
                    WHERE table_name = :table_name AND owner = :schema 
                    ORDER BY column_id
                """, {'table_name': table_name.upper(), 'schema': self.config.ORACLE_SCHEMA.upper()})
                columns_info = cursor1.fetchall()
                metadata['columns'] = [col[0] for col in columns_info]
                metadata['data_types'] = [col[1] for col in columns_info]

            # Get row count using bind variables
            with self.connection.cursor() as cursor2:
                cursor2.execute("""
                    SELECT num_rows FROM all_tables 
                    WHERE table_name = :table_name AND owner = :schema
                """, {'table_name': table_name.upper(), 'schema': self.config.ORACLE_SCHEMA.upper()})
                row_count = cursor2.fetchone()
                metadata['row_count'] = row_count[0] if row_count and row_count[0] is not None else 0

            # Get sample data using bind variables for the WHERE clause
            # and validated identifiers for table name
            if metadata['columns']:
                with self.connection.cursor() as cursor3:
                    # Verify table exists in the schema first (security validation)
                    cursor3.execute("""
                        SELECT table_name 
                        FROM all_tables 
                        WHERE table_name = :table_name AND owner = :schema
                    """, {'table_name': table_name.upper(), 'schema': self.config.ORACLE_SCHEMA.upper()})
                    
                    validated_table = cursor3.fetchone()
                    
                    if validated_table:
                        # Table exists and name is validated from database
                        validated_table_name = validated_table[0]
                        validated_schema = self.config.ORACLE_SCHEMA.upper()
                        
                        # Use validated identifiers with proper quoting to prevent injection
                        # The ROWNUM value uses a bind variable
                        query = f'SELECT * FROM "{validated_schema}"."{validated_table_name}" WHERE ROWNUM <= :max_rows'
                        cursor3.execute(query, {'max_rows': 1})
                        sample_row = cursor3.fetchone()
                        
                        if sample_row:
                            metadata['sample_data'] = dict(zip(metadata['columns'], sample_row))
                    else:
                        logging.warning(f"Table '{table_name}' not found in schema '{self.config.ORACLE_SCHEMA}'")
            
            return metadata
        
        except cx_Oracle.Error as e:
            logging.error(f"Error getting metadata for '{table_name}': {e}")
            return metadata

    def load_table_data(self, table_name: str) -> Optional[pd.DataFrame]:
        """
        Load table data - ALWAYS tries Redis cache first, then falls back to database.
        Automatically caches data after loading from database with chunking support.
        This ensures FAST responses for chatbot queries.
        """
        # STEP 1: Try Redis cache first (FAST PATH)
        if self.redis_enabled:
            cached_df = self._get_from_cache(table_name)
            if cached_df is not None:
                return cached_df
        
        # STEP 2: Cache miss - load from database (SLOW PATH)
        if not self.is_connected or not self.connection:
            logging.error("No active database connection")
            return None
        
        try:
            # Security: validate table name (allow alphanumeric and underscores)
            if not table_name.replace('_', '').isalnum():
                logging.error(f"Invalid table name: {table_name}")
                return None
            
            logging.info(f"Loading '{table_name}' from database...")
            
            # Verify table exists using bind variables (prevents SQL injection)
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT table_name 
                FROM all_tables 
                WHERE table_name = :table_name AND owner = :schema
            """, {'table_name': table_name.upper(), 'schema': self.config.ORACLE_SCHEMA.upper()})
            
            validated_table = cursor.fetchone()
            cursor.close()
            
            if not validated_table:
                logging.error(f"Table '{table_name}' not found in schema '{self.config.ORACLE_SCHEMA}'")
                return None
            
            # Use validated table name with proper quoting
            validated_table_name = validated_table[0]
            validated_schema = self.config.ORACLE_SCHEMA.upper()
            query = f'SELECT * FROM "{validated_schema}"."{validated_table_name}"'
            
            df = pd.read_sql(query, self.connection)
            logging.info(f"Loaded '{table_name}' from database: {len(df)} rows")
            
            # STEP 3: Store in Redis for next time (automatic caching with chunking)
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
        
        messages.append(f"Connected to Oracle as '{self.config.ORACLE_SCHEMA}'")
        
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
        Supports automatic chunking for large tables.
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
                    logging.info(f"'{table_name}' already cached, skipping")
                    results[table_name] = True
                    continue
                
                # Validate table name
                if not table_name.replace('_', '').isalnum():
                    logging.error(f"Invalid table name: {table_name}")
                    results[table_name] = False
                    continue
                
                # Verify table exists using bind variables
                cursor = self.connection.cursor()
                cursor.execute("""
                    SELECT table_name 
                    FROM all_tables 
                    WHERE table_name = :table_name AND owner = :schema
                """, {'table_name': table_name.upper(), 'schema': self.config.ORACLE_SCHEMA.upper()})
                
                validated_table = cursor.fetchone()
                cursor.close()
                
                if not validated_table:
                    logging.error(f"Table '{table_name}' not found")
                    results[table_name] = False
                    continue
                
                # Use validated table name
                validated_table_name = validated_table[0]
                validated_schema = self.config.ORACLE_SCHEMA.upper()
                query = f'SELECT * FROM "{validated_schema}"."{validated_table_name}"'
                
                df = pd.read_sql(query, self.connection)
                
                # Cache it (will automatically chunk if needed)
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
            pattern = f"oracle_table:{self.config.ORACLE_SCHEMA.upper()}:*"
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
                pattern = f"oracle_table:{self.config.ORACLE_SCHEMA.upper()}:*"
                keys = self.redis_client.keys(pattern)
                meta_pattern = f"oracle_meta:{self.config.ORACLE_SCHEMA.upper()}:*"
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
            pattern = f"oracle_table:{self.config.ORACLE_SCHEMA.upper()}:*"
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