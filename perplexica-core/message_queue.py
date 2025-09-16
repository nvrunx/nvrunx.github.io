"""
Message Queue Infrastructure for PerplexicaCore
===============================================

This module sets up the Celery application with RabbitMQ as message broker and Redis
for both result backend and caching. The architecture is designed for horizontal
scaling and fault tolerance.

Key design decisions:
1. RabbitMQ for reliable message delivery with persistence
2. Redis for fast caching and result storage  
3. JSON serialization for interoperability
4. Comprehensive error handling and retries
5. Route-based task distribution for load balancing
"""

import logging
import json
import hashlib
from typing import Any, Optional, Dict, List
from celery import Celery
import redis
from config import Config

# Configure logging for this module
logger = logging.getLogger(__name__)

# ============================================================================
# CELERY APPLICATION SETUP
# ============================================================================

def create_celery_app() -> Celery:
    """
    Creates and configures the Celery application with optimal settings for
    our distributed search engine architecture.
    
    Configuration highlights:
    - JSON serialization for security and interoperability
    - Acknowledges messages after successful execution (late ack)
    - Prefetch multiplier of 1 to ensure fair task distribution
    - Comprehensive retry and error handling policies
    
    Returns:
        Configured Celery application instance
    """
    
    # Create Celery app with descriptive name
    app = Celery('perplexica_core')
    
    # Configure Celery with production-ready settings
    app.conf.update(
        # ========================================================================
        # BROKER AND BACKEND CONFIGURATION
        # ========================================================================
        
        # RabbitMQ as message broker - reliable, persistent, supports routing
        broker_url=Config.RABBITMQ_URL,
        
        # Redis as result backend - fast storage for task results
        result_backend=Config.REDIS_URL,
        
        # ========================================================================
        # SERIALIZATION SETTINGS
        # ========================================================================
        
        # Use JSON for security - avoids pickle vulnerabilities
        # JSON is also language-agnostic for potential polyglot services
        task_serializer='json',
        accept_content=['json'],
        result_serializer='json',
        
        # ========================================================================
        # TASK EXECUTION SETTINGS
        # ========================================================================
        
        # Acknowledge tasks after execution (not before) for reliability
        # This ensures tasks are retried if worker crashes during execution
        task_acks_late=True,
        
        # Prefetch only 1 task per worker to ensure fair distribution
        # This is crucial when tasks have varying execution times
        worker_prefetch_multiplier=1,
        
        # Reject tasks on worker shutdown to avoid loss
        worker_disable_rate_limits=True,
        
        # ========================================================================
        # RESULT SETTINGS
        # ========================================================================
        
        # Store results for 1 hour to allow for debugging and result retrieval
        result_expires=3600,
        
        # Ignore results for fire-and-forget tasks where we don't need the return value
        task_ignore_result=False,
        
        # ========================================================================
        # RETRY AND ERROR HANDLING
        # ========================================================================
        
        # Default retry policy for all tasks
        task_default_retry_delay=60,  # Wait 60 seconds before retry
        task_max_retries=3,           # Maximum 3 retry attempts
        
        # ========================================================================
        # ROUTING CONFIGURATION
        # ========================================================================
        
        # Route different types of tasks to different queues for load balancing
        task_routes={
            'services.search_service.*': {'queue': 'search'},      # Search operations
            'services.scraper_service.*': {'queue': 'scraper'},    # Web scraping
            'services.embedding_service.*': {'queue': 'embedding'}, # AI processing
            'services.llm_gateway.*': {'queue': 'llm'},            # LLM operations
        },
        
        # ========================================================================
        # MONITORING AND DEBUGGING
        # ========================================================================
        
        # Send task events for monitoring tools
        worker_send_task_events=True,
        task_send_sent_event=True,
        
        # Include timezone information
        enable_utc=True,
        timezone='UTC',
    )
    
    return app


# Create the global Celery app instance
# This will be imported by all service modules
celery_app = create_celery_app()

# ============================================================================
# REDIS CACHING INFRASTRUCTURE
# ============================================================================

class CacheManager:
    """
    Redis-based caching manager for PerplexicaCore.
    
    This class provides a clean interface for caching search results and
    intermediate computations. Caching is critical for performance since
    web searches and AI processing are expensive operations.
    
    Key features:
    - Automatic key generation based on content hashing
    - TTL-based expiration for data freshness
    - Error handling for cache failures
    - JSON serialization for complex data structures
    """
    
    def __init__(self):
        """
        Initialize Redis connection with error handling.
        
        The cache is designed to be fault-tolerant - if Redis is unavailable,
        the system continues to work but without caching benefits.
        """
        try:
            # Create Redis connection with connection pooling for efficiency
            self.redis_client = redis.Redis.from_url(
                Config.REDIS_URL,
                decode_responses=True,  # Automatically decode byte responses to strings
                socket_connect_timeout=5,  # Quick timeout for connection attempts
                socket_timeout=5,          # Quick timeout for operations
                retry_on_timeout=True,     # Retry on network timeouts
                max_connections=20         # Connection pool size
            )
            
            # Test the connection
            self.redis_client.ping()
            logger.info("Redis cache connection established successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis cache: {e}")
            self.redis_client = None
    
    def _generate_cache_key(self, prefix: str, data: Any) -> str:
        """
        Generate a unique cache key based on data content.
        
        Using content hashing ensures that identical queries get the same cache key,
        regardless of when they're made. This maximizes cache hit rates.
        
        Args:
            prefix: Cache key prefix (e.g., 'search', 'embedding')
            data: Data to generate key from (will be JSON-serialized)
            
        Returns:
            Unique cache key string
        """
        # Convert data to stable JSON representation
        # sort_keys=True ensures consistent ordering for same content
        data_str = json.dumps(data, sort_keys=True, ensure_ascii=True)
        
        # Generate SHA-256 hash for uniqueness and fixed length
        hash_digest = hashlib.sha256(data_str.encode('utf-8')).hexdigest()
        
        # Return prefixed key (e.g., "search:a1b2c3d4...")
        return f"{prefix}:{hash_digest}"
    
    def get(self, prefix: str, data: Any) -> Optional[Any]:
        """
        Retrieve cached data if available.
        
        Args:
            prefix: Cache key prefix
            data: Data used to generate cache key
            
        Returns:
            Cached data if found and valid, None otherwise
        """
        if not self.redis_client:
            return None
            
        try:
            # Generate cache key from input data
            cache_key = self._generate_cache_key(prefix, data)
            
            # Attempt to retrieve cached data
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                # Deserialize JSON data
                result = json.loads(cached_data)
                logger.debug(f"Cache hit for key: {cache_key}")
                return result
            else:
                logger.debug(f"Cache miss for key: {cache_key}")
                return None
                
        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
            return None
    
    def set(self, prefix: str, data: Any, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Store data in cache with optional TTL.
        
        Args:
            prefix: Cache key prefix  
            data: Data used to generate cache key
            value: Data to cache
            ttl: Time-to-live in seconds (defaults to Config.CACHE_TTL)
            
        Returns:
            True if cached successfully, False otherwise
        """
        if not self.redis_client:
            return False
            
        try:
            # Generate cache key and serialize value
            cache_key = self._generate_cache_key(prefix, data)
            serialized_value = json.dumps(value, ensure_ascii=True)
            
            # Use configured TTL if not specified
            expiration = ttl or Config.CACHE_TTL
            
            # Store in Redis with expiration
            success = self.redis_client.setex(
                cache_key, 
                expiration, 
                serialized_value
            )
            
            if success:
                logger.debug(f"Cached data with key: {cache_key}, TTL: {expiration}s")
            else:
                logger.warning(f"Failed to cache data with key: {cache_key}")
                
            return bool(success)
            
        except Exception as e:
            logger.error(f"Cache storage error: {e}")
            return False
    
    def delete(self, prefix: str, data: Any) -> bool:
        """
        Remove data from cache.
        
        Args:
            prefix: Cache key prefix
            data: Data used to generate cache key
            
        Returns:
            True if deleted successfully, False otherwise
        """
        if not self.redis_client:
            return False
            
        try:
            cache_key = self._generate_cache_key(prefix, data)
            deleted = self.redis_client.delete(cache_key)
            
            if deleted:
                logger.debug(f"Deleted cache key: {cache_key}")
            
            return bool(deleted)
            
        except Exception as e:
            logger.error(f"Cache deletion error: {e}")
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """
        Clear multiple cache entries matching a pattern.
        
        Useful for cache invalidation of related entries (e.g., all search results).
        
        Args:
            pattern: Redis key pattern (e.g., "search:*")
            
        Returns:
            Number of keys deleted
        """
        if not self.redis_client:
            return 0
            
        try:
            # Find all keys matching the pattern
            keys = self.redis_client.keys(pattern)
            
            if keys:
                # Delete all matching keys
                deleted = self.redis_client.delete(*keys)
                logger.info(f"Cleared {deleted} cache entries matching pattern: {pattern}")
                return deleted
            else:
                logger.debug(f"No cache entries found for pattern: {pattern}")
                return 0
                
        except Exception as e:
            logger.error(f"Cache pattern clearing error: {e}")
            return 0


# Create global cache manager instance
# This will be imported by all service modules that need caching
cache_manager = CacheManager()

# ============================================================================
# CELERY TASK DECORATORS AND UTILITIES
# ============================================================================

def cached_task(cache_prefix: str, cache_ttl: Optional[int] = None):
    """
    Decorator to add caching to Celery tasks.
    
    This decorator automatically caches task results based on input parameters,
    significantly improving performance for repeated operations.
    
    Args:
        cache_prefix: Prefix for cache keys (e.g., 'search', 'embedding')
        cache_ttl: Cache TTL in seconds (uses Config.CACHE_TTL if None)
        
    Returns:
        Decorator function
    """
    def decorator(task_func):
        def wrapper(*args, **kwargs):
            # Create cache key from task arguments
            cache_data = {'args': args, 'kwargs': kwargs}
            
            # Try to get result from cache first
            cached_result = cache_manager.get(cache_prefix, cache_data)
            if cached_result is not None:
                logger.info(f"Task {task_func.__name__} served from cache")
                return cached_result
            
            # Execute the task if not cached
            result = task_func(*args, **kwargs)
            
            # Cache the result for future use
            cache_manager.set(cache_prefix, cache_data, result, cache_ttl)
            
            return result
        
        return wrapper
    return decorator


# ============================================================================
# HEALTH CHECK AND MONITORING
# ============================================================================

def get_system_health() -> Dict[str, Any]:
    """
    Check the health of all system components.
    
    This function verifies connectivity to RabbitMQ, Redis, and other
    critical services. It's useful for monitoring and debugging.
    
    Returns:
        Dictionary containing health status of all components
    """
    health = {
        'timestamp': None,
        'celery': False,
        'redis': False,
        'cache_stats': {},
    }
    
    # Add timestamp
    from datetime import datetime
    health['timestamp'] = datetime.utcnow().isoformat()
    
    # Check Celery/RabbitMQ connection
    try:
        # Test if we can inspect the Celery app
        inspect = celery_app.control.inspect()
        stats = inspect.stats()
        health['celery'] = stats is not None
        logger.debug("Celery/RabbitMQ health check passed")
    except Exception as e:
        logger.error(f"Celery/RabbitMQ health check failed: {e}")
        health['celery'] = False
    
    # Check Redis connection
    try:
        if cache_manager.redis_client:
            cache_manager.redis_client.ping()
            health['redis'] = True
            
            # Get cache statistics
            info = cache_manager.redis_client.info()
            health['cache_stats'] = {
                'connected_clients': info.get('connected_clients', 0),
                'used_memory_human': info.get('used_memory_human', 'unknown'),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
            }
            logger.debug("Redis health check passed")
        else:
            health['redis'] = False
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        health['redis'] = False
    
    return health


# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

# Log system initialization
logger.info("Message queue infrastructure initialized")
logger.info(f"Celery broker: {Config.RABBITMQ_URL}")
logger.info(f"Redis backend: {Config.REDIS_URL}")

# Perform initial health check
try:
    health_status = get_system_health()
    logger.info(f"System health check: {health_status}")
except Exception as e:
    logger.error(f"Initial health check failed: {e}")