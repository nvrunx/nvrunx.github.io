"""
Configuration Management for PerplexicaCore
===========================================

This module centralizes all configuration management for the PerplexicaCore search engine.
We use environment variables for all external configuration to follow 12-factor app principles,
making the application easily deployable across different environments without code changes.

The configuration is organized into logical groups (search, database, AI models, etc.) 
to make it easy to understand and maintain.
"""

import os
from typing import Optional

class Config:
    """
    Centralized configuration class that loads all environment variables
    and provides default values where appropriate.
    
    This approach allows us to:
    1. Keep all configuration in one place for easy maintenance
    2. Provide sensible defaults for development
    3. Make the system configurable for production deployments
    4. Validate critical configuration at startup
    """
    
    # ============================================================================
    # MESSAGE QUEUE CONFIGURATION (RabbitMQ + Redis)
    # ============================================================================
    
    # RabbitMQ connection URL for Celery task queue
    # Format: amqp://user:password@host:port/vhost
    RABBITMQ_URL: str = os.getenv(
        'RABBITMQ_URL', 
        'amqp://guest:guest@localhost:5672//'
    )
    
    # Redis URL for caching and Celery result backend
    # We use Redis for both caching search results and storing task results
    REDIS_URL: str = os.getenv(
        'REDIS_URL', 
        'redis://localhost:6379/0'
    )
    
    # Cache TTL in seconds - we cache common queries for 5 minutes to reduce load
    # This balances between performance and data freshness
    CACHE_TTL: int = int(os.getenv('CACHE_TTL', '300'))  # 5 minutes
    
    # ============================================================================
    # SEARCH ENGINE CONFIGURATION (SearxNG)
    # ============================================================================
    
    # SearxNG instance URL - this should point to your self-hosted SearxNG instance
    # SearxNG is a meta-search engine that aggregates results from multiple sources
    SEARXNG_URL: str = os.getenv(
        'SEARXNG_URL', 
        'http://localhost:8080'  # Default local SearxNG instance
    )
    
    # Number of search results to fetch from SearxNG
    # We fetch more results initially to have a larger pool for AI filtering
    SEARCH_RESULTS_LIMIT: int = int(os.getenv('SEARCH_RESULTS_LIMIT', '20'))
    
    # Maximum number of URLs to scrape (after initial search filtering)
    # This limits the computational cost while ensuring good coverage
    MAX_URLS_TO_SCRAPE: int = int(os.getenv('MAX_URLS_TO_SCRAPE', '10'))
    
    # ============================================================================
    # WEB SCRAPING CONFIGURATION
    # ============================================================================
    
    # User agent string for web scraping requests
    # Using a descriptive user agent helps with compliance and debugging
    USER_AGENT: str = os.getenv(
        'USER_AGENT',
        'PerplexicaCore/1.0 (AI Research Tool; +https://github.com/nvrunx)'
    )
    
    # Request timeout in seconds for web scraping
    # Longer timeout allows for slower sites but prevents hanging indefinitely
    SCRAPER_TIMEOUT: int = int(os.getenv('SCRAPER_TIMEOUT', '30'))
    
    # Delay between requests to the same domain (in seconds)
    # This implements polite crawling to avoid overloading servers
    DOMAIN_DELAY: float = float(os.getenv('DOMAIN_DELAY', '1.0'))
    
    # Maximum content length to extract (in characters)
    # This prevents memory issues with extremely large pages
    MAX_CONTENT_LENGTH: int = int(os.getenv('MAX_CONTENT_LENGTH', '50000'))
    
    # ============================================================================
    # AI MODEL CONFIGURATION
    # ============================================================================
    
    # Sentence transformer model for embedding generation
    # all-MiniLM-L6-v2 provides good quality/speed tradeoff for semantic similarity
    EMBEDDING_MODEL: str = os.getenv(
        'EMBEDDING_MODEL', 
        'all-MiniLM-L6-v2'
    )
    
    # Number of most relevant content chunks to send to LLM
    # This balances context richness with token limits and processing speed
    TOP_RELEVANT_CHUNKS: int = int(os.getenv('TOP_RELEVANT_CHUNKS', '5'))
    
    # Minimum similarity threshold for content to be considered relevant
    # Content below this threshold is filtered out to improve answer quality
    SIMILARITY_THRESHOLD: float = float(os.getenv('SIMILARITY_THRESHOLD', '0.3'))
    
    # ============================================================================
    # LLM GATEWAY CONFIGURATION
    # ============================================================================
    
    # LLM Provider selection: 'openai' or 'ollama'
    # This allows switching between cloud and local LLM deployments
    LLM_PROVIDER: str = os.getenv('LLM_PROVIDER', 'openai')
    
    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = os.getenv('OPENAI_API_KEY')
    OPENAI_MODEL: str = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
    OPENAI_MAX_TOKENS: int = int(os.getenv('OPENAI_MAX_TOKENS', '1000'))
    OPENAI_TEMPERATURE: float = float(os.getenv('OPENAI_TEMPERATURE', '0.1'))
    
    # Ollama Configuration (for local LLM deployment)
    OLLAMA_URL: str = os.getenv('OLLAMA_URL', 'http://localhost:11434')
    OLLAMA_MODEL: str = os.getenv('OLLAMA_MODEL', 'llama2')
    OLLAMA_TIMEOUT: int = int(os.getenv('OLLAMA_TIMEOUT', '120'))
    
    # ============================================================================
    # LOGGING CONFIGURATION
    # ============================================================================
    
    # Log level for the application
    # DEBUG for development, INFO for production
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    
    # Log format - structured for easy parsing in production
    LOG_FORMAT: str = os.getenv(
        'LOG_FORMAT',
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # ============================================================================
    # VALIDATION METHODS
    # ============================================================================
    
    @classmethod
    def validate_config(cls) -> None:
        """
        Validates critical configuration at startup.
        
        This method checks that all required configuration is present and valid,
        failing fast if the system cannot operate properly. This prevents
        runtime errors and makes deployment issues obvious immediately.
        """
        errors = []
        
        # Validate LLM configuration based on provider
        if cls.LLM_PROVIDER == 'openai' and not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required when LLM_PROVIDER is 'openai'")
        
        # Validate numeric configurations are positive
        if cls.SEARCH_RESULTS_LIMIT <= 0:
            errors.append("SEARCH_RESULTS_LIMIT must be positive")
            
        if cls.MAX_URLS_TO_SCRAPE <= 0:
            errors.append("MAX_URLS_TO_SCRAPE must be positive")
            
        if cls.TOP_RELEVANT_CHUNKS <= 0:
            errors.append("TOP_RELEVANT_CHUNKS must be positive")
        
        # Validate similarity threshold is between 0 and 1
        if not 0 <= cls.SIMILARITY_THRESHOLD <= 1:
            errors.append("SIMILARITY_THRESHOLD must be between 0 and 1")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    @classmethod
    def get_summary(cls) -> dict:
        """
        Returns a summary of current configuration for debugging/logging.
        
        This excludes sensitive information like API keys but shows all
        other configuration values for troubleshooting.
        """
        return {
            'rabbitmq_url': cls.RABBITMQ_URL,
            'redis_url': cls.REDIS_URL,
            'searxng_url': cls.SEARXNG_URL,
            'search_results_limit': cls.SEARCH_RESULTS_LIMIT,
            'max_urls_to_scrape': cls.MAX_URLS_TO_SCRAPE,
            'embedding_model': cls.EMBEDDING_MODEL,
            'top_relevant_chunks': cls.TOP_RELEVANT_CHUNKS,
            'similarity_threshold': cls.SIMILARITY_THRESHOLD,
            'llm_provider': cls.LLM_PROVIDER,
            'openai_model': cls.OPENAI_MODEL if cls.LLM_PROVIDER == 'openai' else None,
            'ollama_model': cls.OLLAMA_MODEL if cls.LLM_PROVIDER == 'ollama' else None,
            'log_level': cls.LOG_LEVEL,
        }


# Initialize configuration validation at module import
# This ensures the system fails fast if misconfigured
try:
    Config.validate_config()
except ValueError as e:
    # Re-raise with more context for easier debugging
    raise ValueError(f"PerplexicaCore configuration error: {e}")