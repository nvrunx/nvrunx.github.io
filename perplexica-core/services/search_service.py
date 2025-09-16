"""
Search Service for PerplexicaCore
=================================

This service handles search queries by interfacing with SearxNG, a privacy-focused
meta-search engine. It fetches search results and performs initial relevance filtering
before passing URLs to the scraping service.

Key features:
- Asynchronous SearxNG API calls for performance
- Robust error handling and retries
- Result quality filtering and deduplication
- Caching for common queries
- Rate limiting compliance
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse
import httpx
from celery import current_task
from message_queue import celery_app, cached_task
from config import Config

# Configure logging for this module
logger = logging.getLogger(__name__)

# ============================================================================
# SEARXNG INTERFACE CLASS
# ============================================================================

class SearxNGClient:
    """
    Asynchronous client for SearxNG meta-search engine.
    
    SearxNG aggregates results from multiple search engines while preserving
    privacy. This client handles API communication, error recovery, and
    result processing.
    """
    
    def __init__(self):
        """
        Initialize the SearxNG client with optimal HTTP settings.
        
        Configuration focuses on:
        - Connection pooling for efficiency
        - Reasonable timeouts to prevent hanging
        - User agent identification for compliance
        - SSL verification for security
        """
        self.base_url = Config.SEARXNG_URL
        self.timeout = httpx.Timeout(
            connect=10.0,  # Time to establish connection
            read=30.0,     # Time to read response
            write=10.0,    # Time to send request
            pool=60.0      # Total time including retries
        )
        
        # HTTP client with connection pooling
        # Limits prevent resource exhaustion under load
        self.limits = httpx.Limits(
            max_keepalive_connections=20,  # Reuse connections
            max_connections=50,            # Total connection limit
            keepalive_expiry=30.0          # Connection reuse timeout
        )
    
    async def search(
        self, 
        query: str, 
        categories: List[str] = None,
        engines: List[str] = None,
        language: str = 'en',
        safe_search: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Perform search query against SearxNG instance.
        
        Args:
            query: Search query string
            categories: Search categories (e.g., ['general', 'news'])
            engines: Specific engines to use (e.g., ['google', 'bing'])
            language: Language code for results (default: 'en')
            safe_search: Safe search level (0=off, 1=moderate, 2=strict)
            
        Returns:
            List of search result dictionaries with standardized format
            
        Raises:
            httpx.HTTPError: For network/HTTP errors
            ValueError: For malformed responses
        """
        
        # Default to general web search if no categories specified
        if categories is None:
            categories = ['general']
        
        # Construct search API URL
        search_url = urljoin(self.base_url, '/search')
        
        # Build query parameters for SearxNG API
        # These parameters control search behavior and result format
        params = {
            'q': query,                           # Search query
            'format': 'json',                     # JSON response format
            'categories': ','.join(categories),   # Search categories
            'language': language,                 # Result language
            'safesearch': safe_search,            # Safe search level
            'pageno': 1,                         # Page number (first page)
        }
        
        # Add specific engines if requested
        if engines:
            params['engines'] = ','.join(engines)
        
        # Set request headers for identification and format preference
        headers = {
            'User-Agent': Config.USER_AGENT,
            'Accept': 'application/json',
            'Accept-Language': language,
        }
        
        # Perform async HTTP request with error handling
        async with httpx.AsyncClient(
            timeout=self.timeout,
            limits=self.limits,
            follow_redirects=True  # Handle redirects automatically
        ) as client:
            
            logger.debug(f"Searching SearxNG for query: '{query}'")
            
            try:
                response = await client.get(
                    search_url,
                    params=params,
                    headers=headers
                )
                
                # Raise exception for HTTP error status codes
                response.raise_for_status()
                
                # Parse JSON response
                data = response.json()
                
                # Extract results from SearxNG response format
                results = data.get('results', [])
                
                logger.info(f"SearxNG returned {len(results)} results for query: '{query}'")
                
                return results
                
            except httpx.TimeoutException as e:
                logger.error(f"SearxNG search timeout for query '{query}': {e}")
                raise
                
            except httpx.HTTPStatusError as e:
                logger.error(f"SearxNG HTTP error for query '{query}': {e.response.status_code}")
                raise
                
            except Exception as e:
                logger.error(f"Unexpected error during SearxNG search for query '{query}': {e}")
                raise


# ============================================================================
# RESULT PROCESSING UTILITIES
# ============================================================================

def filter_and_rank_results(
    results: List[Dict[str, Any]], 
    max_results: int = None
) -> List[Dict[str, str]]:
    """
    Filter and rank search results for quality and relevance.
    
    This function implements the first stage of relevance filtering by:
    1. Removing invalid or low-quality results
    2. Deduplicating similar URLs
    3. Ranking by search engine confidence scores
    4. Limiting to top results
    
    Args:
        results: Raw search results from SearxNG
        max_results: Maximum number of results to return
        
    Returns:
        List of filtered and ranked result dictionaries
    """
    
    if max_results is None:
        max_results = Config.SEARCH_RESULTS_LIMIT
    
    filtered_results = []
    seen_urls = set()
    seen_domains = {}
    
    for result in results:
        try:
            # Extract required fields with fallbacks
            url = result.get('url', '').strip()
            title = result.get('title', '').strip()
            content = result.get('content', '').strip()
            
            # Skip results missing critical information
            if not url or not title:
                logger.debug(f"Skipping result missing URL or title: {result}")
                continue
            
            # Parse URL to validate and extract domain
            try:
                parsed_url = urlparse(url)
                domain = parsed_url.netloc.lower()
                
                # Skip invalid URLs
                if not domain or not parsed_url.scheme:
                    logger.debug(f"Skipping invalid URL: {url}")
                    continue
                    
            except Exception as e:
                logger.debug(f"Skipping result with unparseable URL '{url}': {e}")
                continue
            
            # Deduplicate exact URLs
            if url in seen_urls:
                logger.debug(f"Skipping duplicate URL: {url}")
                continue
            
            # Limit results per domain to ensure diversity
            # This prevents a single site from dominating results
            domain_count = seen_domains.get(domain, 0)
            if domain_count >= 3:  # Max 3 results per domain
                logger.debug(f"Skipping result from over-represented domain: {domain}")
                continue
            
            # Apply quality filters
            if not _is_quality_result(url, title, content):
                continue
            
            # Create standardized result format
            filtered_result = {
                'url': url,
                'title': title,
                'content': content,
                'domain': domain,
                'score': result.get('score', 0),  # SearxNG relevance score
            }
            
            filtered_results.append(filtered_result)
            seen_urls.add(url)
            seen_domains[domain] = domain_count + 1
            
        except Exception as e:
            logger.warning(f"Error processing search result: {e}")
            continue
    
    # Sort by relevance score (higher is better)
    # SearxNG provides relevance scores that we can trust for initial ranking
    filtered_results.sort(key=lambda x: x.get('score', 0), reverse=True)
    
    # Limit to requested number of results
    final_results = filtered_results[:max_results]
    
    logger.info(f"Filtered {len(results)} raw results to {len(final_results)} quality results")
    
    return final_results


def _is_quality_result(url: str, title: str, content: str) -> bool:
    """
    Determine if a search result meets quality criteria.
    
    This function filters out low-quality results that are unlikely to contain
    useful information for answering user queries.
    
    Args:
        url: Result URL
        title: Result title
        content: Result content/description
        
    Returns:
        True if result meets quality criteria, False otherwise
    """
    
    # Convert to lowercase for case-insensitive matching
    url_lower = url.lower()
    title_lower = title.lower()
    content_lower = content.lower()
    
    # Filter out common low-quality or problematic content types
    excluded_extensions = [
        '.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx',
        '.zip', '.rar', '.tar', '.gz', '.exe', '.dmg', '.iso',
        '.mp3', '.mp4', '.avi', '.mov', '.wav', '.flv',
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'
    ]
    
    # Skip binary files and non-text content
    if any(ext in url_lower for ext in excluded_extensions):
        logger.debug(f"Skipping non-text content: {url}")
        return False
    
    # Filter out social media, forums, and low-content sites
    excluded_domains = [
        'facebook.com', 'twitter.com', 'instagram.com', 'tiktok.com',
        'reddit.com', 'pinterest.com', 'youtube.com',
        'linkedin.com/pulse',  # Allow LinkedIn articles but not profiles
        'quora.com/profile',   # Allow Quora answers but not profiles
    ]
    
    if any(domain in url_lower for domain in excluded_domains):
        logger.debug(f"Skipping social media/forum content: {url}")
        return False
    
    # Filter out very short titles or content (likely low-quality)
    if len(title.strip()) < 10:
        logger.debug(f"Skipping result with short title: {title}")
        return False
    
    # Filter out spam-like content with repeated keywords
    if _has_spam_characteristics(title, content):
        logger.debug(f"Skipping spam-like content: {title}")
        return False
    
    # Filter out error pages and placeholder content
    error_indicators = [
        '404', 'not found', 'page not found', 'error',
        'access denied', 'forbidden', 'unauthorized',
        'coming soon', 'under construction', 'maintenance'
    ]
    
    if any(indicator in title_lower or indicator in content_lower 
           for indicator in error_indicators):
        logger.debug(f"Skipping error/placeholder page: {title}")
        return False
    
    return True


def _has_spam_characteristics(title: str, content: str) -> bool:
    """
    Detect spam-like characteristics in search results.
    
    Args:
        title: Result title
        content: Result content
        
    Returns:
        True if content appears to be spam, False otherwise
    """
    
    # Combine title and content for analysis
    combined_text = f"{title} {content}".lower()
    
    # Check for excessive keyword repetition
    words = combined_text.split()
    if len(words) > 10:
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Only count significant words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # If any word appears more than 30% of the time, likely spam
        max_freq = max(word_freq.values()) if word_freq else 0
        if max_freq > len(words) * 0.3:
            return True
    
    # Check for common spam phrases
    spam_phrases = [
        'click here', 'buy now', 'limited time', 'act now',
        'free money', 'make money', 'work from home',
        'viagra', 'cialis', 'weight loss', 'diet pills'
    ]
    
    if any(phrase in combined_text for phrase in spam_phrases):
        return True
    
    return False


# ============================================================================
# CELERY TASK DEFINITION
# ============================================================================

@celery_app.task(
    bind=True,
    name='services.search_service.search_web',
    max_retries=3,
    default_retry_delay=60,
    autoretry_for=(httpx.HTTPError, asyncio.TimeoutError),
    retry_backoff=True,
    retry_jitter=True
)
@cached_task('search', cache_ttl=300)  # Cache for 5 minutes
def search_web(self, query: str, max_results: int = None) -> Dict[str, Any]:
    """
    Celery task to search the web using SearxNG and return filtered results.
    
    This is the main entry point for web search functionality. It:
    1. Validates input parameters
    2. Performs async search against SearxNG
    3. Filters and ranks results for quality
    4. Returns structured data for downstream processing
    
    Args:
        query: User search query
        max_results: Maximum number of results to return
        
    Returns:
        Dictionary containing:
        - 'results': List of search result dictionaries
        - 'query': Original query
        - 'total_found': Number of results found
        - 'processing_time': Time taken for search
        
    Raises:
        ValueError: For invalid input parameters
        httpx.HTTPError: For network/API errors
    """
    
    # Validate input parameters
    if not query or not query.strip():
        raise ValueError("Search query cannot be empty")
    
    query = query.strip()
    
    if max_results is None:
        max_results = Config.SEARCH_RESULTS_LIMIT
    
    if max_results <= 0:
        raise ValueError("max_results must be positive")
    
    # Log task start
    task_id = current_task.request.id if current_task else 'unknown'
    logger.info(f"Search task {task_id} started for query: '{query}'")
    
    try:
        # Record start time for performance monitoring
        import time
        start_time = time.time()
        
        # Create SearxNG client and perform search
        # We use asyncio.run to handle the async search call
        client = SearxNGClient()
        
        # Run the async search in the current thread
        # This allows us to use async libraries in Celery tasks
        raw_results = asyncio.run(client.search(query))
        
        # Filter and rank results for quality
        filtered_results = filter_and_rank_results(raw_results, max_results)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Prepare response data
        response = {
            'results': filtered_results,
            'query': query,
            'total_found': len(filtered_results),
            'processing_time': round(processing_time, 2),
            'task_id': task_id,
        }
        
        logger.info(
            f"Search task {task_id} completed: "
            f"found {len(filtered_results)} results in {processing_time:.2f}s"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Search task {task_id} failed for query '{query}': {e}")
        
        # Re-raise for Celery retry mechanism
        raise self.retry(exc=e, countdown=60, max_retries=3)


# ============================================================================
# UTILITY FUNCTIONS FOR EXTERNAL USAGE
# ============================================================================

def search_sync(query: str, max_results: int = None) -> Dict[str, Any]:
    """
    Synchronous wrapper for search functionality.
    
    This function provides a simple interface for testing and debugging
    without requiring Celery infrastructure.
    
    Args:
        query: Search query
        max_results: Maximum results to return
        
    Returns:
        Search results in same format as Celery task
    """
    
    try:
        # Run the search directly without Celery
        client = SearxNGClient()
        raw_results = asyncio.run(client.search(query))
        filtered_results = filter_and_rank_results(
            raw_results, 
            max_results or Config.SEARCH_RESULTS_LIMIT
        )
        
        return {
            'results': filtered_results,
            'query': query,
            'total_found': len(filtered_results),
            'processing_time': 0,  # Not tracked in sync mode
            'task_id': 'sync',
        }
        
    except Exception as e:
        logger.error(f"Synchronous search failed for query '{query}': {e}")
        raise


# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

logger.info("Search service initialized")
logger.info(f"SearxNG URL: {Config.SEARXNG_URL}")
logger.info(f"Max search results: {Config.SEARCH_RESULTS_LIMIT}")