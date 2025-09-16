"""
Web Scraper Service for PerplexicaCore
======================================

This service extracts clean text content from web pages using advanced HTML parsing
and content extraction techniques. It implements polite crawling practices and
robust error handling for reliable operation at scale.

Key features:
- Asynchronous HTTP requests for performance
- Advanced content extraction using readability-lxml and trafilatura
- Robots.txt compliance and rate limiting
- Domain-based request throttling
- Content quality filtering and normalization
- Comprehensive error handling and retries
"""

import logging
import asyncio
import time
from typing import Dict, Any, Optional, Set
from urllib.parse import urlparse, urljoin, quote
from urllib.robotparser import RobotFileParser
import httpx
from celery import current_task
import trafilatura
from readability import Document
from bs4 import BeautifulSoup
from message_queue import celery_app
from config import Config

# Configure logging for this module
logger = logging.getLogger(__name__)

# ============================================================================
# DOMAIN RATE LIMITING INFRASTRUCTURE
# ============================================================================

class DomainRateLimiter:
    """
    Domain-based rate limiter to implement polite crawling.
    
    This class ensures we don't overwhelm any single domain with requests,
    maintaining good citizenship on the web while maximizing throughput
    across different domains.
    """
    
    def __init__(self):
        """Initialize rate limiter with domain tracking."""
        # Track last request time per domain
        self._last_request_time: Dict[str, float] = {}
        # Lock for thread-safe access
        self._lock = asyncio.Lock()
    
    async def wait_for_domain(self, domain: str) -> None:
        """
        Wait if necessary to respect rate limits for a domain.
        
        Args:
            domain: Domain name to check rate limits for
        """
        async with self._lock:
            current_time = time.time()
            last_time = self._last_request_time.get(domain, 0)
            
            # Calculate time since last request to this domain
            time_since_last = current_time - last_time
            
            # Wait if we need to respect the domain delay
            if time_since_last < Config.DOMAIN_DELAY:
                wait_time = Config.DOMAIN_DELAY - time_since_last
                logger.debug(f"Rate limiting: waiting {wait_time:.2f}s for domain {domain}")
                await asyncio.sleep(wait_time)
            
            # Update last request time
            self._last_request_time[domain] = time.time()


# Global rate limiter instance
domain_limiter = DomainRateLimiter()

# ============================================================================
# ROBOTS.TXT COMPLIANCE
# ============================================================================

class RobotsChecker:
    """
    Robots.txt compliance checker for ethical web scraping.
    
    This class fetches and caches robots.txt files to ensure we respect
    website owners' crawling preferences.
    """
    
    def __init__(self):
        """Initialize robots checker with caching."""
        # Cache robots.txt parsers by domain
        self._robots_cache: Dict[str, RobotFileParser] = {}
        # Cache expiration times
        self._cache_expiry: Dict[str, float] = {}
        # Cache TTL (1 hour)
        self._cache_ttl = 3600
        # Lock for thread-safe access
        self._lock = asyncio.Lock()
    
    async def can_fetch(self, url: str, user_agent: str = None) -> bool:
        """
        Check if we're allowed to fetch a URL according to robots.txt.
        
        Args:
            url: URL to check
            user_agent: User agent string (defaults to Config.USER_AGENT)
            
        Returns:
            True if allowed to fetch, False otherwise
        """
        if user_agent is None:
            user_agent = Config.USER_AGENT
        
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.lower()
            
            # Get robots parser for this domain
            robots_parser = await self._get_robots_parser(domain)
            
            if robots_parser is None:
                # If we can't fetch robots.txt, assume it's allowed
                # This is the standard behavior for robots.txt
                return True
            
            # Check if URL is allowed for our user agent
            return robots_parser.can_fetch(user_agent, url)
            
        except Exception as e:
            logger.warning(f"Error checking robots.txt for {url}: {e}")
            # Default to allowing if there's an error
            return True
    
    async def _get_robots_parser(self, domain: str) -> Optional[RobotFileParser]:
        """
        Get cached or fetch robots.txt parser for a domain.
        
        Args:
            domain: Domain to get robots.txt for
            
        Returns:
            RobotFileParser instance or None if unavailable
        """
        async with self._lock:
            current_time = time.time()
            
            # Check if we have a valid cached entry
            if (domain in self._robots_cache and 
                domain in self._cache_expiry and
                current_time < self._cache_expiry[domain]):
                return self._robots_cache[domain]
            
            # Fetch robots.txt for this domain
            robots_url = f"https://{domain}/robots.txt"
            
            try:
                async with httpx.AsyncClient(
                    timeout=10.0,
                    follow_redirects=True
                ) as client:
                    
                    response = await client.get(robots_url)
                    
                    if response.status_code == 200:
                        # Parse robots.txt content
                        robots_parser = RobotFileParser()
                        robots_parser.set_url(robots_url)
                        
                        # RobotFileParser expects a file-like object
                        from io import StringIO
                        robots_content = StringIO(response.text)
                        robots_parser.readfp(robots_content)
                        
                        # Cache the parser
                        self._robots_cache[domain] = robots_parser
                        self._cache_expiry[domain] = current_time + self._cache_ttl
                        
                        logger.debug(f"Fetched and cached robots.txt for {domain}")
                        return robots_parser
                    
                    else:
                        logger.debug(f"No robots.txt found for {domain} (status: {response.status_code})")
                        # Cache the fact that there's no robots.txt
                        self._robots_cache[domain] = None
                        self._cache_expiry[domain] = current_time + self._cache_ttl
                        return None
                        
            except Exception as e:
                logger.debug(f"Failed to fetch robots.txt for {domain}: {e}")
                # Cache the failure to avoid repeated attempts
                self._robots_cache[domain] = None
                self._cache_expiry[domain] = current_time + self._cache_ttl
                return None


# Global robots checker instance
robots_checker = RobotsChecker()

# ============================================================================
# CONTENT EXTRACTION ENGINE
# ============================================================================

class ContentExtractor:
    """
    Advanced content extraction engine using multiple techniques.
    
    This class combines trafilatura and readability-lxml to extract clean,
    meaningful content from web pages while filtering out navigation,
    advertisements, and other noise.
    """
    
    def __init__(self):
        """Initialize content extractor with optimal settings."""
        # Configure trafilatura for best quality extraction
        self.trafilatura_config = trafilatura.settings.use_config()
        self.trafilatura_config.set('DEFAULT', 'MIN_EXTRACTED_SIZE', '200')
        self.trafilatura_config.set('DEFAULT', 'MIN_OUTPUT_SIZE', '100')
        self.trafilatura_config.set('DEFAULT', 'MIN_OUTPUT_COMM_SIZE', '50')
    
    def extract_content(self, html: str, url: str) -> Dict[str, Any]:
        """
        Extract clean content from HTML using multiple techniques.
        
        This method tries multiple extraction approaches and chooses the best
        result based on content quality metrics.
        
        Args:
            html: Raw HTML content
            url: Source URL for context
            
        Returns:
            Dictionary containing extracted content and metadata
        """
        
        extraction_results = []
        
        # Method 1: Trafilatura (generally best for news articles and blogs)
        try:
            trafilatura_result = self._extract_with_trafilatura(html, url)
            if trafilatura_result:
                extraction_results.append(('trafilatura', trafilatura_result))
        except Exception as e:
            logger.debug(f"Trafilatura extraction failed for {url}: {e}")
        
        # Method 2: Readability (good for complex layouts)
        try:
            readability_result = self._extract_with_readability(html, url)
            if readability_result:
                extraction_results.append(('readability', readability_result))
        except Exception as e:
            logger.debug(f"Readability extraction failed for {url}: {e}")
        
        # Method 3: BeautifulSoup fallback (basic but reliable)
        try:
            soup_result = self._extract_with_soup(html, url)
            if soup_result:
                extraction_results.append(('beautifulsoup', soup_result))
        except Exception as e:
            logger.debug(f"BeautifulSoup extraction failed for {url}: {e}")
        
        # Choose the best extraction result
        if extraction_results:
            best_method, best_result = self._choose_best_extraction(extraction_results)
            logger.debug(f"Selected {best_method} extraction for {url}")
            return best_result
        else:
            logger.warning(f"All extraction methods failed for {url}")
            return {
                'title': '',
                'content': '',
                'word_count': 0,
                'extraction_method': 'failed',
                'error': 'All extraction methods failed'
            }
    
    def _extract_with_trafilatura(self, html: str, url: str) -> Optional[Dict[str, Any]]:
        """Extract content using trafilatura."""
        
        # Extract main content
        content = trafilatura.extract(
            html,
            config=self.trafilatura_config,
            include_comments=False,  # Exclude comments for cleaner content
            include_tables=True,     # Include tables for factual content
            include_images=False,    # Text-only extraction
            include_formatting=False, # Plain text output
            deduplicate=True,        # Remove duplicate paragraphs
            url=url                  # Provide URL for context
        )
        
        if not content or len(content.strip()) < 100:
            return None
        
        # Extract title separately
        title = trafilatura.extract(
            html,
            config=self.trafilatura_config,
            output_format='xml',
            include_formatting=True
        )
        
        # Parse title from XML output if available
        extracted_title = ''
        if title:
            try:
                from xml.etree import ElementTree as ET
                root = ET.fromstring(f"<root>{title}</root>")
                title_elem = root.find('.//head/title')
                if title_elem is not None and title_elem.text:
                    extracted_title = title_elem.text.strip()
            except Exception:
                pass
        
        # Fall back to HTML title if trafilatura didn't find one
        if not extracted_title:
            soup = BeautifulSoup(html, 'html.parser')
            title_tag = soup.find('title')
            if title_tag and title_tag.string:
                extracted_title = title_tag.string.strip()
        
        return {
            'title': extracted_title,
            'content': content.strip(),
            'word_count': len(content.split()),
            'extraction_method': 'trafilatura'
        }
    
    def _extract_with_readability(self, html: str, url: str) -> Optional[Dict[str, Any]]:
        """Extract content using readability-lxml."""
        
        # Use readability to extract main content
        doc = Document(html, url=url)
        
        # Get cleaned HTML
        content_html = doc.summary()
        title = doc.title() or ''
        
        if not content_html:
            return None
        
        # Convert HTML to plain text
        soup = BeautifulSoup(content_html, 'html.parser')
        
        # Remove script and style elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer']):
            element.decompose()
        
        # Extract text content
        content = soup.get_text()
        
        # Clean up whitespace
        content = ' '.join(content.split())
        
        if len(content.strip()) < 100:
            return None
        
        return {
            'title': title.strip(),
            'content': content.strip(),
            'word_count': len(content.split()),
            'extraction_method': 'readability'
        }
    
    def _extract_with_soup(self, html: str, url: str) -> Optional[Dict[str, Any]]:
        """Basic extraction using BeautifulSoup as fallback."""
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract title
        title = ''
        title_tag = soup.find('title')
        if title_tag and title_tag.string:
            title = title_tag.string.strip()
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 
                           'aside', 'advertisement', 'sidebar']):
            element.decompose()
        
        # Try to find main content areas
        content_selectors = [
            'main', 'article', '.content', '.main-content', 
            '.post-content', '.entry-content', '#content'
        ]
        
        content_element = None
        for selector in content_selectors:
            content_element = soup.select_one(selector)
            if content_element:
                break
        
        # Fall back to body if no main content found
        if not content_element:
            content_element = soup.find('body')
        
        if not content_element:
            return None
        
        # Extract text content
        content = content_element.get_text()
        content = ' '.join(content.split())
        
        if len(content.strip()) < 100:
            return None
        
        return {
            'title': title,
            'content': content.strip(),
            'word_count': len(content.split()),
            'extraction_method': 'beautifulsoup'
        }
    
    def _choose_best_extraction(self, results: list) -> tuple:
        """
        Choose the best extraction result based on quality metrics.
        
        Args:
            results: List of (method_name, result_dict) tuples
            
        Returns:
            Tuple of (best_method, best_result)
        """
        
        def score_result(result_dict):
            """Score an extraction result based on quality metrics."""
            score = 0
            
            # Word count score (prefer more content, but with diminishing returns)
            word_count = result_dict.get('word_count', 0)
            if word_count > 100:
                score += min(word_count / 100, 10)  # Max 10 points for word count
            
            # Title score (prefer results with titles)
            if result_dict.get('title', '').strip():
                score += 5
            
            # Content quality score (basic checks)
            content = result_dict.get('content', '')
            if content:
                # Prefer content with proper sentence structure
                sentence_count = content.count('.') + content.count('!') + content.count('?')
                if sentence_count > 0:
                    score += min(sentence_count / 10, 3)  # Max 3 points for sentences
                
                # Prefer content with paragraphs
                paragraph_count = content.count('\n\n') + 1
                if paragraph_count > 1:
                    score += min(paragraph_count / 5, 2)  # Max 2 points for structure
            
            return score
        
        # Score all results
        scored_results = []
        for method, result in results:
            score = score_result(result)
            scored_results.append((score, method, result))
        
        # Sort by score (highest first)
        scored_results.sort(reverse=True)
        
        # Return the best result
        _, best_method, best_result = scored_results[0]
        return best_method, best_result


# Global content extractor instance
content_extractor = ContentExtractor()

# ============================================================================
# WEB SCRAPING CLIENT
# ============================================================================

class WebScraper:
    """
    Asynchronous web scraping client with ethical crawling practices.
    
    This class handles all aspects of web page fetching including:
    - Robots.txt compliance
    - Rate limiting per domain
    - Content extraction and cleaning
    - Error handling and retries
    """
    
    def __init__(self):
        """Initialize web scraper with optimal HTTP settings."""
        
        # HTTP timeout configuration
        self.timeout = httpx.Timeout(
            connect=10.0,  # Connection timeout
            read=Config.SCRAPER_TIMEOUT,  # Read timeout
            write=10.0,    # Write timeout
            pool=60.0      # Total timeout
        )
        
        # Connection limits
        self.limits = httpx.Limits(
            max_keepalive_connections=20,
            max_connections=50,
            keepalive_expiry=30.0
        )
        
        # Default headers for requests
        self.headers = {
            'User-Agent': Config.USER_AGENT,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',  # Do Not Track
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    
    async def scrape_url(self, url: str) -> Dict[str, Any]:
        """
        Scrape content from a single URL with ethical practices.
        
        Args:
            url: URL to scrape
            
        Returns:
            Dictionary containing scraped content and metadata
        """
        
        try:
            # Parse URL for domain extraction
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.lower()
            
            logger.debug(f"Starting scrape for URL: {url}")
            
            # Check robots.txt compliance
            if not await robots_checker.can_fetch(url):
                logger.info(f"Robots.txt disallows scraping of {url}")
                return {
                    'url': url,
                    'title': '',
                    'content': '',
                    'word_count': 0,
                    'error': 'Disallowed by robots.txt',
                    'success': False
                }
            
            # Apply domain-based rate limiting
            await domain_limiter.wait_for_domain(domain)
            
            # Fetch the web page
            async with httpx.AsyncClient(
                timeout=self.timeout,
                limits=self.limits,
                follow_redirects=True,
                headers=self.headers
            ) as client:
                
                logger.debug(f"Fetching HTML for {url}")
                response = await client.get(url)
                
                # Check for successful response
                response.raise_for_status()
                
                # Verify content type
                content_type = response.headers.get('content-type', '').lower()
                if 'text/html' not in content_type:
                    logger.warning(f"Non-HTML content type for {url}: {content_type}")
                    return {
                        'url': url,
                        'title': '',
                        'content': '',
                        'word_count': 0,
                        'error': f'Non-HTML content: {content_type}',
                        'success': False
                    }
                
                # Get HTML content
                html = response.text
                
                # Check content length
                if len(html) > Config.MAX_CONTENT_LENGTH:
                    logger.warning(f"Content too large for {url}: {len(html)} chars")
                    html = html[:Config.MAX_CONTENT_LENGTH]
                
                # Extract clean content
                logger.debug(f"Extracting content from {url}")
                extracted = content_extractor.extract_content(html, url)
                
                # Add success metadata
                extracted.update({
                    'url': url,
                    'success': True,
                    'response_size': len(html),
                    'status_code': response.status_code
                })
                
                logger.info(f"Successfully scraped {url}: {extracted.get('word_count', 0)} words")
                
                return extracted
                
        except httpx.HTTPStatusError as e:
            logger.warning(f"HTTP error scraping {url}: {e.response.status_code}")
            return {
                'url': url,
                'title': '',
                'content': '',
                'word_count': 0,
                'error': f'HTTP {e.response.status_code}',
                'success': False
            }
            
        except httpx.TimeoutException as e:
            logger.warning(f"Timeout scraping {url}: {e}")
            return {
                'url': url,
                'title': '',
                'content': '',
                'word_count': 0,
                'error': 'Request timeout',
                'success': False
            }
            
        except Exception as e:
            logger.error(f"Unexpected error scraping {url}: {e}")
            return {
                'url': url,
                'title': '',
                'content': '',
                'word_count': 0,
                'error': f'Scraping failed: {str(e)}',
                'success': False
            }


# ============================================================================
# CELERY TASK DEFINITION
# ============================================================================

@celery_app.task(
    bind=True,
    name='services.scraper_service.scrape_url',
    max_retries=2,
    default_retry_delay=120,
    autoretry_for=(httpx.HTTPError, asyncio.TimeoutError),
    retry_backoff=True,
    retry_jitter=True
)
def scrape_url(self, url: str) -> Dict[str, Any]:
    """
    Celery task to scrape content from a single URL.
    
    This task implements ethical web scraping with robots.txt compliance,
    rate limiting, and robust error handling.
    
    Args:
        url: URL to scrape
        
    Returns:
        Dictionary containing:
        - 'url': Original URL
        - 'title': Extracted page title
        - 'content': Clean text content
        - 'word_count': Number of words in content
        - 'success': Boolean indicating success
        - 'error': Error message if failed
        - Additional metadata
        
    Raises:
        ValueError: For invalid URLs
        httpx.HTTPError: For network/HTTP errors (will trigger retry)
    """
    
    # Validate URL
    if not url or not url.strip():
        raise ValueError("URL cannot be empty")
    
    url = url.strip()
    
    # Basic URL validation
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid URL format: {url}")
    except Exception as e:
        raise ValueError(f"URL parsing failed: {e}")
    
    # Log task start
    task_id = current_task.request.id if current_task else 'unknown'
    logger.info(f"Scraper task {task_id} started for URL: {url}")
    
    try:
        # Create scraper and perform scraping
        scraper = WebScraper()
        
        # Run the async scraping operation
        result = asyncio.run(scraper.scrape_url(url))
        
        # Log completion
        if result['success']:
            logger.info(
                f"Scraper task {task_id} completed successfully: "
                f"{result.get('word_count', 0)} words from {url}"
            )
        else:
            logger.warning(
                f"Scraper task {task_id} failed: "
                f"{result.get('error', 'unknown')} for {url}"
            )
        
        return result
        
    except Exception as e:
        logger.error(f"Scraper task {task_id} failed for URL {url}: {e}")
        
        # Re-raise for Celery retry mechanism
        raise self.retry(exc=e, countdown=120, max_retries=2)


# ============================================================================
# BATCH SCRAPING UTILITIES
# ============================================================================

@celery_app.task(
    name='services.scraper_service.scrape_urls_batch',
    max_retries=1
)
def scrape_urls_batch(urls: list) -> Dict[str, Any]:
    """
    Celery task to scrape multiple URLs in parallel.
    
    This task coordinates parallel scraping of multiple URLs while
    respecting rate limits and resource constraints.
    
    Args:
        urls: List of URLs to scrape
        
    Returns:
        Dictionary containing:
        - 'results': List of scraping results
        - 'total_urls': Number of URLs processed
        - 'successful': Number of successful scrapes
        - 'failed': Number of failed scrapes
    """
    
    if not urls:
        return {
            'results': [],
            'total_urls': 0,
            'successful': 0,
            'failed': 0
        }
    
    task_id = current_task.request.id if current_task else 'unknown'
    logger.info(f"Batch scraper task {task_id} started for {len(urls)} URLs")
    
    # Submit individual scraping tasks
    from celery import group
    job = group(scrape_url.s(url) for url in urls)
    result_group = job.apply_async()
    
    # Wait for all tasks to complete
    results = result_group.get()
    
    # Calculate statistics
    successful = sum(1 for r in results if r.get('success', False))
    failed = len(results) - successful
    
    logger.info(
        f"Batch scraper task {task_id} completed: "
        f"{successful} successful, {failed} failed out of {len(urls)} URLs"
    )
    
    return {
        'results': results,
        'total_urls': len(urls),
        'successful': successful,
        'failed': failed
    }


# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

logger.info("Scraper service initialized")
logger.info(f"Domain delay: {Config.DOMAIN_DELAY}s")
logger.info(f"Scraper timeout: {Config.SCRAPER_TIMEOUT}s")
logger.info(f"Max content length: {Config.MAX_CONTENT_LENGTH} chars")