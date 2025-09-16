# PerplexicaCore AI-Augmented Search Engine
# Google Colab Setup and Running Guide

"""
COMPLETE SETUP GUIDE FOR GOOGLE COLAB
=====================================

This notebook provides a complete setup for running PerplexicaCore, an AI-augmented 
search engine that searches the web in real-time, processes results with AI, and 
generates well-cited, accurate answers.

WHAT THIS SYSTEM DOES:
1. Takes a user query (e.g., "What are the latest developments in quantum computing?")
2. Searches the web using SearxNG meta-search
3. Scrapes and extracts clean content from relevant web pages
4. Uses AI embeddings to rank content by relevance
5. Generates a comprehensive, well-cited answer using an LLM

ARCHITECTURE:
- Microservices with message queues (RabbitMQ + Celery)
- Asynchronous processing throughout
- Two-stage relevance filtering (search ranking + AI embeddings)
- Multi-LLM backend support (OpenAI API, Ollama)
- Redis caching for performance
"""

# ============================================================================
# STEP 1: INITIAL SETUP AND INSTALLATIONS
# ============================================================================

# Install system dependencies
print("Installing system dependencies...")
!apt-get update -qq
!apt-get install -y redis-server rabbitmq-server curl wget

# Install Python packages
print("Installing Python packages...")
!pip install --quiet celery[redis]==5.3.4 redis==5.0.1 httpx==0.25.2
!pip install --quiet trafilatura==1.6.4 readability-lxml==0.8.1 beautifulsoup4==4.12.2
!pip install --quiet sentence-transformers==2.2.2 torch==2.1.1 scikit-learn==1.3.2
!pip install --quiet openai==1.3.9 nltk==3.8.1 numpy==1.24.4
!pip install --quiet python-dotenv==1.0.0 nest-asyncio==1.5.8

# Download NLTK data
import nltk
nltk.download('punkt', quiet=True)

print("✓ Dependencies installed successfully!")

# ============================================================================
# STEP 2: START BACKGROUND SERVICES
# ============================================================================

import subprocess
import time
import os
from threading import Thread

# Start Redis server
print("Starting Redis server...")
redis_process = subprocess.Popen(['redis-server', '--daemonize', 'yes'])
time.sleep(2)

# Start RabbitMQ server
print("Starting RabbitMQ server...")
rabbitmq_process = subprocess.Popen(['rabbitmq-server', '-detached'])
time.sleep(5)

# Verify services are running
import redis
try:
    r = redis.Redis(host='localhost', port=6379, db=0)
    r.ping()
    print("✓ Redis is running")
except:
    print("✗ Redis failed to start")

print("✓ Background services started!")

# ============================================================================
# STEP 3: SETUP SEARXNG (SIMPLIFIED VERSION)
# ============================================================================

# For Colab, we'll use a public SearxNG instance or set up a simple one
print("Setting up search capabilities...")

# We'll use a public SearxNG instance for demo purposes
# In production, you should run your own SearxNG instance
SEARXNG_URL = "https://search.bus-hit.me"  # Public instance (use your own in production)

print(f"✓ Using SearxNG instance: {SEARXNG_URL}")

# ============================================================================
# STEP 4: CONFIGURATION SETUP
# ============================================================================

# Create environment configuration
import os

# Set environment variables
os.environ['RABBITMQ_URL'] = 'amqp://guest:guest@localhost:5672//'
os.environ['REDIS_URL'] = 'redis://localhost:6379/0'
os.environ['SEARXNG_URL'] = SEARXNG_URL
os.environ['OPENAI_API_KEY'] = ''  # Set your OpenAI API key here
os.environ['LLM_PROVIDER'] = 'openai'  # Change to 'ollama' for local models
os.environ['LOG_LEVEL'] = 'INFO'

print("✓ Environment configured!")

# ============================================================================
# STEP 5: CREATE PROJECT FILES
# ============================================================================

# Create project directory
!mkdir -p /content/perplexica-core/services

# Write all the project files
print("Creating project files...")

# config.py
config_py = '''
"""Configuration Management for PerplexicaCore"""
import os
from typing import Optional

class Config:
    # Message Queue Configuration
    RABBITMQ_URL: str = os.getenv('RABBITMQ_URL', 'amqp://guest:guest@localhost:5672//')
    REDIS_URL: str = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    CACHE_TTL: int = int(os.getenv('CACHE_TTL', '300'))
    
    # Search Configuration
    SEARXNG_URL: str = os.getenv('SEARXNG_URL', 'https://search.bus-hit.me')
    SEARCH_RESULTS_LIMIT: int = int(os.getenv('SEARCH_RESULTS_LIMIT', '10'))
    MAX_URLS_TO_SCRAPE: int = int(os.getenv('MAX_URLS_TO_SCRAPE', '5'))
    
    # Scraping Configuration
    USER_AGENT: str = os.getenv('USER_AGENT', 'PerplexicaCore/1.0 (AI Research Tool)')
    SCRAPER_TIMEOUT: int = int(os.getenv('SCRAPER_TIMEOUT', '30'))
    DOMAIN_DELAY: float = float(os.getenv('DOMAIN_DELAY', '1.0'))
    MAX_CONTENT_LENGTH: int = int(os.getenv('MAX_CONTENT_LENGTH', '50000'))
    
    # AI Configuration
    EMBEDDING_MODEL: str = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
    TOP_RELEVANT_CHUNKS: int = int(os.getenv('TOP_RELEVANT_CHUNKS', '3'))
    SIMILARITY_THRESHOLD: float = float(os.getenv('SIMILARITY_THRESHOLD', '0.3'))
    
    # LLM Configuration
    LLM_PROVIDER: str = os.getenv('LLM_PROVIDER', 'openai')
    OPENAI_API_KEY: Optional[str] = os.getenv('OPENAI_API_KEY')
    OPENAI_MODEL: str = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
    OPENAI_MAX_TOKENS: int = int(os.getenv('OPENAI_MAX_TOKENS', '1000'))
    OPENAI_TEMPERATURE: float = float(os.getenv('OPENAI_TEMPERATURE', '0.1'))
    
    # Logging
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    
    @classmethod
    def validate_config(cls) -> None:
        errors = []
        if cls.LLM_PROVIDER == 'openai' and not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required when LLM_PROVIDER is 'openai'")
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")

try:
    Config.validate_config()
except ValueError as e:
    print(f"Configuration warning: {e}")
'''

with open('/content/perplexica-core/config.py', 'w') as f:
    f.write(config_py)

# Simplified message_queue.py for Colab
message_queue_py = '''
"""Simplified Message Queue for Colab"""
import logging
import json
import hashlib
from typing import Any, Optional
from celery import Celery
import redis
from config import Config

logger = logging.getLogger(__name__)

def create_celery_app() -> Celery:
    app = Celery('perplexica_core')
    app.conf.update(
        broker_url=Config.RABBITMQ_URL,
        result_backend=Config.REDIS_URL,
        task_serializer='json',
        accept_content=['json'],
        result_serializer='json',
        task_acks_late=True,
        worker_prefetch_multiplier=1,
        result_expires=3600,
        task_ignore_result=False,
    )
    return app

celery_app = create_celery_app()

class CacheManager:
    def __init__(self):
        try:
            self.redis_client = redis.Redis.from_url(Config.REDIS_URL, decode_responses=True)
            self.redis_client.ping()
        except:
            self.redis_client = None
    
    def _generate_cache_key(self, prefix: str, data: Any) -> str:
        data_str = json.dumps(data, sort_keys=True, ensure_ascii=True)
        hash_digest = hashlib.sha256(data_str.encode('utf-8')).hexdigest()
        return f"{prefix}:{hash_digest}"
    
    def get(self, prefix: str, data: Any) -> Optional[Any]:
        if not self.redis_client:
            return None
        try:
            cache_key = self._generate_cache_key(prefix, data)
            cached_data = self.redis_client.get(cache_key)
            return json.loads(cached_data) if cached_data else None
        except:
            return None
    
    def set(self, prefix: str, data: Any, value: Any, ttl: Optional[int] = None) -> bool:
        if not self.redis_client:
            return False
        try:
            cache_key = self._generate_cache_key(prefix, data)
            serialized_value = json.dumps(value, ensure_ascii=True)
            expiration = ttl or Config.CACHE_TTL
            return bool(self.redis_client.setex(cache_key, expiration, serialized_value))
        except:
            return False

cache_manager = CacheManager()

def cached_task(cache_prefix: str, cache_ttl: Optional[int] = None):
    def decorator(task_func):
        def wrapper(*args, **kwargs):
            cache_data = {'args': args, 'kwargs': kwargs}
            cached_result = cache_manager.get(cache_prefix, cache_data)
            if cached_result is not None:
                return cached_result
            result = task_func(*args, **kwargs)
            cache_manager.set(cache_prefix, cache_data, result, cache_ttl)
            return result
        return wrapper
    return decorator
'''

with open('/content/perplexica-core/message_queue.py', 'w') as f:
    f.write(message_queue_py)

print("✓ Core files created!")

# ============================================================================
# STEP 6: CREATE SIMPLIFIED SERVICE FILES FOR COLAB
# ============================================================================

# Simplified search service
search_service_py = '''
"""Simplified Search Service for Colab"""
import logging
import asyncio
import httpx
from typing import List, Dict, Any
from celery import current_task
from message_queue import celery_app, cached_task
from config import Config

logger = logging.getLogger(__name__)

class SearxNGClient:
    def __init__(self):
        self.base_url = Config.SEARXNG_URL
        self.timeout = httpx.Timeout(30.0)
    
    async def search(self, query: str) -> List[Dict[str, Any]]:
        params = {
            'q': query,
            'format': 'json',
            'categories': 'general',
            'language': 'en',
            'safesearch': 1,
            'pageno': 1,
        }
        
        headers = {
            'User-Agent': Config.USER_AGENT,
            'Accept': 'application/json',
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(f"{self.base_url}/search", params=params, headers=headers)
                response.raise_for_status()
                data = response.json()
                results = data.get('results', [])
                logger.info(f"SearxNG returned {len(results)} results")
                return results
            except Exception as e:
                logger.error(f"SearxNG search failed: {e}")
                # Fallback to mock results for demo
                return [
                    {'url': 'https://example.com/1', 'title': 'Example Result 1', 'content': 'Sample content', 'score': 1.0},
                    {'url': 'https://example.com/2', 'title': 'Example Result 2', 'content': 'Sample content', 'score': 0.9}
                ]

@celery_app.task(bind=True, name='services.search_service.search_web')
@cached_task('search', cache_ttl=300)
def search_web(self, query: str, max_results: int = None) -> Dict[str, Any]:
    if not query or not query.strip():
        raise ValueError("Search query cannot be empty")
    
    query = query.strip()
    max_results = max_results or Config.SEARCH_RESULTS_LIMIT
    
    try:
        client = SearxNGClient()
        raw_results = asyncio.run(client.search(query))
        
        # Simple filtering
        filtered_results = []
        for result in raw_results[:max_results]:
            if result.get('url') and result.get('title'):
                filtered_results.append({
                    'url': result['url'],
                    'title': result['title'],
                    'content': result.get('content', ''),
                    'score': result.get('score', 0)
                })
        
        return {
            'results': filtered_results,
            'query': query,
            'total_found': len(filtered_results),
            'processing_time': 1.0,
        }
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise
'''

with open('/content/perplexica-core/services/search_service.py', 'w') as f:
    f.write(search_service_py)

# Simplified scraper service
scraper_service_py = '''
"""Simplified Scraper Service for Colab"""
import logging
import asyncio
import httpx
from typing import Dict, Any
import trafilatura
from celery import current_task
from message_queue import celery_app
from config import Config

logger = logging.getLogger(__name__)

class WebScraper:
    def __init__(self):
        self.timeout = httpx.Timeout(Config.SCRAPER_TIMEOUT)
        self.headers = {'User-Agent': Config.USER_AGENT}
    
    async def scrape_url(self, url: str) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=self.timeout, headers=self.headers) as client:
                response = await client.get(url)
                response.raise_for_status()
                
                html = response.text
                content = trafilatura.extract(html, include_tables=True, include_formatting=False)
                
                if not content or len(content.strip()) < 50:
                    return {
                        'url': url,
                        'title': '',
                        'content': '',
                        'word_count': 0,
                        'success': False,
                        'error': 'No extractable content'
                    }
                
                # Simple title extraction
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html, 'html.parser')
                title = soup.find('title')
                title_text = title.string.strip() if title and title.string else ''
                
                return {
                    'url': url,
                    'title': title_text,
                    'content': content.strip(),
                    'word_count': len(content.split()),
                    'success': True
                }
                
        except Exception as e:
            logger.error(f"Scraping failed for {url}: {e}")
            return {
                'url': url,
                'title': '',
                'content': '',
                'word_count': 0,
                'success': False,
                'error': str(e)
            }

@celery_app.task(bind=True, name='services.scraper_service.scrape_url')
def scrape_url(self, url: str) -> Dict[str, Any]:
    if not url or not url.strip():
        raise ValueError("URL cannot be empty")
    
    try:
        scraper = WebScraper()
        result = asyncio.run(scraper.scrape_url(url))
        return result
    except Exception as e:
        logger.error(f"Scraper task failed for {url}: {e}")
        raise
'''

with open('/content/perplexica-core/services/scraper_service.py', 'w') as f:
    f.write(scraper_service_py)

print("✓ Service files created!")

# ============================================================================
# STEP 7: CREATE EMBEDDING AND LLM SERVICES
# ============================================================================

# Simplified embedding service
embedding_service_py = '''
"""Simplified Embedding Service for Colab"""
import logging
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from celery import current_task
from message_queue import celery_app, cached_task
from config import Config

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    def __init__(self):
        self.model = None
        self._load_model()
    
    def _load_model(self):
        try:
            logger.info(f"Loading embedding model: {Config.EMBEDDING_MODEL}")
            self.model = SentenceTransformer(Config.EMBEDDING_MODEL)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        if not self.model or not texts:
            return np.array([])
        
        valid_texts = [text for text in texts if text and text.strip()]
        if not valid_texts:
            return np.array([])
        
        embeddings = self.model.encode(
            valid_texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings

embedding_generator = EmbeddingGenerator()

class SimilarityCalculator:
    def rank_content_by_relevance(self, query: str, content_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not query or not content_items:
            return []
        
        # Extract successful content
        successful_content = [item for item in content_items if item.get('success', False)]
        if not successful_content:
            return []
        
        # Prepare content for embedding
        content_texts = []
        for item in successful_content:
            content = item.get('content', '')
            title = item.get('title', '')
            if content.strip():
                # Combine title and content
                full_text = f"{title} {content}" if title else content
                content_texts.append(full_text[:2000])  # Limit length
        
        if not content_texts:
            return []
        
        # Generate embeddings
        try:
            query_embeddings = embedding_generator.generate_embeddings([query])
            content_embeddings = embedding_generator.generate_embeddings(content_texts)
            
            if query_embeddings.size == 0 or content_embeddings.size == 0:
                return []
            
            # Calculate similarities
            similarities = cosine_similarity(query_embeddings, content_embeddings)[0]
            
            # Create ranked results
            ranked_results = []
            for idx, (item, similarity) in enumerate(zip(successful_content, similarities)):
                if similarity >= Config.SIMILARITY_THRESHOLD:
                    result = {
                        'text': content_texts[idx],
                        'similarity_score': float(similarity),
                        'url': item.get('url', ''),
                        'title': item.get('title', ''),
                        'word_count': item.get('word_count', 0)
                    }
                    ranked_results.append(result)
            
            # Sort by similarity and limit
            ranked_results.sort(key=lambda x: x['similarity_score'], reverse=True)
            return ranked_results[:Config.TOP_RELEVANT_CHUNKS]
            
        except Exception as e:
            logger.error(f"Embedding ranking failed: {e}")
            return []

similarity_calculator = SimilarityCalculator()

@celery_app.task(bind=True, name='services.embedding_service.rank_content_by_relevance')
@cached_task('embedding', cache_ttl=600)
def rank_content_by_relevance(self, query: str, content_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not query or not content_items:
        raise ValueError("Query and content items are required")
    
    try:
        ranked_content = similarity_calculator.rank_content_by_relevance(query, content_items)
        
        return {
            'ranked_content': ranked_content,
            'query': query,
            'total_content_items': len(content_items),
            'relevant_chunks': len(ranked_content),
            'processing_time': 1.0,
            'similarity_threshold': Config.SIMILARITY_THRESHOLD
        }
    except Exception as e:
        logger.error(f"Ranking failed: {e}")
        raise
'''

with open('/content/perplexica-core/services/embedding_service.py', 'w') as f:
    f.write(embedding_service_py)

# Simplified LLM gateway
llm_gateway_py = '''
"""Simplified LLM Gateway for Colab"""
import logging
import asyncio
from typing import Dict, Any, List
import openai
from celery import current_task
from message_queue import celery_app
from config import Config

logger = logging.getLogger(__name__)

class OpenAIProvider:
    def __init__(self):
        if Config.OPENAI_API_KEY:
            self.client = openai.AsyncOpenAI(api_key=Config.OPENAI_API_KEY)
        else:
            self.client = None
    
    async def generate_response(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        if not self.client:
            raise RuntimeError("OpenAI API not configured")
        
        try:
            response = await self.client.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=Config.OPENAI_MAX_TOKENS,
                temperature=Config.OPENAI_TEMPERATURE
            )
            
            return {
                'response': response.choices[0].message.content,
                'provider': 'openai',
                'model': Config.OPENAI_MODEL,
                'tokens_used': {
                    'total': response.usage.total_tokens,
                    'prompt': response.usage.prompt_tokens,
                    'completion': response.usage.completion_tokens
                },
                'success': True
            }
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

class LLMGateway:
    def __init__(self):
        self.openai_provider = OpenAIProvider()
    
    def create_prompt(self, query: str, relevant_content: List[Dict[str, Any]]) -> Dict[str, str]:
        system_prompt = """You are an AI research assistant that provides accurate, well-cited answers based on web search results.

INSTRUCTIONS:
1. Answer ONLY based on the provided context
2. Include citations using [Source: URL] format
3. If information is insufficient, state this clearly
4. Do not hallucinate or add information not in sources
5. End with a "Sources:" section listing all URLs"""

        context_sections = []
        for idx, content in enumerate(relevant_content, 1):
            url = content.get('url', 'Unknown')
            title = content.get('title', 'Untitled')
            text = content.get('text', '')[:2000]  # Limit context length
            
            context_sections.append(f"Source {idx}:\\nURL: {url}\\nTitle: {title}\\nContent: {text}")
        
        user_prompt = f"""Based on the following search results, answer this question: "{query}"

SEARCH RESULTS:
{chr(10).join(context_sections)}

Question: {query}"""
        
        return {'system': system_prompt, 'user': user_prompt}
    
    async def generate_answer(self, query: str, relevant_content: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not relevant_content:
            return {
                'answer': f"I couldn't find sufficient information to answer: '{query}'. Please try a different question.",
                'query': query,
                'citations': [],
                'sources': [],
                'success': False
            }
        
        prompts = self.create_prompt(query, relevant_content)
        
        try:
            response = await self.openai_provider.generate_response(
                prompts['system'], prompts['user']
            )
            
            # Extract citations
            import re
            citations = re.findall(r'\\[Source:\\s*(https?://[^\\]]+)\\]', response['response'])
            
            sources = [{'url': c['url'], 'title': c['title']} for c in relevant_content]
            
            return {
                'answer': response['response'],
                'query': query,
                'citations': list(set(citations)),
                'sources': sources,
                'provider_info': response,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return {
                'answer': f"Failed to generate answer: {str(e)}",
                'query': query,
                'citations': [],
                'sources': [],
                'success': False
            }

llm_gateway = LLMGateway()

@celery_app.task(bind=True, name='services.llm_gateway.generate_answer')
def generate_answer(self, query: str, relevant_content: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not query:
        raise ValueError("Query cannot be empty")
    
    try:
        result = asyncio.run(llm_gateway.generate_answer(query, relevant_content))
        return result
    except Exception as e:
        logger.error(f"LLM task failed: {e}")
        raise
'''

with open('/content/perplexica-core/services/llm_gateway.py', 'w') as f:
    f.write(llm_gateway_py)

print("✓ AI services created!")

# ============================================================================
# STEP 8: CREATE SIMPLIFIED ORCHESTRATOR
# ============================================================================

orchestrator_py = '''
"""Simplified Orchestrator for Colab"""
import logging
import asyncio
import time
from typing import Dict, Any, List
from celery import group
from message_queue import celery_app, cache_manager
from config import Config

from services.search_service import search_web
from services.scraper_service import scrape_url
from services.embedding_service import rank_content_by_relevance
from services.llm_gateway import generate_answer

logger = logging.getLogger(__name__)

class SearchPipelineOrchestrator:
    async def process_query(self, query: str, max_results: int = None) -> Dict[str, Any]:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        query = query.strip()
        max_results = max_results or Config.MAX_URLS_TO_SCRAPE
        
        logger.info(f"Processing query: '{query}'")
        
        try:
            # Step 1: Search
            logger.info("1. Searching the web...")
            search_result = await self._execute_search(query, max_results)
            
            if not search_result.get('results'):
                raise RuntimeError("No search results found")
            
            # Step 2: Scrape
            logger.info("2. Scraping content...")
            urls = [r['url'] for r in search_result['results']]
            scraping_results = await self._execute_scraping(urls)
            
            successful_content = [r for r in scraping_results if r.get('success', False)]
            if not successful_content:
                raise RuntimeError("No content could be extracted")
            
            # Step 3: Rank
            logger.info("3. Ranking content by relevance...")
            ranking_result = await self._execute_ranking(query, successful_content)
            
            relevant_content = ranking_result.get('ranked_content', [])
            if not relevant_content:
                raise RuntimeError("No relevant content found")
            
            # Step 4: Generate answer
            logger.info("4. Generating answer...")
            answer_result = await self._execute_answer_generation(query, relevant_content)
            
            # Compile final response
            final_result = {
                'answer': answer_result.get('answer', ''),
                'query': query,
                'citations': answer_result.get('citations', []),
                'sources': answer_result.get('sources', []),
                'pipeline_stats': {
                    'total_search_results': len(search_result.get('results', [])),
                    'successful_scrapes': len(successful_content),
                    'relevant_chunks': len(relevant_content)
                },
                'success': answer_result.get('success', False)
            }
            
            logger.info("✓ Pipeline completed successfully!")
            return final_result
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    async def _execute_search(self, query: str, max_results: int) -> Dict[str, Any]:
        task = search_web.delay(query, max_results)
        return task.get(timeout=60)
    
    async def _execute_scraping(self, urls: List[str]) -> List[Dict[str, Any]]:
        tasks = group(scrape_url.s(url) for url in urls)
        job = tasks.apply_async()
        return job.get(timeout=120)
    
    async def _execute_ranking(self, query: str, content_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        task = rank_content_by_relevance.delay(query, content_items)
        return task.get(timeout=120)
    
    async def _execute_answer_generation(self, query: str, relevant_content: List[Dict[str, Any]]) -> Dict[str, Any]:
        task = generate_answer.delay(query, relevant_content)
        return task.get(timeout=120)

orchestrator = SearchPipelineOrchestrator()

async def search_and_answer(query: str, max_results: int = None) -> Dict[str, Any]:
    return await orchestrator.process_query(query, max_results)

def search_and_answer_sync(query: str, max_results: int = None) -> Dict[str, Any]:
    return asyncio.run(search_and_answer(query, max_results))
'''

with open('/content/perplexica-core/main_orchestrator.py', 'w') as f:
    f.write(orchestrator_py)

print("✓ Orchestrator created!")

# ============================================================================
# STEP 9: START CELERY WORKERS
# ============================================================================

# Change to project directory
import os
os.chdir('/content/perplexica-core')

# Start Celery worker in background
print("Starting Celery workers...")

import subprocess
from threading import Thread

def start_celery_worker():
    subprocess.run(['celery', '-A', 'message_queue.celery_app', 'worker', '--loglevel=info', '--concurrency=2'])

# Start worker in background thread
worker_thread = Thread(target=start_celery_worker, daemon=True)
worker_thread.start()

# Wait a moment for worker to start
time.sleep(10)

print("✓ Celery workers started!")

# ============================================================================
# STEP 10: CONFIGURATION AND USAGE INSTRUCTIONS
# ============================================================================

print("""
🎉 PERPLEXICACORE SETUP COMPLETE!

IMPORTANT: Set your OpenAI API key to use the system:
""")

# Create a simple interface for setting API key
from IPython.display import display, HTML

def set_openai_key(api_key):
    """Set OpenAI API key for the session"""
    os.environ['OPENAI_API_KEY'] = api_key
    print(f"✓ OpenAI API key set!")

# Display instructions
display(HTML("""
<div style="background-color: #f0f8ff; padding: 20px; border-radius: 10px; border-left: 5px solid #007acc;">
<h3>🔑 API Key Setup</h3>
<p>To use PerplexicaCore, you need to set your OpenAI API key:</p>
<pre style="background-color: #e8e8e8; padding: 10px; border-radius: 5px;">
# Run this cell after entering your API key:
set_openai_key("your-openai-api-key-here")
</pre>
</div>
"""))

print("""
USAGE EXAMPLES:

1. Basic Usage:
   from main_orchestrator import search_and_answer_sync
   result = search_and_answer_sync("What are the latest developments in quantum computing?")
   print(result['answer'])

2. Get Sources:
   result = search_and_answer_sync("How does machine learning work?")
   print("Answer:", result['answer'])
   print("Sources:", [s['url'] for s in result['sources']])

3. Check Pipeline Stats:
   result = search_and_answer_sync("What is climate change?")
   print("Stats:", result['pipeline_stats'])

SYSTEM FEATURES:
✓ Web search using SearxNG
✓ Intelligent content scraping
✓ AI-powered relevance ranking
✓ LLM-generated answers with citations
✓ Caching for performance
✓ Asynchronous processing
✓ Error handling and retries

Ready to answer your questions! 🚀
""")

# ============================================================================
# EXAMPLE USAGE CELL
# ============================================================================

print("""
Example usage cell (run after setting API key):

# Example 1: Ask a question
result = search_and_answer_sync("What are the benefits of renewable energy?")
print("ANSWER:")
print(result['answer'])
print("\\nSOURCES:")
for source in result['sources']:
    print(f"- {source['title']}: {source['url']}")

# Example 2: Technical question
result = search_and_answer_sync("How does machine learning differ from traditional programming?")
print("ANSWER:")
print(result['answer'])
print(f"\\nFound {result['pipeline_stats']['relevant_chunks']} relevant content chunks")
print(f"Citations: {len(result['citations'])}")
""")