"""
Services Package for PerplexicaCore
===================================

This package contains all the microservices that make up the PerplexicaCore
AI-augmented search engine pipeline.

Services:
- search_service: Web search using SearxNG
- scraper_service: Ethical web content extraction  
- embedding_service: AI-powered content ranking
- llm_gateway: Multi-provider LLM integration

Each service is implemented as a Celery task for distributed processing.
"""

# Service task imports for easy access
from .search_service import search_web
from .scraper_service import scrape_url, scrape_urls_batch
from .embedding_service import rank_content_by_relevance
from .llm_gateway import generate_answer

__all__ = [
    'search_web',
    'scrape_url',
    'scrape_urls_batch', 
    'rank_content_by_relevance',
    'generate_answer'
]