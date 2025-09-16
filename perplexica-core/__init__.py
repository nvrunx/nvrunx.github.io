"""
PerplexicaCore AI-Augmented Search Engine
=========================================

A powerful, scalable, and accurate AI-augmented search engine that answers user 
queries by searching the web in real-time, processing results with advanced AI, 
and generating well-cited, accurate answers.

Main Components:
- Search Service: SearxNG integration for web search
- Scraper Service: Ethical web content extraction
- Embedding Service: AI-powered relevance ranking
- LLM Gateway: Multi-provider language model integration
- Orchestrator: Pipeline coordination and management

Quick Start:
    from main_orchestrator import search_and_answer_sync
    
    result = search_and_answer_sync("What is machine learning?")
    print(result['answer'])

For detailed setup instructions, see README.md and COLAB_SETUP.py
"""

__version__ = "1.0.0"
__author__ = "PerplexicaCore Team"
__email__ = "contact@perplexicacore.com"
__license__ = "MIT"

# Main API exports
from main_orchestrator import (
    search_and_answer,
    search_and_answer_sync,
    get_pipeline_health
)

__all__ = [
    'search_and_answer',
    'search_and_answer_sync', 
    'get_pipeline_health'
]