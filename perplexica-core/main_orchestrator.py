"""
Main Orchestrator for PerplexicaCore
====================================

This is the central coordination service that orchestrates the entire AI-augmented
search pipeline. It coordinates between all microservices to provide a unified
interface for answering user queries with web-based research.

Key responsibilities:
1. Receive and validate user queries
2. Coordinate search, scraping, embedding, and LLM services
3. Manage parallel processing and error handling
4. Provide real-time progress updates
5. Return comprehensive, well-cited answers

The orchestrator implements intelligent task coordination with proper error handling,
retry logic, and performance monitoring throughout the pipeline.
"""

import logging
import asyncio
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from celery import group, chain
from celery.result import GroupResult
from message_queue import celery_app, cache_manager
from config import Config

# Import all service tasks
from services.search_service import search_web
from services.scraper_service import scrape_url, scrape_urls_batch
from services.embedding_service import rank_content_by_relevance
from services.llm_gateway import generate_answer

# Configure logging for this module
logger = logging.getLogger(__name__)

# ============================================================================
# PIPELINE STATUS AND PROGRESS TRACKING
# ============================================================================

class PipelineStage(Enum):
    """Enumeration of pipeline stages for progress tracking."""
    INITIALIZING = "initializing"
    SEARCHING = "searching"
    SCRAPING = "scraping"
    RANKING = "ranking"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PipelineProgress:
    """Data class for tracking pipeline execution progress."""
    stage: PipelineStage
    progress_percent: float
    message: str
    current_step: str
    total_steps: int
    completed_steps: int
    start_time: float
    stage_start_time: float
    errors: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert progress to dictionary for JSON serialization."""
        return {
            'stage': self.stage.value,
            'progress_percent': round(self.progress_percent, 1),
            'message': self.message,
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'completed_steps': self.completed_steps,
            'elapsed_time': round(time.time() - self.start_time, 2),
            'stage_time': round(time.time() - self.stage_start_time, 2),
            'errors': self.errors
        }


class ProgressTracker:
    """
    Real-time progress tracking for pipeline execution.
    
    This class provides detailed progress information that can be used
    for real-time updates in user interfaces or monitoring systems.
    """
    
    def __init__(self, query: str, session_id: str = None):
        """
        Initialize progress tracker.
        
        Args:
            query: User query being processed
            session_id: Optional session identifier for tracking
        """
        self.query = query
        self.session_id = session_id or f"session_{int(time.time())}"
        self.start_time = time.time()
        self.current_progress = PipelineProgress(
            stage=PipelineStage.INITIALIZING,
            progress_percent=0.0,
            message="Initializing search pipeline...",
            current_step="Setup",
            total_steps=5,  # Search, Scrape, Rank, Generate, Complete
            completed_steps=0,
            start_time=self.start_time,
            stage_start_time=self.start_time,
            errors=[]
        )
    
    def update_stage(
        self, 
        stage: PipelineStage, 
        message: str, 
        step_name: str = None
    ):
        """
        Update current pipeline stage.
        
        Args:
            stage: New pipeline stage
            message: Progress message
            step_name: Optional step name override
        """
        current_time = time.time()
        
        # Calculate progress percentage based on stage
        stage_progress = {
            PipelineStage.INITIALIZING: 0,
            PipelineStage.SEARCHING: 20,
            PipelineStage.SCRAPING: 40,
            PipelineStage.RANKING: 70,
            PipelineStage.GENERATING: 85,
            PipelineStage.COMPLETED: 100,
            PipelineStage.FAILED: 0
        }
        
        # Update completed steps
        if stage != self.current_progress.stage:
            self.current_progress.completed_steps = min(
                self.current_progress.completed_steps + 1,
                self.current_progress.total_steps
            )
        
        self.current_progress.stage = stage
        self.current_progress.progress_percent = stage_progress.get(stage, 0)
        self.current_progress.message = message
        self.current_progress.current_step = step_name or stage.value.title()
        self.current_progress.stage_start_time = current_time
        
        logger.info(f"Pipeline {self.session_id}: {stage.value} - {message}")
    
    def add_error(self, error_message: str):
        """Add error to progress tracking."""
        self.current_progress.errors.append(error_message)
        logger.error(f"Pipeline {self.session_id}: Error - {error_message}")
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current progress as dictionary."""
        return self.current_progress.to_dict()


# ============================================================================
# PIPELINE ORCHESTRATOR
# ============================================================================

class SearchPipelineOrchestrator:
    """
    Main orchestrator for the AI-augmented search pipeline.
    
    This class coordinates all microservices to process user queries through
    the complete pipeline: search → scrape → rank → generate answer.
    """
    
    def __init__(self):
        """Initialize the orchestrator with default settings."""
        self.default_timeout = 300  # 5 minutes total pipeline timeout
    
    async def process_query(
        self, 
        query: str,
        max_results: int = None,
        progress_callback: Callable[[Dict[str, Any]], None] = None,
        session_id: str = None
    ) -> Dict[str, Any]:
        """
        Process a user query through the complete AI search pipeline.
        
        This is the main entry point that orchestrates all services to provide
        a comprehensive answer to the user's question.
        
        Args:
            query: User's search query
            max_results: Maximum search results to process
            progress_callback: Optional callback for progress updates
            session_id: Optional session identifier
            
        Returns:
            Dictionary containing the final answer and metadata
            
        Raises:
            ValueError: For invalid input parameters
            RuntimeError: For pipeline execution failures
        """
        
        # Validate inputs
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        query = query.strip()
        
        if max_results is None:
            max_results = Config.MAX_URLS_TO_SCRAPE
        
        # Initialize progress tracking
        progress = ProgressTracker(query, session_id)
        
        def update_progress(stage: PipelineStage, message: str, step: str = None):
            progress.update_stage(stage, message, step)
            if progress_callback:
                progress_callback(progress.get_progress())
        
        try:
            logger.info(f"Starting search pipeline for query: '{query}'")
            
            # Check cache first
            cache_key = {'query': query, 'max_results': max_results}
            cached_result = cache_manager.get('full_pipeline', cache_key)
            
            if cached_result:
                logger.info(f"Returning cached result for query: '{query}'")
                update_progress(PipelineStage.COMPLETED, "Retrieved from cache")
                cached_result['from_cache'] = True
                return cached_result
            
            # Stage 1: Web Search
            update_progress(PipelineStage.SEARCHING, "Searching the web...")
            
            search_result = await self._execute_search(query, max_results)
            
            if not search_result.get('results'):
                raise RuntimeError("No search results found")
            
            search_urls = [result['url'] for result in search_result['results']]
            
            update_progress(
                PipelineStage.SCRAPING, 
                f"Scraping content from {len(search_urls)} websites..."
            )
            
            # Stage 2: Parallel Web Scraping
            scraping_results = await self._execute_scraping(search_urls)
            
            # Filter successful scraping results
            successful_content = [
                result for result in scraping_results 
                if result.get('success', False) and result.get('content', '').strip()
            ]
            
            if not successful_content:
                raise RuntimeError("No content could be extracted from search results")
            
            update_progress(
                PipelineStage.RANKING, 
                f"Analyzing relevance of {len(successful_content)} content pieces..."
            )
            
            # Stage 3: AI-Powered Content Ranking
            ranking_result = await self._execute_ranking(query, successful_content)
            
            relevant_content = ranking_result.get('ranked_content', [])
            
            if not relevant_content:
                raise RuntimeError("No relevant content found for the query")
            
            update_progress(
                PipelineStage.GENERATING, 
                f"Generating answer using {len(relevant_content)} relevant sources..."
            )
            
            # Stage 4: LLM Answer Generation
            answer_result = await self._execute_answer_generation(query, relevant_content)
            
            # Stage 5: Compile Final Response
            update_progress(PipelineStage.COMPLETED, "Answer generated successfully!")
            
            final_result = self._compile_final_response(
                query, search_result, scraping_results, ranking_result, answer_result
            )
            
            # Cache the final result
            cache_manager.set('full_pipeline', cache_key, final_result, ttl=Config.CACHE_TTL)
            
            logger.info(f"Pipeline completed successfully for query: '{query}'")
            
            return final_result
            
        except Exception as e:
            error_message = f"Pipeline failed: {str(e)}"
            progress.add_error(error_message)
            update_progress(PipelineStage.FAILED, error_message)
            
            logger.error(f"Pipeline failed for query '{query}': {e}")
            raise RuntimeError(error_message) from e
    
    async def _execute_search(self, query: str, max_results: int) -> Dict[str, Any]:
        """Execute web search stage."""
        try:
            # Submit search task
            search_task = search_web.delay(query, max_results)
            
            # Wait for completion with timeout
            result = search_task.get(timeout=60)  # 1 minute timeout for search
            
            logger.debug(f"Search completed: found {len(result.get('results', []))} results")
            
            return result
            
        except Exception as e:
            logger.error(f"Search stage failed: {e}")
            raise RuntimeError(f"Web search failed: {e}")
    
    async def _execute_scraping(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Execute parallel web scraping stage."""
        try:
            # Create parallel scraping tasks
            scraping_tasks = group(scrape_url.s(url) for url in urls)
            
            # Submit all tasks
            job = scraping_tasks.apply_async()
            
            # Wait for completion with timeout
            results = job.get(timeout=120)  # 2 minutes timeout for scraping
            
            successful_count = sum(1 for r in results if r.get('success', False))
            
            logger.debug(
                f"Scraping completed: {successful_count}/{len(urls)} successful"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Scraping stage failed: {e}")
            raise RuntimeError(f"Content scraping failed: {e}")
    
    async def _execute_ranking(
        self, 
        query: str, 
        content_items: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute AI content ranking stage."""
        try:
            # Submit ranking task
            ranking_task = rank_content_by_relevance.delay(query, content_items)
            
            # Wait for completion with timeout
            result = ranking_task.get(timeout=180)  # 3 minutes timeout for ranking
            
            relevant_count = len(result.get('ranked_content', []))
            
            logger.debug(f"Ranking completed: found {relevant_count} relevant chunks")
            
            return result
            
        except Exception as e:
            logger.error(f"Ranking stage failed: {e}")
            raise RuntimeError(f"Content ranking failed: {e}")
    
    async def _execute_answer_generation(
        self, 
        query: str, 
        relevant_content: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute LLM answer generation stage."""
        try:
            # Submit answer generation task
            answer_task = generate_answer.delay(query, relevant_content)
            
            # Wait for completion with timeout
            result = answer_task.get(timeout=180)  # 3 minutes timeout for LLM
            
            logger.debug(f"Answer generation completed")
            
            return result
            
        except Exception as e:
            logger.error(f"Answer generation stage failed: {e}")
            raise RuntimeError(f"Answer generation failed: {e}")
    
    def _compile_final_response(
        self,
        query: str,
        search_result: Dict[str, Any],
        scraping_results: List[Dict[str, Any]],
        ranking_result: Dict[str, Any],
        answer_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compile comprehensive final response from all pipeline stages.
        
        Args:
            query: Original query
            search_result: Search stage results
            scraping_results: Scraping stage results
            ranking_result: Ranking stage results
            answer_result: Answer generation results
            
        Returns:
            Comprehensive response dictionary
        """
        
        # Calculate pipeline statistics
        total_urls_found = len(search_result.get('results', []))
        total_scraped = len(scraping_results)
        successful_scrapes = sum(1 for r in scraping_results if r.get('success', False))
        relevant_chunks = len(ranking_result.get('ranked_content', []))
        
        # Compile source information
        sources = []
        for result in scraping_results:
            if result.get('success', False):
                source_info = {
                    'url': result.get('url', ''),
                    'title': result.get('title', ''),
                    'word_count': result.get('word_count', 0),
                    'successfully_scraped': True
                }
                sources.append(source_info)
        
        # Build comprehensive response
        final_response = {
            # Main answer content
            'answer': answer_result.get('answer', ''),
            'query': query,
            'citations': answer_result.get('citations', []),
            
            # Pipeline execution metadata
            'pipeline_stats': {
                'total_search_results': total_urls_found,
                'urls_scraped': total_scraped,
                'successful_scrapes': successful_scrapes,
                'relevant_content_chunks': relevant_chunks,
                'total_processing_time': (
                    search_result.get('processing_time', 0) +
                    ranking_result.get('processing_time', 0) +
                    answer_result.get('processing_time', 0)
                )
            },
            
            # Source information
            'sources': sources,
            'relevant_content_summary': [
                {
                    'url': chunk.get('url', ''),
                    'title': chunk.get('title', ''),
                    'similarity_score': chunk.get('similarity_score', 0),
                    'word_count': chunk.get('word_count', 0)
                }
                for chunk in ranking_result.get('ranked_content', [])
            ],
            
            # Quality and provider information
            'quality_metrics': answer_result.get('quality_metrics', {}),
            'provider_info': answer_result.get('provider_info', {}),
            
            # Stage-specific results (for debugging/analysis)
            'detailed_results': {
                'search': {
                    'found_results': total_urls_found,
                    'processing_time': search_result.get('processing_time', 0)
                },
                'scraping': {
                    'attempted': total_scraped,
                    'successful': successful_scrapes,
                    'success_rate': round(successful_scrapes / max(total_scraped, 1), 2)
                },
                'ranking': {
                    'chunks_analyzed': ranking_result.get('total_content_items', 0),
                    'relevant_found': relevant_chunks,
                    'similarity_threshold': ranking_result.get('similarity_threshold', 0),
                    'processing_time': ranking_result.get('processing_time', 0)
                },
                'generation': {
                    'provider': answer_result.get('provider_info', {}).get('provider', 'unknown'),
                    'model': answer_result.get('provider_info', {}).get('model', 'unknown'),
                    'tokens_used': answer_result.get('provider_info', {}).get('tokens_used', {}),
                    'processing_time': answer_result.get('processing_time', 0)
                }
            },
            
            # Metadata
            'timestamp': time.time(),
            'from_cache': False,
            'version': '1.0'
        }
        
        return final_response


# ============================================================================
# SIMPLIFIED INTERFACE FUNCTIONS
# ============================================================================

# Global orchestrator instance
orchestrator = SearchPipelineOrchestrator()

async def search_and_answer(
    query: str,
    max_results: int = None,
    progress_callback: Callable[[Dict[str, Any]], None] = None,
    session_id: str = None
) -> Dict[str, Any]:
    """
    Main interface function for processing search queries.
    
    This is the primary entry point for the PerplexicaCore system.
    
    Args:
        query: User's search query
        max_results: Maximum search results to process
        progress_callback: Optional callback for progress updates
        session_id: Optional session identifier
        
    Returns:
        Comprehensive answer with sources and metadata
    """
    
    return await orchestrator.process_query(
        query=query,
        max_results=max_results,
        progress_callback=progress_callback,
        session_id=session_id
    )


def search_and_answer_sync(
    query: str,
    max_results: int = None,
    progress_callback: Callable[[Dict[str, Any]], None] = None,
    session_id: str = None
) -> Dict[str, Any]:
    """
    Synchronous wrapper for search and answer functionality.
    
    Args:
        query: User's search query
        max_results: Maximum search results to process
        progress_callback: Optional callback for progress updates
        session_id: Optional session identifier
        
    Returns:
        Comprehensive answer with sources and metadata
    """
    
    return asyncio.run(search_and_answer(
        query=query,
        max_results=max_results,
        progress_callback=progress_callback,
        session_id=session_id
    ))


# ============================================================================
# HEALTH CHECK AND MONITORING
# ============================================================================

def get_pipeline_health() -> Dict[str, Any]:
    """
    Check health status of the entire pipeline.
    
    Returns:
        Dictionary with health status of all components
    """
    
    from message_queue import get_system_health
    
    # Get basic system health
    health = get_system_health()
    
    # Add pipeline-specific health checks
    health['pipeline'] = {
        'orchestrator_ready': True,
        'config_valid': True,
        'services_available': {
            'search': True,
            'scraper': True,
            'embedding': True,
            'llm': True
        }
    }
    
    # Check individual service health
    try:
        # Test if services are importable and functional
        from services import search_service, scraper_service, embedding_service, llm_gateway
        
        health['pipeline']['services_available']['search'] = hasattr(search_service, 'search_web')
        health['pipeline']['services_available']['scraper'] = hasattr(scraper_service, 'scrape_url')
        health['pipeline']['services_available']['embedding'] = hasattr(embedding_service, 'rank_content_by_relevance')
        health['pipeline']['services_available']['llm'] = hasattr(llm_gateway, 'generate_answer')
        
    except ImportError as e:
        health['pipeline']['services_available'] = {
            'search': False,
            'scraper': False,
            'embedding': False,
            'llm': False
        }
        health['pipeline']['import_error'] = str(e)
    
    # Overall pipeline health
    all_services_ok = all(health['pipeline']['services_available'].values())
    health['pipeline']['overall_status'] = 'healthy' if all_services_ok else 'degraded'
    
    return health


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """
    Command line interface for testing the pipeline.
    
    This function provides a simple CLI for testing and demonstrating
    the PerplexicaCore functionality.
    """
    
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='PerplexicaCore AI Search Engine')
    parser.add_argument('query', help='Search query to process')
    parser.add_argument('--max-results', type=int, default=None, 
                       help='Maximum search results to process')
    parser.add_argument('--output', choices=['json', 'text'], default='text',
                       help='Output format')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--health-check', action='store_true',
                       help='Perform system health check')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # Health check mode
    if args.health_check:
        health = get_pipeline_health()
        if args.output == 'json':
            print(json.dumps(health, indent=2))
        else:
            print(f"Pipeline Health: {health['pipeline']['overall_status']}")
            print(f"Redis: {'OK' if health['redis'] else 'FAILED'}")
            print(f"Celery: {'OK' if health['celery'] else 'FAILED'}")
        return
    
    # Process query
    def progress_callback(progress_data):
        if args.verbose:
            print(f"Progress: {progress_data['message']} ({progress_data['progress_percent']:.1f}%)")
    
    try:
        print(f"Processing query: {args.query}")
        
        result = search_and_answer_sync(
            query=args.query,
            max_results=args.max_results,
            progress_callback=progress_callback if args.verbose else None
        )
        
        if args.output == 'json':
            print(json.dumps(result, indent=2))
        else:
            print("\nANSWER:")
            print("=" * 50)
            print(result['answer'])
            print("\nSOURCES:")
            print("=" * 50)
            for source in result['sources']:
                print(f"- {source['title']}: {source['url']}")
            
            stats = result['pipeline_stats']
            print(f"\nSTATISTICS:")
            print("=" * 50)
            print(f"Search results: {stats['total_search_results']}")
            print(f"Successful scrapes: {stats['successful_scrapes']}")
            print(f"Relevant chunks: {stats['relevant_content_chunks']}")
            print(f"Processing time: {stats['total_processing_time']:.2f}s")
            
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        exit(1)


# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

logger.info("Main orchestrator initialized")
logger.info(f"Default pipeline timeout: {orchestrator.default_timeout}s")

if __name__ == '__main__':
    main()