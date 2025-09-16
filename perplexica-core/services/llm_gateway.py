"""
LLM Gateway Service for PerplexicaCore
======================================

This service provides a unified interface to multiple Large Language Model (LLM)
backends for generating final responses. It supports both cloud-based APIs (OpenAI)
and local deployments (Ollama) with intelligent prompt engineering for accurate,
well-cited responses.

Key features:
- Multi-provider LLM support (OpenAI, Ollama)
- Sophisticated prompt engineering for factual accuracy
- Citation tracking and source attribution
- Response quality validation
- Configurable model parameters
- Robust error handling and fallbacks
"""

import logging
import asyncio
import json
from typing import Dict, Any, List, Optional
from urllib.parse import urljoin
import httpx
import openai
from celery import current_task
from message_queue import celery_app
from config import Config

# Configure logging for this module
logger = logging.getLogger(__name__)

# ============================================================================
# PROMPT ENGINEERING UTILITIES
# ============================================================================

class PromptEngineer:
    """
    Advanced prompt engineering for factual and well-cited responses.
    
    This class creates sophisticated prompts that instruct the LLM to:
    1. Answer only based on provided context
    2. Include proper citations for all claims
    3. Acknowledge when information is insufficient
    4. Maintain factual accuracy and avoid hallucination
    """
    
    def __init__(self):
        """Initialize prompt engineer with templates."""
        # Base system prompt that sets behavior expectations
        self.system_prompt = """You are PerplexicaCore, an AI research assistant that provides accurate, well-cited answers based on web search results.

CRITICAL INSTRUCTIONS:
1. Answer ONLY based on the provided context from web search results
2. Include citations using [Source: URL] format for ALL factual claims
3. If the context doesn't contain sufficient information, explicitly state this
4. Do not hallucinate or add information not present in the sources
5. Maintain objectivity and present multiple perspectives when available
6. Structure your response clearly with proper paragraphs
7. Prioritize recent and authoritative sources when available

FORMAT REQUIREMENTS:
- Use clear, professional language
- Include [Source: URL] after each factual claim
- End with a "Sources:" section listing all referenced URLs
- If uncertain about any information, clearly indicate this"""

    def create_answer_prompt(
        self, 
        query: str, 
        relevant_content: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Create a comprehensive prompt for answering a user query.
        
        Args:
            query: User's original question
            relevant_content: List of relevant content chunks with metadata
            
        Returns:
            Dictionary with 'system' and 'user' prompts
        """
        
        # Build context from relevant content
        context_sections = []
        source_urls = set()
        
        for idx, content in enumerate(relevant_content, 1):
            url = content.get('url', 'Unknown')
            title = content.get('title', 'Untitled')
            text = content.get('text', '')
            similarity = content.get('similarity_score', 0)
            
            if text.strip():
                # Clean and format the text
                clean_text = self._clean_context_text(text)
                
                context_section = f"""--- Source {idx} ---
URL: {url}
Title: {title}
Relevance Score: {similarity:.3f}
Content: {clean_text}
"""
                context_sections.append(context_section)
                source_urls.add(url)
        
        # Combine all context
        full_context = "\n".join(context_sections)
        
        # Create user prompt with query and context
        user_prompt = f"""Based on the following web search results, please answer this question: "{query}"

SEARCH RESULTS CONTEXT:
{full_context}

INSTRUCTIONS:
- Answer the question using ONLY the information provided in the search results above
- Cite your sources using [Source: URL] format after each factual claim
- If the search results don't contain enough information to fully answer the question, clearly state what is missing
- Provide a comprehensive answer that synthesizes information from multiple sources when possible
- End your response with a "Sources:" section listing all URLs you referenced

QUESTION TO ANSWER: {query}
"""
        
        return {
            'system': self.system_prompt,
            'user': user_prompt
        }
    
    def _clean_context_text(self, text: str) -> str:
        """
        Clean context text for optimal LLM processing.
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text suitable for LLM context
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Limit context length to prevent token overflow
        max_context_chars = 4000  # Conservative limit for context
        if len(text) > max_context_chars:
            text = text[:max_context_chars] + "..."
        
        return text
    
    def extract_citations(self, response_text: str) -> List[str]:
        """
        Extract citation URLs from LLM response.
        
        Args:
            response_text: Generated response text
            
        Returns:
            List of unique URLs found in citations
        """
        import re
        
        # Pattern to match [Source: URL] citations
        citation_pattern = r'\[Source:\s*(https?://[^\]]+)\]'
        
        # Find all citations
        citations = re.findall(citation_pattern, response_text, re.IGNORECASE)
        
        # Return unique URLs
        return list(set(citations))


# ============================================================================
# OPENAI LLM PROVIDER
# ============================================================================

class OpenAIProvider:
    """
    OpenAI API provider for cloud-based LLM access.
    
    This provider handles OpenAI API communication with proper error handling,
    rate limiting awareness, and response validation.
    """
    
    def __init__(self):
        """Initialize OpenAI provider with API configuration."""
        if not Config.OPENAI_API_KEY:
            logger.error("OpenAI API key not configured")
            self.client = None
        else:
            self.client = openai.AsyncOpenAI(api_key=Config.OPENAI_API_KEY)
            logger.info(f"OpenAI provider initialized with model: {Config.OPENAI_MODEL}")
    
    async def generate_response(
        self, 
        system_prompt: str, 
        user_prompt: str
    ) -> Dict[str, Any]:
        """
        Generate response using OpenAI API.
        
        Args:
            system_prompt: System instruction prompt
            user_prompt: User query and context prompt
            
        Returns:
            Dictionary with response and metadata
            
        Raises:
            RuntimeError: If API is not configured or request fails
        """
        
        if not self.client:
            raise RuntimeError("OpenAI API not configured")
        
        try:
            logger.debug("Sending request to OpenAI API")
            
            # Prepare messages for chat completion
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Make API request with configured parameters
            response = await self.client.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=messages,
                max_tokens=Config.OPENAI_MAX_TOKENS,
                temperature=Config.OPENAI_TEMPERATURE,
                top_p=0.9,  # Nucleus sampling for quality
                frequency_penalty=0.1,  # Reduce repetition
                presence_penalty=0.1    # Encourage diverse topics
            )
            
            # Extract response content
            response_text = response.choices[0].message.content
            
            # Calculate token usage
            usage = response.usage
            
            result = {
                'response': response_text,
                'provider': 'openai',
                'model': Config.OPENAI_MODEL,
                'tokens_used': {
                    'prompt': usage.prompt_tokens,
                    'completion': usage.completion_tokens,
                    'total': usage.total_tokens
                },
                'finish_reason': response.choices[0].finish_reason,
                'success': True
            }
            
            logger.info(f"OpenAI response generated: {usage.total_tokens} tokens used")
            
            return result
            
        except openai.RateLimitError as e:
            logger.error(f"OpenAI rate limit exceeded: {e}")
            raise RuntimeError(f"OpenAI rate limit exceeded: {e}")
            
        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise RuntimeError(f"OpenAI API error: {e}")
            
        except Exception as e:
            logger.error(f"Unexpected OpenAI error: {e}")
            raise RuntimeError(f"OpenAI request failed: {e}")


# ============================================================================
# OLLAMA LLM PROVIDER
# ============================================================================

class OllamaProvider:
    """
    Ollama provider for local LLM deployment.
    
    This provider interfaces with Ollama for running LLMs locally, providing
    privacy and cost benefits while maintaining performance.
    """
    
    def __init__(self):
        """Initialize Ollama provider with connection settings."""
        self.base_url = Config.OLLAMA_URL
        self.model = Config.OLLAMA_MODEL
        self.timeout = Config.OLLAMA_TIMEOUT
        
        logger.info(f"Ollama provider initialized: {self.base_url} with model {self.model}")
    
    async def generate_response(
        self, 
        system_prompt: str, 
        user_prompt: str
    ) -> Dict[str, Any]:
        """
        Generate response using Ollama local API.
        
        Args:
            system_prompt: System instruction prompt
            user_prompt: User query and context prompt
            
        Returns:
            Dictionary with response and metadata
            
        Raises:
            RuntimeError: If Ollama is not accessible or request fails
        """
        
        try:
            logger.debug("Sending request to Ollama")
            
            # Combine system and user prompts for Ollama
            # Ollama typically works better with combined prompts
            combined_prompt = f"""SYSTEM: {system_prompt}

USER: {user_prompt}

ASSISTANT:"""
            
            # Prepare request payload
            payload = {
                'model': self.model,
                'prompt': combined_prompt,
                'stream': False,  # Get complete response
                'options': {
                    'temperature': 0.1,    # Low temperature for factual responses
                    'top_p': 0.9,         # Nucleus sampling
                    'top_k': 40,          # Limit vocabulary sampling
                    'repeat_penalty': 1.1  # Reduce repetition
                }
            }
            
            # Make HTTP request to Ollama API
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    urljoin(self.base_url, '/api/generate'),
                    json=payload,
                    headers={'Content-Type': 'application/json'}
                )
                
                response.raise_for_status()
                
                # Parse response
                result_data = response.json()
                
                response_text = result_data.get('response', '')
                
                if not response_text:
                    raise RuntimeError("Ollama returned empty response")
                
                result = {
                    'response': response_text,
                    'provider': 'ollama',
                    'model': self.model,
                    'tokens_used': {
                        'prompt': result_data.get('prompt_eval_count', 0),
                        'completion': result_data.get('eval_count', 0),
                        'total': (result_data.get('prompt_eval_count', 0) + 
                                result_data.get('eval_count', 0))
                    },
                    'eval_duration': result_data.get('eval_duration', 0),
                    'success': True
                }
                
                logger.info(f"Ollama response generated: {result['tokens_used']['total']} tokens")
                
                return result
                
        except httpx.TimeoutException as e:
            logger.error(f"Ollama timeout: {e}")
            raise RuntimeError(f"Ollama request timeout: {e}")
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama HTTP error: {e.response.status_code}")
            raise RuntimeError(f"Ollama HTTP error: {e.response.status_code}")
            
        except Exception as e:
            logger.error(f"Unexpected Ollama error: {e}")
            raise RuntimeError(f"Ollama request failed: {e}")


# ============================================================================
# LLM GATEWAY COORDINATOR
# ============================================================================

class LLMGateway:
    """
    Main LLM gateway that coordinates between different providers.
    
    This class provides a unified interface for LLM operations while
    handling provider selection, fallbacks, and response validation.
    """
    
    def __init__(self):
        """Initialize LLM gateway with available providers."""
        self.prompt_engineer = PromptEngineer()
        
        # Initialize providers based on configuration
        self.providers = {}
        
        # Initialize OpenAI provider if configured
        if Config.LLM_PROVIDER == 'openai' and Config.OPENAI_API_KEY:
            self.providers['openai'] = OpenAIProvider()
        
        # Initialize Ollama provider if configured
        if Config.LLM_PROVIDER == 'ollama':
            self.providers['ollama'] = OllamaProvider()
        
        # Set primary provider
        self.primary_provider = Config.LLM_PROVIDER
        
        if not self.providers:
            logger.error("No LLM providers configured")
        else:
            logger.info(f"LLM Gateway initialized with providers: {list(self.providers.keys())}")
    
    async def generate_answer(
        self, 
        query: str, 
        relevant_content: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive answer to a user query.
        
        This method orchestrates the complete answer generation process:
        1. Create optimized prompts with context
        2. Send to configured LLM provider
        3. Validate and enhance response
        4. Extract citations and metadata
        
        Args:
            query: User's original question
            relevant_content: List of relevant content chunks
            
        Returns:
            Dictionary with generated answer and metadata
            
        Raises:
            RuntimeError: If no providers are available or all fail
        """
        
        if not self.providers:
            raise RuntimeError("No LLM providers configured")
        
        if not relevant_content:
            logger.warning("No relevant content provided for answer generation")
            return self._create_no_content_response(query)
        
        # Create optimized prompts
        prompts = self.prompt_engineer.create_answer_prompt(query, relevant_content)
        
        # Try primary provider first
        primary_provider = self.providers.get(self.primary_provider)
        if primary_provider:
            try:
                logger.info(f"Attempting answer generation with {self.primary_provider}")
                
                response = await primary_provider.generate_response(
                    prompts['system'], 
                    prompts['user']
                )
                
                # Enhance response with metadata
                enhanced_response = self._enhance_response(
                    response, query, relevant_content
                )
                
                return enhanced_response
                
            except Exception as e:
                logger.error(f"Primary provider {self.primary_provider} failed: {e}")
                
                # Try fallback providers
                for provider_name, provider in self.providers.items():
                    if provider_name != self.primary_provider:
                        try:
                            logger.info(f"Trying fallback provider: {provider_name}")
                            
                            response = await provider.generate_response(
                                prompts['system'], 
                                prompts['user']
                            )
                            
                            enhanced_response = self._enhance_response(
                                response, query, relevant_content
                            )
                            
                            # Mark as fallback
                            enhanced_response['used_fallback'] = True
                            enhanced_response['fallback_provider'] = provider_name
                            
                            return enhanced_response
                            
                        except Exception as fallback_error:
                            logger.error(f"Fallback provider {provider_name} failed: {fallback_error}")
                            continue
        
        # If we reach here, all providers failed
        raise RuntimeError("All LLM providers failed to generate response")
    
    def _enhance_response(
        self, 
        llm_response: Dict[str, Any], 
        query: str, 
        relevant_content: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Enhance LLM response with additional metadata and validation.
        
        Args:
            llm_response: Raw response from LLM provider
            query: Original user query
            relevant_content: Content used for generation
            
        Returns:
            Enhanced response dictionary
        """
        
        response_text = llm_response.get('response', '')
        
        # Extract citations from response
        citations = self.prompt_engineer.extract_citations(response_text)
        
        # Collect source information
        sources = []
        for content in relevant_content:
            source_info = {
                'url': content.get('url', ''),
                'title': content.get('title', ''),
                'similarity_score': content.get('similarity_score', 0)
            }
            if source_info['url'] and source_info not in sources:
                sources.append(source_info)
        
        # Calculate response quality metrics
        quality_metrics = self._calculate_quality_metrics(
            response_text, query, citations, sources
        )
        
        # Build enhanced response
        enhanced = {
            'answer': response_text,
            'query': query,
            'citations': citations,
            'sources': sources,
            'provider_info': {
                'provider': llm_response.get('provider', 'unknown'),
                'model': llm_response.get('model', 'unknown'),
                'tokens_used': llm_response.get('tokens_used', {}),
                'success': llm_response.get('success', False)
            },
            'quality_metrics': quality_metrics,
            'content_chunks_used': len(relevant_content),
            'used_fallback': False
        }
        
        return enhanced
    
    def _calculate_quality_metrics(
        self, 
        response_text: str, 
        query: str, 
        citations: List[str], 
        sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate quality metrics for response validation.
        
        Args:
            response_text: Generated response text
            query: Original query
            citations: Extracted citations
            sources: Available sources
            
        Returns:
            Dictionary with quality metrics
        """
        
        # Basic metrics
        word_count = len(response_text.split())
        citation_count = len(citations)
        source_count = len(sources)
        
        # Citation coverage (what percentage of sources were cited)
        citation_coverage = 0
        if source_count > 0:
            cited_urls = set(citations)
            source_urls = set(s['url'] for s in sources if s['url'])
            overlapping_citations = cited_urls.intersection(source_urls)
            citation_coverage = len(overlapping_citations) / source_count
        
        # Response completeness heuristics
        has_conclusion = any(phrase in response_text.lower() 
                           for phrase in ['in conclusion', 'to summarize', 'overall', 'in summary'])
        
        acknowledges_limitations = any(phrase in response_text.lower() 
                                     for phrase in ['however', 'limited information', 
                                                  'not enough information', 'unclear'])
        
        return {
            'word_count': word_count,
            'citation_count': citation_count,
            'source_count': source_count,
            'citation_coverage': round(citation_coverage, 2),
            'has_conclusion': has_conclusion,
            'acknowledges_limitations': acknowledges_limitations,
            'estimated_quality_score': self._estimate_quality_score(
                word_count, citation_count, citation_coverage, 
                has_conclusion, acknowledges_limitations
            )
        }
    
    def _estimate_quality_score(
        self, 
        word_count: int, 
        citation_count: int, 
        citation_coverage: float,
        has_conclusion: bool, 
        acknowledges_limitations: bool
    ) -> float:
        """
        Estimate overall response quality score (0-1).
        
        Args:
            word_count: Number of words in response
            citation_count: Number of citations
            citation_coverage: Ratio of sources cited
            has_conclusion: Whether response has concluding language
            acknowledges_limitations: Whether response acknowledges limits
            
        Returns:
            Quality score between 0 and 1
        """
        
        score = 0.0
        
        # Word count score (diminishing returns after 200 words)
        if word_count >= 50:
            score += min(word_count / 200, 0.3)
        
        # Citation score
        if citation_count > 0:
            score += min(citation_count / 5, 0.3)  # Max points for 5+ citations
        
        # Coverage score
        score += citation_coverage * 0.2
        
        # Structure and quality indicators
        if has_conclusion:
            score += 0.1
        
        if acknowledges_limitations:
            score += 0.1
        
        return min(score, 1.0)
    
    def _create_no_content_response(self, query: str) -> Dict[str, Any]:
        """
        Create response when no relevant content is available.
        
        Args:
            query: User query
            
        Returns:
            Response indicating lack of content
        """
        
        return {
            'answer': (
                f"I apologize, but I couldn't find sufficient relevant information "
                f"to answer your question: '{query}'. This could be because:\n\n"
                f"1. The search didn't return relevant results\n"
                f"2. The web pages couldn't be accessed or parsed\n"
                f"3. The content didn't meet the relevance threshold\n\n"
                f"Please try rephrasing your question or using different keywords."
            ),
            'query': query,
            'citations': [],
            'sources': [],
            'provider_info': {
                'provider': 'none',
                'model': 'none',
                'tokens_used': {},
                'success': False
            },
            'quality_metrics': {
                'word_count': 0,
                'citation_count': 0,
                'source_count': 0,
                'citation_coverage': 0,
                'estimated_quality_score': 0
            },
            'content_chunks_used': 0,
            'error': 'No relevant content available'
        }


# Global LLM gateway instance
llm_gateway = LLMGateway()

# ============================================================================
# CELERY TASK DEFINITION
# ============================================================================

@celery_app.task(
    bind=True,
    name='services.llm_gateway.generate_answer',
    max_retries=2,
    default_retry_delay=60,
    autoretry_for=(RuntimeError, httpx.HTTPError, openai.APIError),
    retry_backoff=True
)
def generate_answer(
    self, 
    query: str, 
    relevant_content: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Celery task to generate final answer using LLM.
    
    This task coordinates with the configured LLM provider to generate
    a comprehensive, well-cited answer based on relevant content.
    
    Args:
        query: User's original question
        relevant_content: List of relevant content chunks from embedding service
        
    Returns:
        Dictionary containing:
        - 'answer': Generated response text
        - 'query': Original query
        - 'citations': List of cited URLs
        - 'sources': List of source information
        - 'provider_info': Information about LLM provider used
        - 'quality_metrics': Response quality indicators
        - 'processing_time': Time taken for generation
        
    Raises:
        ValueError: For invalid input parameters
        RuntimeError: For LLM provider errors (will trigger retry)
    """
    
    # Validate input parameters
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    
    if not isinstance(relevant_content, list):
        raise ValueError("Relevant content must be a list")
    
    query = query.strip()
    
    # Log task start
    task_id = current_task.request.id if current_task else 'unknown'
    logger.info(f"LLM task {task_id} started for query: '{query}'")
    
    try:
        # Record start time for performance monitoring
        import time
        start_time = time.time()
        
        # Generate answer using LLM gateway
        answer_result = asyncio.run(
            llm_gateway.generate_answer(query, relevant_content)
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Add task metadata
        answer_result.update({
            'processing_time': round(processing_time, 2),
            'task_id': task_id
        })
        
        # Log completion
        quality_score = answer_result.get('quality_metrics', {}).get('estimated_quality_score', 0)
        citation_count = answer_result.get('quality_metrics', {}).get('citation_count', 0)
        
        logger.info(
            f"LLM task {task_id} completed: "
            f"quality_score={quality_score:.2f}, citations={citation_count}, "
            f"time={processing_time:.2f}s"
        )
        
        return answer_result
        
    except Exception as e:
        logger.error(f"LLM task {task_id} failed for query '{query}': {e}")
        
        # Re-raise for Celery retry mechanism
        raise self.retry(exc=e, countdown=60, max_retries=2)


# ============================================================================
# UTILITY FUNCTIONS FOR EXTERNAL USAGE
# ============================================================================

def generate_answer_sync(
    query: str, 
    relevant_content: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Synchronous wrapper for answer generation functionality.
    
    This function provides a simple interface for testing and debugging
    without requiring Celery infrastructure.
    
    Args:
        query: User query
        relevant_content: List of relevant content chunks
        
    Returns:
        Answer results in same format as Celery task
    """
    
    try:
        # Generate answer directly without Celery
        result = asyncio.run(
            llm_gateway.generate_answer(query, relevant_content)
        )
        
        # Add sync metadata
        result.update({
            'processing_time': 0,  # Not tracked in sync mode
            'task_id': 'sync'
        })
        
        return result
        
    except Exception as e:
        logger.error(f"Synchronous answer generation failed for query '{query}': {e}")
        raise


# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

logger.info("LLM Gateway service initialized")
logger.info(f"Primary LLM provider: {Config.LLM_PROVIDER}")

if Config.LLM_PROVIDER == 'openai':
    logger.info(f"OpenAI model: {Config.OPENAI_MODEL}")
    logger.info(f"OpenAI max tokens: {Config.OPENAI_MAX_TOKENS}")
elif Config.LLM_PROVIDER == 'ollama':
    logger.info(f"Ollama URL: {Config.OLLAMA_URL}")
    logger.info(f"Ollama model: {Config.OLLAMA_MODEL}")

# Log gateway initialization status
if llm_gateway.providers:
    logger.info(f"LLM Gateway ready with {len(llm_gateway.providers)} provider(s)")
else:
    logger.error("LLM Gateway initialization failed - no providers available")