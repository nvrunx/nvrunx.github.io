"""
Embedding Service for PerplexicaCore
====================================

This service implements the second stage of relevance filtering using AI embeddings.
It vectorizes both the user query and scraped content, then uses cosine similarity
to identify the most relevant content chunks for answering the user's question.

Key features:
- Sentence transformer embeddings for semantic similarity
- Efficient batch processing of content
- Configurable similarity thresholds
- Content chunking for large documents
- Caching of embeddings for performance
- GPU acceleration when available
"""

import logging
import asyncio
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import sent_tokenize
from celery import current_task
from message_queue import celery_app, cached_task
from config import Config

# Configure logging for this module
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    logger.warning(f"Failed to download NLTK data: {e}")

# ============================================================================
# TEXT PREPROCESSING AND CHUNKING
# ============================================================================

class TextProcessor:
    """
    Text preprocessing and chunking utilities for optimal embedding generation.
    
    This class handles text normalization, sentence segmentation, and intelligent
    chunking to maximize the effectiveness of semantic similarity matching.
    """
    
    def __init__(self, max_chunk_size: int = 500, overlap_size: int = 50):
        """
        Initialize text processor with chunking parameters.
        
        Args:
            max_chunk_size: Maximum words per chunk
            overlap_size: Word overlap between adjacent chunks
        """
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text for embedding generation.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text suitable for embedding
        """
        if not text:
            return ""
        
        # Basic cleaning
        text = text.strip()
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove very short lines (likely navigation/UI elements)
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if len(line) > 10:  # Only keep substantial lines
                cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        
        # Limit text length to prevent memory issues
        if len(text) > Config.MAX_CONTENT_LENGTH:
            text = text[:Config.MAX_CONTENT_LENGTH]
        
        return text
    
    def chunk_text(self, text: str, title: str = "") -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks for better semantic matching.
        
        This method creates chunks that:
        1. Respect sentence boundaries
        2. Have configurable overlap to maintain context
        3. Include title context for better relevance
        
        Args:
            text: Text content to chunk
            title: Optional title to include in chunks
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        
        # Clean the input text
        clean_text = self.clean_text(text)
        
        if not clean_text:
            return []
        
        # Try to tokenize into sentences
        try:
            sentences = sent_tokenize(clean_text)
        except Exception as e:
            logger.debug(f"Sentence tokenization failed, using simple splitting: {e}")
            # Fallback to simple sentence splitting
            sentences = [s.strip() for s in clean_text.split('.') if s.strip()]
        
        if not sentences:
            return []
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        # Add title context if available
        title_context = ""
        if title and title.strip():
            title_context = f"Title: {title.strip()}\n\n"
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_words = len(sentence.split())
            
            # If adding this sentence would exceed max chunk size, finalize current chunk
            if (current_word_count + sentence_words > self.max_chunk_size and 
                current_chunk):
                
                # Create chunk from current sentences
                chunk_text = title_context + ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'word_count': current_word_count,
                    'sentence_count': len(current_chunk),
                    'chunk_index': len(chunks)
                })
                
                # Start new chunk with overlap
                if self.overlap_size > 0 and len(current_chunk) > 1:
                    # Keep last few sentences for overlap
                    overlap_sentences = []
                    overlap_words = 0
                    
                    for sent in reversed(current_chunk):
                        sent_words = len(sent.split())
                        if overlap_words + sent_words <= self.overlap_size:
                            overlap_sentences.insert(0, sent)
                            overlap_words += sent_words
                        else:
                            break
                    
                    current_chunk = overlap_sentences
                    current_word_count = overlap_words
                else:
                    current_chunk = []
                    current_word_count = 0
            
            # Add current sentence to chunk
            current_chunk.append(sentence)
            current_word_count += sentence_words
        
        # Add final chunk if it has content
        if current_chunk:
            chunk_text = title_context + ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'word_count': current_word_count,
                'sentence_count': len(current_chunk),
                'chunk_index': len(chunks)
            })
        
        logger.debug(f"Split text into {len(chunks)} chunks (max {self.max_chunk_size} words each)")
        
        return chunks


# ============================================================================
# EMBEDDING GENERATION ENGINE
# ============================================================================

class EmbeddingGenerator:
    """
    High-performance embedding generation using sentence transformers.
    
    This class handles:
    - Model loading and GPU utilization
    - Batch embedding generation for efficiency
    - Embedding normalization for cosine similarity
    - Error handling and fallbacks
    """
    
    def __init__(self):
        """Initialize embedding generator with model loading."""
        self.model = None
        self.device = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model with optimal settings."""
        try:
            # Determine best available device
            if torch.cuda.is_available():
                self.device = 'cuda'
                logger.info("Using GPU for embedding generation")
            else:
                self.device = 'cpu'
                logger.info("Using CPU for embedding generation")
            
            # Load the pre-trained model
            logger.info(f"Loading embedding model: {Config.EMBEDDING_MODEL}")
            self.model = SentenceTransformer(
                Config.EMBEDDING_MODEL,
                device=self.device
            )
            
            # Set model to evaluation mode for inference
            self.model.eval()
            
            logger.info("Embedding model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.model = None
            self.device = 'cpu'
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing (larger = faster but more memory)
            
        Returns:
            Numpy array of embeddings with shape (n_texts, embedding_dim)
            
        Raises:
            RuntimeError: If model is not loaded or embedding fails
        """
        
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        
        if not texts:
            return np.array([])
        
        try:
            # Filter out empty texts
            valid_texts = [text for text in texts if text and text.strip()]
            
            if not valid_texts:
                logger.warning("No valid texts provided for embedding")
                return np.array([])
            
            logger.debug(f"Generating embeddings for {len(valid_texts)} texts")
            
            # Generate embeddings in batches for memory efficiency
            embeddings = self.model.encode(
                valid_texts,
                batch_size=batch_size,
                show_progress_bar=False,  # Disable for cleaner logs
                convert_to_numpy=True,    # Return numpy arrays
                normalize_embeddings=True # Normalize for cosine similarity
            )
            
            logger.debug(f"Generated embeddings with shape: {embeddings.shape}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise RuntimeError(f"Failed to generate embeddings: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model metadata
        """
        if not self.model:
            return {'loaded': False, 'error': 'Model not loaded'}
        
        return {
            'loaded': True,
            'model_name': Config.EMBEDDING_MODEL,
            'device': self.device,
            'max_seq_length': getattr(self.model, 'max_seq_length', 'unknown'),
            'embedding_dimension': self.model.get_sentence_embedding_dimension(),
        }


# Global embedding generator instance
embedding_generator = EmbeddingGenerator()

# ============================================================================
# SIMILARITY CALCULATION AND RANKING
# ============================================================================

class SimilarityCalculator:
    """
    Semantic similarity calculation and content ranking.
    
    This class implements sophisticated similarity metrics to identify
    the most relevant content chunks for a given query.
    """
    
    def __init__(self):
        """Initialize similarity calculator."""
        self.text_processor = TextProcessor()
    
    def calculate_similarities(
        self, 
        query_embedding: np.ndarray, 
        content_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Calculate cosine similarities between query and content embeddings.
        
        Args:
            query_embedding: Query embedding vector (1D array)
            content_embeddings: Content embeddings matrix (2D array)
            
        Returns:
            Array of similarity scores
        """
        
        if query_embedding.size == 0 or content_embeddings.size == 0:
            return np.array([])
        
        # Ensure query_embedding is 2D for sklearn
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding, content_embeddings)[0]
        
        return similarities
    
    def rank_content_by_relevance(
        self, 
        query: str, 
        content_items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Rank content items by semantic relevance to query.
        
        This method:
        1. Chunks large content into smaller pieces
        2. Generates embeddings for query and all content
        3. Calculates semantic similarities
        4. Ranks and filters by relevance threshold
        5. Returns top N most relevant chunks
        
        Args:
            query: User search query
            content_items: List of content dictionaries from scraper
            
        Returns:
            List of ranked and filtered content chunks with relevance scores
        """
        
        if not query or not content_items:
            logger.warning("Empty query or content items provided")
            return []
        
        logger.info(f"Ranking {len(content_items)} content items for query: '{query}'")
        
        # Step 1: Prepare content chunks
        all_chunks = []
        chunk_to_source = []  # Map chunks back to their source URLs
        
        for item_idx, item in enumerate(content_items):
            if not item.get('success', False):
                logger.debug(f"Skipping failed content item: {item.get('url', 'unknown')}")
                continue
            
            content = item.get('content', '')
            title = item.get('title', '')
            url = item.get('url', '')
            
            if not content.strip():
                logger.debug(f"Skipping empty content for URL: {url}")
                continue
            
            # Chunk the content
            chunks = self.text_processor.chunk_text(content, title)
            
            for chunk in chunks:
                all_chunks.append(chunk['text'])
                chunk_to_source.append({
                    'url': url,
                    'title': title,
                    'item_index': item_idx,
                    'chunk_index': chunk['chunk_index'],
                    'word_count': chunk['word_count']
                })
        
        if not all_chunks:
            logger.warning("No valid content chunks found")
            return []
        
        logger.debug(f"Created {len(all_chunks)} content chunks for embedding")
        
        # Step 2: Generate embeddings
        try:
            # Generate query embedding
            query_embeddings = embedding_generator.generate_embeddings([query])
            if query_embeddings.size == 0:
                logger.error("Failed to generate query embedding")
                return []
            
            query_embedding = query_embeddings[0]
            
            # Generate content embeddings
            content_embeddings = embedding_generator.generate_embeddings(all_chunks)
            if content_embeddings.size == 0:
                logger.error("Failed to generate content embeddings")
                return []
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return []
        
        # Step 3: Calculate similarities
        similarities = self.calculate_similarities(query_embedding, content_embeddings)
        
        if similarities.size == 0:
            logger.error("Failed to calculate similarities")
            return []
        
        # Step 4: Create ranked results
        ranked_results = []
        
        for idx, (chunk_text, source_info, similarity) in enumerate(
            zip(all_chunks, chunk_to_source, similarities)
        ):
            # Filter by similarity threshold
            if similarity < Config.SIMILARITY_THRESHOLD:
                continue
            
            result = {
                'text': chunk_text,
                'similarity_score': float(similarity),
                'url': source_info['url'],
                'title': source_info['title'],
                'word_count': source_info['word_count'],
                'chunk_index': source_info['chunk_index'],
                'item_index': source_info['item_index']
            }
            
            ranked_results.append(result)
        
        # Step 5: Sort by similarity score (highest first)
        ranked_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Step 6: Limit to top N results
        top_results = ranked_results[:Config.TOP_RELEVANT_CHUNKS]
        
        logger.info(
            f"Ranked content: {len(ranked_results)} relevant chunks found, "
            f"returning top {len(top_results)}"
        )
        
        return top_results


# Global similarity calculator instance
similarity_calculator = SimilarityCalculator()

# ============================================================================
# CELERY TASK DEFINITION
# ============================================================================

@celery_app.task(
    bind=True,
    name='services.embedding_service.rank_content_by_relevance',
    max_retries=2,
    default_retry_delay=60,
    autoretry_for=(RuntimeError, torch.cuda.OutOfMemoryError),
    retry_backoff=True
)
@cached_task('embedding', cache_ttl=600)  # Cache for 10 minutes
def rank_content_by_relevance(
    self, 
    query: str, 
    content_items: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Celery task to rank content by semantic relevance to a query.
    
    This task implements the second stage of relevance filtering using
    AI embeddings and semantic similarity matching.
    
    Args:
        query: User search query
        content_items: List of scraped content dictionaries
        
    Returns:
        Dictionary containing:
        - 'ranked_content': List of top relevant content chunks
        - 'query': Original query
        - 'total_chunks': Total number of content chunks processed
        - 'relevant_chunks': Number of chunks above similarity threshold
        - 'top_chunks': Number of top chunks returned
        - 'processing_time': Time taken for ranking
        - 'model_info': Information about embedding model used
        
    Raises:
        ValueError: For invalid input parameters
        RuntimeError: For embedding or similarity calculation errors
    """
    
    # Validate input parameters
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    
    if not content_items or not isinstance(content_items, list):
        raise ValueError("Content items must be a non-empty list")
    
    query = query.strip()
    
    # Log task start
    task_id = current_task.request.id if current_task else 'unknown'
    logger.info(f"Embedding task {task_id} started for query: '{query}'")
    
    try:
        # Record start time for performance monitoring
        import time
        start_time = time.time()
        
        # Perform content ranking
        ranked_content = similarity_calculator.rank_content_by_relevance(
            query, content_items
        )
        
        # Calculate statistics
        total_content_items = len(content_items)
        successful_items = sum(1 for item in content_items if item.get('success', False))
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Get model information
        model_info = embedding_generator.get_model_info()
        
        # Prepare response
        response = {
            'ranked_content': ranked_content,
            'query': query,
            'total_content_items': total_content_items,
            'successful_items': successful_items,
            'relevant_chunks': len(ranked_content),
            'top_chunks': min(len(ranked_content), Config.TOP_RELEVANT_CHUNKS),
            'processing_time': round(processing_time, 2),
            'model_info': model_info,
            'task_id': task_id,
            'similarity_threshold': Config.SIMILARITY_THRESHOLD
        }
        
        logger.info(
            f"Embedding task {task_id} completed: "
            f"found {len(ranked_content)} relevant chunks in {processing_time:.2f}s"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Embedding task {task_id} failed for query '{query}': {e}")
        
        # Re-raise for Celery retry mechanism
        raise self.retry(exc=e, countdown=60, max_retries=2)


# ============================================================================
# UTILITY FUNCTIONS FOR EXTERNAL USAGE
# ============================================================================

def rank_content_sync(
    query: str, 
    content_items: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Synchronous wrapper for content ranking functionality.
    
    This function provides a simple interface for testing and debugging
    without requiring Celery infrastructure.
    
    Args:
        query: Search query
        content_items: List of content dictionaries
        
    Returns:
        Ranking results in same format as Celery task
    """
    
    try:
        # Perform ranking directly without Celery
        ranked_content = similarity_calculator.rank_content_by_relevance(
            query, content_items
        )
        
        # Calculate basic statistics
        total_content_items = len(content_items)
        successful_items = sum(1 for item in content_items if item.get('success', False))
        
        return {
            'ranked_content': ranked_content,
            'query': query,
            'total_content_items': total_content_items,
            'successful_items': successful_items,
            'relevant_chunks': len(ranked_content),
            'top_chunks': min(len(ranked_content), Config.TOP_RELEVANT_CHUNKS),
            'processing_time': 0,  # Not tracked in sync mode
            'model_info': embedding_generator.get_model_info(),
            'task_id': 'sync',
            'similarity_threshold': Config.SIMILARITY_THRESHOLD
        }
        
    except Exception as e:
        logger.error(f"Synchronous content ranking failed for query '{query}': {e}")
        raise


# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

logger.info("Embedding service initialized")
logger.info(f"Embedding model: {Config.EMBEDDING_MODEL}")
logger.info(f"Similarity threshold: {Config.SIMILARITY_THRESHOLD}")
logger.info(f"Top relevant chunks: {Config.TOP_RELEVANT_CHUNKS}")

# Log model initialization status
model_info = embedding_generator.get_model_info()
if model_info.get('loaded'):
    logger.info(f"Model loaded successfully on {model_info.get('device', 'unknown')}")
else:
    logger.error(f"Model loading failed: {model_info.get('error', 'unknown')}")