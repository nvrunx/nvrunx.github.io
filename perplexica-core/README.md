# PerplexicaCore: AI-Augmented Search Engine

A powerful, scalable, and accurate AI-augmented search engine that answers user queries by searching the web in real-time, processing results with advanced AI, and generating well-cited, accurate answers.

## 🎯 Project Overview

PerplexicaCore is a production-ready prototype that implements a sophisticated AI search pipeline:

1. **Web Search**: Uses SearxNG meta-search for comprehensive results
2. **Content Extraction**: Intelligent scraping with robots.txt compliance
3. **AI Ranking**: Two-stage relevance filtering using embeddings
4. **Answer Generation**: Multi-LLM support with proper citations

## 🏗️ Architecture

### Microservices Design
- **Message Queue**: RabbitMQ + Celery for distributed processing
- **Caching**: Redis for performance optimization
- **Search Service**: SearxNG integration with result filtering
- **Scraper Service**: Ethical web scraping with rate limiting
- **Embedding Service**: AI-powered content relevance ranking
- **LLM Gateway**: Multi-provider LLM integration
- **Orchestrator**: Central coordination and pipeline management

### Key Features
- ✅ Horizontal scaling with microservices
- ✅ Asynchronous processing throughout
- ✅ Two-stage relevance filtering (search + AI)
- ✅ Robust content extraction
- ✅ Citation transparency
- ✅ Multi-LLM backend support
- ✅ Ethical crawling practices
- ✅ Comprehensive error handling

## 📁 Project Structure

```
perplexica-core/
├── config.py                 # Centralized configuration management
├── message_queue.py          # Celery + RabbitMQ + Redis setup
├── main_orchestrator.py      # Pipeline coordination
├── requirements.txt          # Python dependencies
├── COLAB_SETUP.py           # Google Colab setup guide
├── README.md                # This file
└── services/
    ├── search_service.py     # SearxNG integration
    ├── scraper_service.py    # Web scraping service
    ├── embedding_service.py  # AI content ranking
    └── llm_gateway.py        # LLM integration
```

## 🚀 Quick Start with Google Colab

The fastest way to get started is using Google Colab:

1. Open Google Colab
2. Upload and run `COLAB_SETUP.py`
3. Set your OpenAI API key when prompted
4. Start asking questions!

```python
# After setup in Colab:
from main_orchestrator import search_and_answer_sync

result = search_and_answer_sync("What are the latest developments in quantum computing?")
print(result['answer'])
```

## 🛠️ Local Installation

### Prerequisites
- Python 3.8+
- Redis server
- RabbitMQ server
- SearxNG instance (optional - can use public instances)

### Installation Steps

1. **Clone and Setup**
```bash
git clone <repository-url>
cd perplexica-core
pip install -r requirements.txt
```

2. **Start Services**
```bash
# Start Redis
redis-server

# Start RabbitMQ
rabbitmq-server

# Start SearxNG (optional - use public instance)
# See SearxNG documentation for setup
```

3. **Configure Environment**
```bash
export OPENAI_API_KEY="your-api-key-here"
export SEARXNG_URL="http://localhost:8080"  # or public instance
export REDIS_URL="redis://localhost:6379/0"
export RABBITMQ_URL="amqp://guest:guest@localhost:5672//"
```

4. **Start Celery Workers**
```bash
# Terminal 1: Start worker
celery -A message_queue.celery_app worker --loglevel=info

# Terminal 2: Start monitoring (optional)
celery -A message_queue.celery_app flower
```

5. **Test the System**
```python
from main_orchestrator import search_and_answer_sync

result = search_and_answer_sync("How does machine learning work?")
print(result['answer'])
```

## ⚙️ Configuration

All configuration is handled through environment variables:

### Required Settings
```bash
OPENAI_API_KEY=your-openai-api-key        # For LLM access
SEARXNG_URL=http://localhost:8080         # SearxNG instance
REDIS_URL=redis://localhost:6379/0        # Redis connection
RABBITMQ_URL=amqp://guest:guest@localhost:5672//  # RabbitMQ connection
```

### Optional Settings
```bash
# Search Configuration
SEARCH_RESULTS_LIMIT=20                   # Max search results
MAX_URLS_TO_SCRAPE=10                     # Max URLs to scrape
SIMILARITY_THRESHOLD=0.3                  # AI relevance threshold
TOP_RELEVANT_CHUNKS=5                     # Top content chunks

# LLM Configuration
LLM_PROVIDER=openai                       # or 'ollama' for local
OPENAI_MODEL=gpt-3.5-turbo               # OpenAI model
OPENAI_MAX_TOKENS=1000                   # Response length limit
OPENAI_TEMPERATURE=0.1                   # Response randomness

# Performance
CACHE_TTL=300                            # Cache duration (seconds)
DOMAIN_DELAY=1.0                         # Delay between requests
SCRAPER_TIMEOUT=30                       # Scraping timeout
```

## 📖 Usage Examples

### Basic Query
```python
from main_orchestrator import search_and_answer_sync

result = search_and_answer_sync("What is climate change?")
print("Answer:", result['answer'])
print("Sources:", [s['url'] for s in result['sources']])
```

### Advanced Usage with Progress Tracking
```python
import asyncio
from main_orchestrator import search_and_answer

def progress_callback(progress):
    print(f"Stage: {progress['stage']} - {progress['message']}")

result = asyncio.run(search_and_answer(
    "Explain quantum computing",
    progress_callback=progress_callback
))
```

### Batch Processing
```python
queries = [
    "What is artificial intelligence?",
    "How does blockchain work?",
    "What are the benefits of renewable energy?"
]

results = []
for query in queries:
    result = search_and_answer_sync(query)
    results.append({
        'query': query,
        'answer': result['answer'],
        'sources': len(result['sources'])
    })
```

## 🔧 API Reference

### Main Functions

#### `search_and_answer_sync(query, max_results=None)`
Synchronous interface for processing queries.

**Parameters:**
- `query` (str): User's search question
- `max_results` (int, optional): Maximum search results to process

**Returns:**
- Dictionary with answer, citations, sources, and metadata

#### `search_and_answer(query, max_results=None, progress_callback=None)`
Asynchronous interface with progress tracking.

**Parameters:**
- `query` (str): User's search question
- `max_results` (int, optional): Maximum search results to process
- `progress_callback` (callable, optional): Progress update function

### Response Format
```python
{
    'answer': 'Generated answer with citations...',
    'query': 'Original user question',
    'citations': ['http://example.com/1', 'http://example.com/2'],
    'sources': [
        {'url': 'http://example.com/1', 'title': 'Source Title 1'},
        {'url': 'http://example.com/2', 'title': 'Source Title 2'}
    ],
    'pipeline_stats': {
        'total_search_results': 20,
        'successful_scrapes': 15,
        'relevant_chunks': 5
    },
    'quality_metrics': {
        'citation_count': 3,
        'estimated_quality_score': 0.85
    }
}
```

## 🔍 Monitoring and Health Checks

### System Health
```python
from main_orchestrator import get_pipeline_health

health = get_pipeline_health()
print("System Status:", health['pipeline']['overall_status'])
print("Redis:", "OK" if health['redis'] else "FAILED")
print("Celery:", "OK" if health['celery'] else "FAILED")
```

### Performance Monitoring
```python
# Access detailed pipeline statistics
result = search_and_answer_sync("Your question here")
stats = result['detailed_results']

print("Search time:", stats['search']['processing_time'])
print("Scraping success rate:", stats['scraping']['success_rate'])
print("Ranking time:", stats['ranking']['processing_time'])
print("LLM tokens used:", stats['generation']['tokens_used'])
```

## 🛡️ Production Considerations

### Security
- API keys stored in environment variables
- Rate limiting for ethical crawling
- Input validation and sanitization
- Error handling without information leakage

### Scalability
- Horizontal scaling with additional Celery workers
- Redis clustering for larger caches
- RabbitMQ clustering for high availability
- Load balancing for multiple orchestrator instances

### Monitoring
- Structured logging throughout
- Prometheus metrics integration ready
- Health check endpoints
- Performance tracking and alerting

### Error Handling
- Comprehensive retry mechanisms
- Graceful degradation
- Circuit breaker patterns
- Detailed error reporting

## 🔧 Troubleshooting

### Common Issues

**"No LLM providers configured"**
- Ensure OPENAI_API_KEY is set
- Check LLM_PROVIDER setting
- Verify API key validity

**"Search results not found"**
- Check SEARXNG_URL configuration
- Verify SearxNG instance is running
- Try a different SearxNG instance

**"Celery workers not responding"**
- Verify RabbitMQ is running
- Check Redis connectivity
- Restart Celery workers

**"Content extraction failed"**
- Check internet connectivity
- Verify target sites are accessible
- Review robots.txt compliance

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with verbose logging
result = search_and_answer_sync("Your question", verbose=True)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests
4. Update documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **SearxNG**: Privacy-focused meta-search engine
- **Sentence Transformers**: Semantic similarity models
- **Trafilatura**: Content extraction library
- **OpenAI**: LLM API services
- **Celery**: Distributed task processing

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the troubleshooting section
- Review configuration documentation

---

**PerplexicaCore**: Bridging web search and AI for accurate, cited answers. 🔍🤖