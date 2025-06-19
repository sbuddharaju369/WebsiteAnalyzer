# Web Content Analyzer

A comprehensive Streamlit-based web application that transforms any publicly accessible website into intelligent, searchable knowledge using AI-powered content analysis.

## 🏗️ Project Structure

```
web-content-analyzer/
├── app.py                          # Main Streamlit application
├── web_crawler.py                  # Web crawling and content extraction
├── web_rag_engine.py              # ChromaDB-based RAG engine
├── simple_rag_engine.py           # Lightweight in-memory RAG engine
├── config/
│   └── settings.py                 # Application configuration and constants
├── src/
│   ├── core/                       # Core business logic modules
│   │   ├── crawler.py              # Advanced web crawling features
│   │   └── rag_engine.py           # Enhanced RAG processing
│   ├── ui/                         # User interface components
│   │   └── visualizations.py      # Charts, graphs, and UI helpers
│   └── utils/                      # Utility modules
│       └── cache_manager.py       # Cache file management
├── data/                           # Data storage (auto-created)
│   ├── cache/                      # Cached crawled content with embeddings
│   └── chroma/                     # Vector database storage
├── .streamlit/
│   └── config.toml                 # Streamlit configuration
├── pyproject.toml                  # Python dependencies
├── replit.md                       # Project documentation and changelog
└── README.md                       # This file
```

## 🚀 Features

### Web Crawling
- **Recursive URL Discovery**: Automatically discovers and crawls all pages within a domain
- **Intelligent Crawling**: Respects robots.txt and implements rate limiting
- **Content Extraction**: Uses trafilatura for clean text extraction
- **Domain Scoping**: Stays within target website boundaries
- **Progress Tracking**: Real-time crawling progress with metrics

## How Recursive Web Crawling Works

### URL Discovery Process
The Web Content Analyzer performs intelligent recursive crawling starting from any URL you provide:

1. **Starting Point**: Begin crawling from your provided URL (e.g., `https://example.com/page1`)
2. **Link Extraction**: Extract all links from each visited page using Beautiful Soup
3. **Domain Scoping**: Only follow links within the same domain to prevent external crawling
4. **Recursive Discovery**: Continue discovering new pages through internal links automatically
5. **Depth Control**: Configurable maximum page limits (1-100 pages) and request delays (0.5-5 seconds)

### Crawling Strategy
- **Respectful Crawling**: Implements rate limiting with configurable delays between requests
- **Content Filtering**: Focuses on text content, skipping non-content files (images, PDFs, etc.)
- **Real-time Progress**: Live tracking shows pages visited vs content successfully extracted
- **Error Handling**: Gracefully handles failed requests and continues crawling other pages
- **Queue Management**: Maintains a queue of discovered URLs for systematic processing

### Domain Boundaries & Security
- **Same-Domain Only**: If you provide `https://company.com/about`, it will only crawl `company.com` pages
- **Subdomain Handling**: Includes subdomains within the same root domain
- **External Link Filtering**: Automatically excludes links to external websites
- **Path Validation**: Ensures URLs are properly formatted and accessible
- **Duplicate Prevention**: Avoids crawling the same page multiple times

### Content Processing Pipeline
1. **Text Extraction**: Uses Trafilatura for clean content extraction from HTML
2. **Smart Chunking**: Breaks content into 1000-token chunks with 100-token overlap using tiktoken
3. **Semantic Boundaries**: Respects paragraph and sentence boundaries for better context preservation
4. **Metadata Preservation**: Maintains URL, title, word count, and source information for each chunk
5. **Caching System**: Saves processed content to JSON files with human-readable names (domain_date_time_pagecount.json)

### Usage Instructions
To start recursive crawling:
1. Enter any website URL in the sidebar's Web Crawler section
2. Configure your settings (max pages: 1-100, delay: 0.5-5 seconds)
3. Click "Start Crawling" to begin automatic discovery
4. Monitor real-time progress as pages are discovered and processed
5. The system will automatically find and process all accessible pages within the domain

### AI-Powered Analysis
- **Dual RAG Implementation**: ChromaDB-based engine for advanced features, SimpleRAGEngine for lightweight deployment
- **Semantic Search**: Keyword-based relevance scoring with similarity matching
- **Smart Chunking**: Token-aware content segmentation with 100-token overlap for better context
- **Content Caching**: Local JSON storage with embedded metadata for fast repeated analysis

### User Interface
- **Collapsible Sidebar**: Organized navigation with drawer system
- **Interactive Visualizations**: Plotly charts and network graphs
- **Multi-Tab Analytics**: Content analysis, search, and raw data views
- **Configurable Verbosity**: Concise, balanced, or comprehensive answers

### Data Management
- **Flexible Storage**: ChromaDB support with fallback to in-memory storage for compatibility
- **Cache Management**: Human-readable filenames with domain, date, time, and page count
- **Coverage Analysis**: Website size estimation using sitemap discovery and link analysis
- **Source Attribution**: Confidence scoring based on content relevance and similarity matching

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.11+
- OpenAI API key
- Internet connection for web crawling

### Environment Setup
```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Install dependencies (handled automatically by Replit)
# pip install -r requirements.txt
```

### Configuration
The application uses environment-based configuration in `config/settings.py`:

- **OPENAI_API_KEY**: Required for AI analysis
- **DATABASE_URL**: PostgreSQL connection (provided by Replit)
- **Data directories**: Auto-created in `data/` folder

## 📖 Usage

### Basic Workflow
1. **Start Application**: Run `streamlit run app.py`
2. **Navigate Sidebar**: Use the collapsible drawer system
3. **Enter URL**: Input website URL in the Web Crawler section
4. **Configure Settings**: Adjust pages limit and crawl delay
5. **Start Crawling**: Click "Start Crawling" and monitor progress
6. **Ask Questions**: Use the AI-powered question interface
7. **Explore Analytics**: View visualizations and raw content

### Sidebar Navigation
- **🕷️ Web**: Web crawler interface and settings
- **💾 Cache**: Load and manage cached content files
- **📊 Content**: Overview statistics and metrics

### Question Interface
- **Suggested Questions**: Auto-generated based on content
- **Answer Styles**: Choose verbosity level (Concise/Balanced/Comprehensive)
- **Source Attribution**: View supporting pages with relevance scores
- **Reliability Guide**: Tips for improving answer quality

### Analytics Features
- **Content Charts**: Word count and depth distribution
- **Network Graph**: Interactive page relationship visualization
- **Semantic Search**: Direct content lookup without AI interpretation
- **Raw Data View**: Complete crawled content with metadata

## 🔧 Technical Architecture

### Core Modules

**WebCrawler** (`web_crawler.py`)
- Handles website crawling and content extraction using Beautiful Soup and Trafilatura
- Implements respectful crawling with configurable delays and rate limiting
- Provides intelligent link discovery with domain scoping
- Manages cache file generation with human-readable naming

**SimpleRAGEngine** (`simple_rag_engine.py`)
- Lightweight in-memory RAG implementation for maximum compatibility
- Implements smart text chunking with semantic boundary awareness
- Provides keyword-based search with relevance scoring
- Optimized for environments with dependency constraints

**Alternative: WebRAGEngine** (`web_rag_engine.py`)
- Advanced ChromaDB-based implementation for enhanced semantic search
- Supports vector embeddings with persistent storage
- Provides more sophisticated similarity matching
- Requires additional system dependencies

### Data Flow
1. **Crawling**: Extract clean content from website pages using Trafilatura
2. **Processing**: Intelligently chunk text with 1000-token segments and 100-token overlap
3. **Storage**: Save to local JSON cache with metadata and optional ChromaDB vector storage
4. **Analysis**: Use RAG engine for AI-powered question answering with GPT-4o
5. **Visualization**: Display results through interactive analytics tabs and charts

### Configuration Management
- Centralized settings in `config/settings.py`
- Environment variable integration
- Automatic directory creation
- Configurable crawling and RAG parameters

## 🚦 Performance Considerations

### Optimization Features
- **Content Caching**: Reuse processed content and metadata between sessions
- **Smart Chunking**: Token-aware content segmentation with semantic boundary respect
- **Efficient Search**: Keyword-based relevance scoring for fast content lookup
- **Progressive Loading**: Real-time progress tracking with live metrics and ETAs

### Resource Management
- **Rate Limiting**: Respectful website crawling (1-5 second delays)
- **Memory Efficiency**: Streaming content processing
- **Storage Optimization**: Compressed cache files with metadata
- **Token Management**: Optimized prompt engineering for cost efficiency

## 🔍 Troubleshooting

### Common Issues
- **Missing API Key**: Set OPENAI_API_KEY environment variable in Replit Secrets
- **Crawling Failures**: Check URL accessibility, robots.txt compliance, or increase delay
- **Empty Results**: Verify website has extractable text content (not just images/videos)
- **Performance Issues**: Reduce page limit (try 10-25 pages) or increase delay (2-3 seconds)
- **Dependency Conflicts**: Application uses SimpleRAGEngine as fallback for maximum compatibility

### Debug Features
- Real-time progress tracking during crawling
- Confidence scoring for answer reliability
- Source attribution with similarity scores
- Comprehensive error messaging

## 📄 License & Credits

Built with:
- **Streamlit**: Web application framework and UI components
- **OpenAI GPT-4o**: AI-powered question answering and content analysis
- **Trafilatura**: Clean content extraction from web pages
- **Beautiful Soup**: HTML parsing and link discovery
- **Tiktoken**: Token counting and text chunking
- **Optional ChromaDB**: Vector database for advanced semantic search
- **Plotly**: Interactive visualizations and charts

## 🔄 Recent Updates

### Current Implementation (June 2025)
- **Dual RAG Architecture**: ChromaDB-based engine with SimpleRAGEngine fallback for compatibility
- **Enhanced Analytics**: Complete Analytics/Search/Content tabs with interactive visualizations
- **Improved Caching**: Human-readable cache filenames with embedded metadata
- **Smart Crawling**: Enhanced link discovery with 30+ links per page and better URL filtering
- **Real-time Progress**: Live crawling metrics with ETA calculations and performance tracking
- **Dependency Resilience**: Automatic fallback to lightweight engine when advanced dependencies unavailable

See `replit.md` for detailed changelog and technical architecture decisions.