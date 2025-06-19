# Web Content Analyzer

A comprehensive Streamlit-based web application that transforms any publicly accessible website into intelligent, searchable knowledge using AI-powered content analysis.

## 🏗️ Project Structure

```
web-content-analyzer/
├── app.py                          # Main Streamlit application
├── config/
│   └── settings.py                 # Application configuration and constants
├── src/
│   ├── core/                       # Core business logic
│   │   ├── crawler.py              # Web crawling and content extraction
│   │   └── rag_engine.py           # RAG processing and AI analysis
│   ├── ui/                         # User interface components
│   │   └── visualizations.py      # Charts, graphs, and UI helpers
│   └── utils/                      # Utility modules
│       └── cache_manager.py       # Cache file management
├── data/                           # Data storage (auto-created)
│   ├── cache/                      # Cached crawled content
│   └── chroma/                     # ChromaDB vector database
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
2. **Smart Chunking**: Breaks content into 1000-token chunks with 100-token overlap
3. **Semantic Boundaries**: Respects paragraph and sentence boundaries for better context
4. **Metadata Preservation**: Maintains URL, title, and source information for each chunk
5. **Caching System**: Saves processed content with chunks for future analysis

### Usage Instructions
To start recursive crawling:
1. Enter any website URL in the sidebar's Web Crawler section
2. Configure your settings (max pages: 1-100, delay: 0.5-5 seconds)
3. Click "Start Crawling" to begin automatic discovery
4. Monitor real-time progress as pages are discovered and processed
5. The system will automatically find and process all accessible pages within the domain

### AI-Powered Analysis
- **RAG Engine**: Retrieval-Augmented Generation using OpenAI GPT-4
- **Semantic Search**: ChromaDB vector database for similarity search
- **Smart Chunking**: Token-aware content segmentation with overlap
- **Embedding Caching**: Reuse AI processing for faster repeated analysis

### User Interface
- **Collapsible Sidebar**: Organized navigation with drawer system
- **Interactive Visualizations**: Plotly charts and network graphs
- **Multi-Tab Analytics**: Content analysis, search, and raw data views
- **Configurable Verbosity**: Concise, balanced, or comprehensive answers

### Data Management
- **Persistent Storage**: ChromaDB for embeddings, local cache for content
- **Cache Management**: Automatic file organization with human-readable names
- **Coverage Analysis**: Website size estimation using sitemaps
- **Source Attribution**: Confidence scoring and relevance tracking

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

**WebCrawler** (`src/core/crawler.py`)
- Handles website crawling and content extraction
- Implements rate limiting and robots.txt compliance
- Provides website size estimation using sitemaps
- Manages cache file generation with embeddings

**WebRAGEngine** (`src/core/rag_engine.py`)
- Processes content for AI analysis using ChromaDB
- Implements smart text chunking with token awareness
- Provides semantic search and question answering
- Manages embedding caching for performance optimization

**CacheManager** (`src/utils/cache_manager.py`)
- Handles cache file storage and retrieval
- Provides human-readable file naming
- Manages cache cleanup and organization

### Data Flow
1. **Crawling**: Extract content from website pages
2. **Processing**: Chunk text and generate embeddings
3. **Storage**: Save to ChromaDB and local cache
4. **Analysis**: Use RAG for question answering
5. **Visualization**: Display results and analytics

### Configuration Management
- Centralized settings in `config/settings.py`
- Environment variable integration
- Automatic directory creation
- Configurable crawling and RAG parameters

## 🚦 Performance Considerations

### Optimization Features
- **Embedding Caching**: Reuse AI processing between sessions
- **Smart Chunking**: Token-aware content segmentation
- **Connection Pooling**: Efficient HTTP request management
- **Progressive Loading**: Real-time progress tracking

### Resource Management
- **Rate Limiting**: Respectful website crawling (1-5 second delays)
- **Memory Efficiency**: Streaming content processing
- **Storage Optimization**: Compressed cache files with metadata
- **Token Management**: Optimized prompt engineering for cost efficiency

## 🔍 Troubleshooting

### Common Issues
- **Missing API Key**: Set OPENAI_API_KEY environment variable
- **Crawling Failures**: Check URL accessibility and robots.txt
- **Empty Results**: Verify website has extractable text content
- **Performance Issues**: Reduce page limit or increase delay

### Debug Features
- Real-time progress tracking during crawling
- Confidence scoring for answer reliability
- Source attribution with similarity scores
- Comprehensive error messaging

## 📄 License & Credits

Built with:
- **Streamlit**: Web application framework
- **OpenAI**: GPT-4 and text embeddings
- **ChromaDB**: Vector database for semantic search
- **Trafilatura**: Content extraction
- **Plotly**: Interactive visualizations

## 🔄 Changelog

See `replit.md` for detailed changelog and recent improvements.