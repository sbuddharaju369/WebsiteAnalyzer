# Web Content Analyzer

## Overview

This is a comprehensive Streamlit-based web application that transforms any publicly accessible website into intelligent, searchable knowledge using AI-powered content analysis. The application crawls websites, extracts clean content, creates semantic embeddings, and provides RAG-powered insights and question-answering capabilities.

## System Architecture

The application follows a modular architecture with clear separation of concerns:

1. **Frontend**: Streamlit-based web interface with advanced analytics
2. **Web Crawling**: Intelligent website crawling and content extraction
3. **AI/RAG Engine**: OpenAI-powered content analysis with vector search
4. **Data Storage**: ChromaDB vector database with local JSON caching
5. **Visualization**: Interactive charts and network graphs
6. **Deployment**: Configured for Replit's autoscale deployment

## Key Components

### 1. Web Crawler (`web_crawler.py`)
- **Purpose**: Intelligently crawls and extracts content from any website
- **Technology**: Beautiful Soup, Trafilatura, Requests
- **Features**: 
  - Respects robots.txt and implements rate limiting
  - Domain-scoped crawling to stay within target site
  - Content filtering to skip non-text files
  - Metadata extraction (titles, descriptions, headings)
  - Local JSON caching with timestamps
- **Architecture Decision**: Uses trafilatura for clean content extraction and implements smart URL filtering

### 2. Web RAG Engine (`web_rag_engine.py`)
- **Purpose**: Processes web content for intelligent question-answering
- **Technology**: OpenAI GPT-4o, ChromaDB vector database, tiktoken
- **Features**:
  - Smart text chunking based on semantic boundaries
  - Token-aware processing for optimal embedding
  - Persistent vector storage with rich metadata
  - Confidence scoring and source attribution
  - Context-aware analysis with domain insights
- **Architecture Decision**: Persistent ChromaDB for reliability with intelligent chunking strategy

### 3. Enhanced Streamlit Frontend (`app.py`)
- **Purpose**: Comprehensive web interface for content analysis
- **Features**:
  - Interactive crawling controls with progress tracking
  - AI-powered question answering with suggested queries
  - Visual analytics with charts and network graphs
  - Cache management and content browsing
  - Semantic search across all content
- **Architecture Decision**: Wide layout with tabbed interface for rich user experience

### 4. Data Visualization
- **Technology**: Plotly, Streamlit-agraph, Pandas
- **Features**:
  - Word count distribution analysis
  - Site structure visualization
  - Content relationship network graphs
  - Interactive data tables
  - Performance metrics dashboard

### 5. Caching System
- **Purpose**: Local storage for crawled content and performance
- **Features**:
  - Automatic cache generation with metadata
  - Cache loading and management interface
  - Domain-based file organization
  - Timestamp tracking for freshness

## Data Flow

1. **Data Collection**: Scraper fetches plan information from Verizon's website
2. **Data Processing**: Raw HTML content is cleaned and structured
3. **Storage**: Processed data is saved to PostgreSQL database and JSON backup files
4. **Indexing**: RAG engine chunks text and creates vector embeddings for semantic search
5. **Query Processing**: User questions are embedded and matched against stored data
6. **Response Generation**: Relevant context is provided to OpenAI for answer generation
7. **Validation**: Accuracy testing framework verifies responses against source data
8. **Persistence**: All queries, responses, and accuracy metrics are stored in database

## External Dependencies

### Core Dependencies
- **OpenAI API**: For text embeddings and chat completions
- **Streamlit**: Web framework for the user interface
- **ChromaDB**: Vector database for semantic search and embeddings storage
- **Beautiful Soup**: HTML parsing and web scraping
- **Trafilatura**: Content extraction from web pages
- **Requests**: HTTP client for web scraping
- **PostgreSQL**: Database for persistent data storage
- **psycopg2-binary**: PostgreSQL adapter for Python

### Environment Requirements
- **Python 3.11+**: Runtime environment
- **OpenAI API Key**: Required environment variable for AI functionality
- **PostgreSQL Database**: Provided by Replit environment for data persistence

## Deployment Strategy

- **Platform**: Replit with autoscale deployment
- **Configuration**: Uses .replit file for deployment settings
- **Port**: Configured to run on port 5000
- **Runtime**: Python 3.11 with Nix package management
- **Scalability**: Autoscale deployment handles traffic variations

## Changelog

- June 15, 2025: Initial setup with basic RAG functionality using FAISS
- June 15, 2025: Added PostgreSQL database integration with comprehensive data persistence
- June 15, 2025: Implemented accuracy testing framework for response validation
- June 15, 2025: Added database search, query history, and performance tracking features
- June 15, 2025: Migrated from FAISS to ChromaDB for enhanced vector database capabilities

## User Preferences

Preferred communication style: Simple, everyday language.