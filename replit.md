# Verizon Plan Assistant

## Overview

This is a Streamlit-based web application that serves as an intelligent assistant for Verizon telecommunications plans. The application uses web scraping to gather current plan information from Verizon's website and implements a Retrieval-Augmented Generation (RAG) system powered by OpenAI's GPT models to provide intelligent responses to user queries about Verizon plans.

## System Architecture

The application follows a modular architecture with clear separation of concerns:

1. **Frontend**: Streamlit-based web interface for user interaction
2. **Data Collection**: Web scraping module for gathering plan information
3. **AI/RAG Engine**: OpenAI-powered question-answering system with vector search
4. **Data Storage**: PostgreSQL database with file-based JSON backup
5. **Testing Framework**: Accuracy verification and response validation system
6. **Deployment**: Configured for Replit's autoscale deployment

## Key Components

### 1. Web Scraping (`scraper.py`)
- **Purpose**: Extracts current plan information from Verizon's website
- **Technology**: Beautiful Soup, Trafilatura, Requests
- **Features**: 
  - Rate-limited requests to respect website policies
  - Multiple plan categories (mobile, internet, prepaid, bundles)
  - Content extraction and cleaning
- **Architecture Decision**: Chose trafilatura for robust content extraction over basic BeautifulSoup parsing to handle dynamic content better

### 2. RAG Engine (`chroma_engine.py`)
- **Purpose**: Provides intelligent question-answering using scraped data
- **Technology**: OpenAI GPT models, ChromaDB vector database, text embeddings
- **Features**:
  - Text chunking with overlap for better context preservation
  - Semantic similarity search with metadata filtering
  - Persistent vector storage with query history
  - Advanced filtering by category and content
- **Architecture Decision**: Migrated from FAISS to ChromaDB for better metadata handling, persistence, and filtering capabilities

### 3. Streamlit Frontend (`app.py`)
- **Purpose**: User interface for interacting with the system
- **Features**:
  - Real-time data scraping capabilities
  - Interactive chat interface
  - Session state management
  - Data freshness indicators
- **Architecture Decision**: Chose Streamlit for rapid prototyping and built-in session management over more complex frameworks

### 4. Database Layer (`database_simple.py`)
- **Purpose**: PostgreSQL database operations for persistent storage
- **Technology**: PostgreSQL, psycopg2
- **Features**:
  - Plan data storage with full metadata
  - Query history tracking with performance metrics
  - Scraping session management
  - Advanced search and filtering capabilities
- **Architecture Decision**: Used direct psycopg2 for simplicity and reliability over ORM solutions

### 5. Testing Framework (`testing_utils.py`)
- **Purpose**: Response accuracy verification and system validation
- **Features**:
  - Source authenticity verification
  - Price and plan name validation
  - Accuracy scoring and reporting
  - Automated test question generation
- **Architecture Decision**: Built custom testing framework to validate AI responses against scraped data

### 6. Utility Functions (`utils.py`)
- **Purpose**: Data persistence and text processing utilities
- **Features**:
  - JSON-based data storage with metadata
  - Text cleaning and preprocessing
  - File I/O operations with error handling

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

- June 15, 2025: Initial setup with basic RAG functionality
- June 15, 2025: Added PostgreSQL database integration with comprehensive data persistence
- June 15, 2025: Implemented accuracy testing framework for response validation
- June 15, 2025: Added database search, query history, and performance tracking features

## User Preferences

Preferred communication style: Simple, everyday language.