"""
Application configuration settings
"""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
CHROMA_DIR = DATA_DIR / "chroma"

# Ensure directories exist
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Application Settings
DEFAULT_MAX_PAGES = 25
DEFAULT_DELAY = 1.0
DEFAULT_CHUNK_SIZE = 600
DEFAULT_OVERLAP = 100
DEFAULT_VERBOSITY = "concise"

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL")

# Streamlit Configuration
STREAMLIT_CONFIG = {
    "server": {
        "headless": True,
        "address": "0.0.0.0",
        "port": 5000
    }
}

# Web Crawler Settings
CRAWLER_SETTINGS = {
    "max_pages": DEFAULT_MAX_PAGES,
    "delay": DEFAULT_DELAY,
    "timeout": 10,
    "user_agent": "Mozilla/5.0 (compatible; WebContentAnalyzer/1.0)",
    "respect_robots_txt": True
}

# RAG Engine Settings
RAG_SETTINGS = {
    "chunk_size": DEFAULT_CHUNK_SIZE,
    "overlap": DEFAULT_OVERLAP,
    "similarity_threshold": 0.3,
    "max_context_length": 4000,
    "embedding_model": "text-embedding-3-small"
}