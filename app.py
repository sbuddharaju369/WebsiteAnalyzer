"""
Web Content Analyzer - Streamlined Version
A comprehensive Streamlit application for crawling and analyzing web content using AI-powered RAG
"""
import streamlit as st
import os
import json
import time
from datetime import datetime
from urllib.parse import urlparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import requests
from bs4 import BeautifulSoup
import trafilatura
import tiktoken
import openai

# Configure Streamlit page
st.set_page_config(
    page_title="Web Content Analyzer",
    page_icon="🕷️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration constants
DEFAULT_VERBOSITY = 'concise'
DEFAULT_MAX_PAGES = 50
DEFAULT_DELAY = 1.0


class SimpleWebCrawler:
    """Simplified web crawler without external dependencies"""
    
    def __init__(self, max_pages: int = 50, delay: float = 1.0):
        self.max_pages = max_pages
        self.delay = delay
        self.crawled_content = []
        
    def crawl_website(self, start_url: str, progress_callback=None):
        """Crawl website and extract content"""
        visited_urls = set()
        to_visit = [start_url]
        base_domain = urlparse(start_url).netloc
        
        pages_visited = 0
        pages_extracted = 0
        
        while to_visit and pages_visited < self.max_pages:
            url = to_visit.pop(0)
            
            if url in visited_urls:
                continue
                
            visited_urls.add(url)
            pages_visited += 1
            
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                
                # Extract content using trafilatura
                text_content = trafilatura.extract(response.text)
                
                if text_content and len(text_content.strip()) > 100:
                    # Extract title
                    soup = BeautifulSoup(response.text, 'html.parser')
                    title = soup.title.string if soup.title else url
                    
                    page_data = {
                        'url': url,
                        'title': title.strip(),
                        'content': text_content,
                        'word_count': len(text_content.split()),
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    self.crawled_content.append(page_data)
                    pages_extracted += 1
                    
                    # Find more links
                    links = soup.find_all('a', href=True)
                    for link in links[:5]:  # Limit links per page
                        href = link['href']
                        if href.startswith('/'):
                            full_url = f"https://{base_domain}{href}"
                        elif href.startswith('http') and base_domain in href:
                            full_url = href
                        else:
                            continue
                            
                        if full_url not in visited_urls and full_url not in to_visit:
                            to_visit.append(full_url)
                
                if progress_callback:
                    progress_callback(pages_visited, pages_extracted, url, title)
                
                time.sleep(self.delay)
                
            except Exception as e:
                st.warning(f"Error crawling {url}: {str(e)}")
                continue
        
        return self.crawled_content
    
    def get_crawl_stats(self):
        """Get statistics about crawled content"""
        if not self.crawled_content:
            return {}
            
        total_words = sum(page.get('word_count', 0) for page in self.crawled_content)
        avg_words = total_words / len(self.crawled_content) if self.crawled_content else 0
        
        return {
            'total_pages': len(self.crawled_content),
            'total_words': total_words,
            'average_words_per_page': avg_words
        }


class SimpleRAGEngine:
    """RAG engine with intelligent chunking and semantic search"""
    
    def __init__(self):
        self.content = []
        self.chunks = []
        self.chunk_metadata = []
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Check OpenAI API key
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found")
            
        openai.api_key = self.api_key
    
    def _smart_chunk_text(self, text: str, max_tokens: int = 1000, overlap_tokens: int = 100) -> List[str]:
        """Intelligently chunk text based on semantic boundaries"""
        # Split by paragraphs first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph_tokens = len(self.encoding.encode(paragraph))
            current_tokens = len(self.encoding.encode(current_chunk))
            
            # If paragraph is too large, split it by sentences
            if paragraph_tokens > max_tokens:
                sentences = [s.strip() + '.' for s in paragraph.split('.') if s.strip()]
                
                for sentence in sentences:
                    sentence_tokens = len(self.encoding.encode(sentence))
                    
                    if current_tokens + sentence_tokens <= max_tokens:
                        current_chunk += " " + sentence if current_chunk else sentence
                        current_tokens = len(self.encoding.encode(current_chunk))
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence
                        current_tokens = sentence_tokens
            else:
                # Check if we can add this paragraph to current chunk
                if current_tokens + paragraph_tokens <= max_tokens:
                    current_chunk += "\n\n" + paragraph if current_chunk else paragraph
                else:
                    # Save current chunk and start new one
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = paragraph
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Add overlap between chunks
        if len(chunks) > 1 and overlap_tokens > 0:
            overlapped_chunks = []
            
            for i, chunk in enumerate(chunks):
                if i == 0:
                    overlapped_chunks.append(chunk)
                else:
                    # Get overlap from previous chunk
                    prev_chunk = chunks[i-1]
                    words = prev_chunk.split()
                    overlap_text = ""
                    
                    # Build overlap from the end
                    for word in reversed(words):
                        test_text = word + " " + overlap_text if overlap_text else word
                        if len(self.encoding.encode(test_text)) <= overlap_tokens:
                            overlap_text = test_text
                        else:
                            break
                    
                    if overlap_text:
                        overlapped_chunk = overlap_text.strip() + "\n\n" + chunk
                        overlapped_chunks.append(overlapped_chunk)
                    else:
                        overlapped_chunks.append(chunk)
            
            return overlapped_chunks
        
        return chunks

    def process_web_content(self, web_content: List[Dict[str, Any]], domain: str = None):
        """Process and store web content with intelligent chunking"""
        self.content = web_content
        self.chunks = []
        self.chunk_metadata = []
        
        chunk_id = 0
        
        # Process each page with smart chunking
        for page in web_content:
            content = page.get('content', '')
            if not content.strip():
                continue
            
            # Use cached chunks if available
            if 'chunks' in page and page['chunks']:
                page_chunks = page['chunks']
            else:
                page_chunks = self._smart_chunk_text(content, max_tokens=1000, overlap_tokens=100)
                # Cache chunks for future use
                page['chunks'] = page_chunks
            
            # Store chunks with metadata
            for chunk in page_chunks:
                self.chunks.append(chunk)
                self.chunk_metadata.append({
                    'url': page.get('url', ''),
                    'title': page.get('title', ''),
                    'domain': domain or page.get('domain', ''),
                    'word_count': len(chunk.split()),
                    'chunk_id': chunk_id,
                    'page_index': len(self.chunks) - 1
                })
                chunk_id += 1
        
        st.success(f"Processed {len(web_content)} pages into {len(self.chunks)} intelligent chunks for analysis")
    
    def _semantic_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Perform semantic search across chunks"""
        if not self.chunks:
            return []
        
        # Simple keyword-based relevance scoring
        query_words = set(query.lower().split())
        scored_chunks = []
        
        for i, chunk in enumerate(self.chunks):
            chunk_words = set(chunk.lower().split())
            # Calculate relevance score based on word overlap
            overlap = len(query_words.intersection(chunk_words))
            total_words = len(query_words.union(chunk_words))
            relevance = overlap / total_words if total_words > 0 else 0
            
            # Boost score if query terms appear close together
            if overlap > 0:
                chunk_lower = chunk.lower()
                for word in query_words:
                    if word in chunk_lower:
                        relevance += 0.1
            
            if relevance > 0:
                scored_chunks.append({
                    'chunk': chunk,
                    'metadata': self.chunk_metadata[i],
                    'relevance': relevance
                })
        
        # Sort by relevance and return top k
        scored_chunks.sort(key=lambda x: x['relevance'], reverse=True)
        return scored_chunks[:k]

    def analyze_content(self, question: str, verbosity: str = 'concise'):
        """Analyze content with semantic search and provide answers"""
        if not self.chunks:
            return {"answer": "No content available for analysis"}
        
        # Find most relevant chunks using semantic search
        relevant_chunks = self._semantic_search(question, k=5)
        
        if not relevant_chunks:
            # Fallback to first few chunks if no relevant ones found
            context_parts = []
            for i, chunk in enumerate(self.chunks[:3]):
                metadata = self.chunk_metadata[i]
                context_parts.append(f"Source: {metadata['title']}\nContent: {chunk[:800]}...")
        else:
            context_parts = []
            sources = []
            for item in relevant_chunks:
                chunk = item['chunk']
                metadata = item['metadata']
                context_parts.append(f"Source: {metadata['title']}\nContent: {chunk}")
                sources.append({
                    'title': metadata['title'],
                    'url': metadata['url'],
                    'relevance': f"{item['relevance']:.2f}"
                })
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Create prompt based on verbosity
        system_prompt = "You are a helpful assistant that analyzes website content and provides accurate, well-sourced answers based solely on the provided information. Always cite which sources you're drawing from."
        
        if verbosity == 'concise':
            user_prompt = f"Based on this website content, provide a brief, focused answer to: {question}\n\nContent:\n{context}"
            max_tokens = 400
        elif verbosity == 'comprehensive':
            user_prompt = f"Based on this website content, provide a detailed, thorough analysis for: {question}. Include specific examples and details from the sources.\n\nContent:\n{context}"
            max_tokens = 1200
        else:  # balanced
            user_prompt = f"Based on this website content, provide a balanced, informative answer to: {question}\n\nContent:\n{context}"
            max_tokens = 800
        
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.3
            )
            
            # Calculate confidence based on relevance of sources
            if relevant_chunks:
                avg_relevance = sum(item['relevance'] for item in relevant_chunks) / len(relevant_chunks)
                confidence = min(avg_relevance * 2, 1.0)  # Scale to 0-1
            else:
                confidence = 0.5  # Medium confidence for fallback
            
            return {
                "answer": response.choices[0].message.content,
                "sources": sources if relevant_chunks else [{"title": meta["title"], "url": meta["url"]} for meta in self.chunk_metadata[:3]],
                "confidence": confidence,
                "chunks_used": len(relevant_chunks) if relevant_chunks else 3
            }
            
        except Exception as e:
            return {"answer": f"Error analyzing content: {str(e)}", "sources": [], "confidence": 0.0}
    
    def suggest_questions(self):
        """Suggest relevant questions"""
        return [
            "What is this website about?",
            "What products or services are offered?",
            "How can I contact them?",
            "What are the main topics covered?",
            "What are the key features mentioned?"
        ]
    
    def get_content_summary(self):
        """Get content summary"""
        return {
            "total_chunks": len(self.chunks),
            "total_pages": len(self.content),
            "avg_chunk_size": sum(len(chunk.split()) for chunk in self.chunks) / len(self.chunks) if self.chunks else 0
        }


class WebContentAnalyzer:
    """Main application class"""
    
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state"""
        defaults = {
            'crawler': None,
            'rag_engine': None,
            'crawled_content': [],
            'crawl_stats': {},
            'current_domain': None,
            'crawl_in_progress': False,
            'answer_verbosity': DEFAULT_VERBOSITY,
            'active_drawer': None
        }
        
        for key, default in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default
    
    def check_openai_key(self) -> bool:
        """Check if OpenAI API key is available"""
        return bool(os.getenv('OPENAI_API_KEY'))
    
    def render_sidebar(self):
        """Render sidebar interface"""
        with st.sidebar:
            st.markdown("### Navigation")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🕷️  \nWeb", use_container_width=True):
                    st.session_state.active_drawer = 'crawler' if st.session_state.active_drawer != 'crawler' else None
            
            with col2:
                if st.button("💾  \nCache", use_container_width=True):
                    st.session_state.active_drawer = 'cache' if st.session_state.active_drawer != 'cache' else None
            
            with col3:
                if st.button("📊  \nContent", use_container_width=True):
                    st.session_state.active_drawer = 'overview' if st.session_state.active_drawer != 'overview' else None
            
            st.divider()
            
            if st.session_state.active_drawer == 'crawler':
                self.render_crawler_drawer()
            elif st.session_state.active_drawer == 'cache':
                self.render_cache_drawer()
            elif st.session_state.active_drawer == 'overview':
                self.render_overview_drawer()
    
    def render_crawler_drawer(self):
        """Render web crawler interface"""
        st.markdown("#### 🕷️ Web Crawler")
        
        url_input = st.text_input(
            "Website URL",
            placeholder="https://example.com",
            help="Enter the URL of the website you want to analyze"
        )
        
        with st.expander("⚙️ Crawling Settings"):
            max_pages = st.slider("Maximum pages to crawl", 1, 100, DEFAULT_MAX_PAGES)
            delay = st.slider("Delay between requests (seconds)", 0.5, 5.0, DEFAULT_DELAY, 0.5)
        
        if st.button("🔍 Start Crawling", disabled=st.session_state.crawl_in_progress):
            if url_input:
                self.start_crawling(url_input, max_pages, delay)
            else:
                st.warning("Please enter a valid URL")
    
    def render_cache_drawer(self):
        """Render cache management interface"""
        st.markdown("#### 💾 Cache Management")
        
        cache_files = self.get_cache_files()
        
        if cache_files:
            selected_cache = st.selectbox(
                "Load from cache:",
                options=[None] + cache_files,
                format_func=lambda x: self.format_cache_name(x) if x else "Select cache file..."
            )
            
            if st.button("📂 Load Cache") and selected_cache:
                self.load_cached_content(selected_cache)
        else:
            st.info("No cache files found")
    
    def render_overview_drawer(self):
        """Render content overview interface"""
        if st.session_state.crawled_content:
            st.markdown("#### 📊 Content Overview")
            
            stats = st.session_state.crawl_stats
            if stats:
                st.metric("Total Pages", stats.get('total_pages', 0))
                st.metric("Total Words", f"{stats.get('total_words', 0):,}")
                st.metric("Avg Words/Page", f"{stats.get('average_words_per_page', 0):.0f}")
        else:
            st.info("No content loaded yet")
    
    def start_crawling(self, url: str, max_pages: int, delay: float):
        """Start crawling process"""
        st.session_state.crawl_in_progress = True
        domain = urlparse(url).netloc
        st.session_state.current_domain = domain
        
        with st.sidebar:
            st.markdown("---")
            st.markdown("### 🕷️ Crawling Progress")
            
            pages_visited_placeholder = st.empty()
            pages_extracted_placeholder = st.empty()
            progress_bar_placeholder = st.empty()
            current_page_placeholder = st.empty()
            
            start_time = datetime.now()
        
        crawler = SimpleWebCrawler(max_pages=max_pages, delay=delay)
        
        def progress_callback(visited, extracted, current_url=None, page_title=None):
            pages_visited_placeholder.metric("📄 Pages Visited", visited)
            pages_extracted_placeholder.metric("✅ Content Extracted", extracted)
            
            progress = min(visited / max_pages, 1.0)
            progress_bar_placeholder.progress(progress, text=f"{progress:.0%} Complete")
            
            if current_url and page_title:
                title_display = page_title[:40] + "..." if len(page_title) > 40 else page_title
                current_page_placeholder.info(f"🔍 **Current:** {title_display}")
        
        content = crawler.crawl_website(url, progress_callback=progress_callback)
        
        with st.sidebar:
            st.markdown("---")
            if content:
                st.success(f"✅ Crawling completed: {len(content)} pages")
            else:
                st.error("❌ Crawling failed")
        
        if content:
            st.session_state.crawled_content = content
            stats = crawler.get_crawl_stats()
            st.session_state.crawl_stats = stats
            
            # Initialize RAG engine
            try:
                rag_engine = SimpleRAGEngine()
                rag_engine.process_web_content(content, domain)
                st.session_state.rag_engine = rag_engine
            except Exception as e:
                st.error(f"Error initializing AI analysis: {str(e)}")
                st.session_state.rag_engine = None
            
            # Save to cache
            cache_file = self.save_cache(content, domain)
            st.info(f"Cache saved: {cache_file}")
            
            st.success(f"Successfully analyzed {len(content)} pages!")
        else:
            st.error("Failed to crawl website. Please check the URL and try again.")
        
        st.session_state.crawl_in_progress = False
    
    def get_cache_files(self) -> List[str]:
        """Get list of cache files"""
        cache_dir = Path("data/cache")
        if not cache_dir.exists():
            return []
        
        return sorted([f.name for f in cache_dir.glob("*.json")], reverse=True)
    
    def format_cache_name(self, filename: str) -> str:
        """Format cache filename for display"""
        if not filename:
            return ""
        return filename.replace('.json', '').replace('_', ' - ')
    
    def save_cache(self, content: List[Dict[str, Any]], domain: str) -> str:
        """Save content to cache"""
        cache_dir = Path("data/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now()
        filename = f"{domain}_{timestamp.strftime('%Y%m%d_%H%M%S')}_{len(content)}pages.json"
        filepath = cache_dir / filename
        
        cache_data = {
            'domain': domain,
            'timestamp': timestamp.isoformat(),
            'total_pages': len(content),
            'content': content
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        
        return filename
    
    def load_cached_content(self, filename: str):
        """Load content from cache"""
        try:
            cache_file = Path("data/cache") / filename
            
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            if cache_data and cache_data.get('content'):
                content = cache_data['content']
                st.session_state.crawled_content = content
                domain = cache_data.get('domain', 'unknown')
                st.session_state.current_domain = domain
                
                # Update stats
                if content:
                    total_words = sum(page.get('word_count', 0) for page in content)
                    avg_words = total_words / len(content) if content else 0
                    
                    stats = {
                        'total_pages': len(content),
                        'total_words': total_words,
                        'average_words_per_page': avg_words
                    }
                    st.session_state.crawl_stats = stats
                
                # Initialize RAG engine
                try:
                    rag_engine = SimpleRAGEngine()
                    rag_engine.process_web_content(content, domain)
                    st.session_state.rag_engine = rag_engine
                except Exception as e:
                    st.error(f"Error initializing AI analysis: {str(e)}")
                    st.session_state.rag_engine = None
                
                st.success(f"Loaded {len(content)} pages from cache!")
            else:
                st.error("Cache file is empty or corrupted")
        except Exception as e:
            st.error(f"Error loading cache: {str(e)}")
    
    def render_main_content(self):
        """Render main content area"""
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        ">
            <h1 style="
                color: white;
                font-size: 2.5rem;
                font-weight: 700;
                margin: 0;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            ">
                🕷️ Web Content Analyzer
            </h1>
            <p style="
                color: rgba(255,255,255,0.9);
                font-size: 1.2rem;
                margin: 0.5rem 0 0 0;
                font-weight: 300;
            ">
                Transform any website into intelligent, searchable knowledge
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if not st.session_state.crawled_content:
            self.render_welcome_screen()
        else:
            self.render_analysis_interface()
    
    def render_welcome_screen(self):
        """Render welcome screen"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### 🚀 Key Features
            - **Intelligent Web Crawling**: Respects rate limits and extracts clean content
            - **AI-Powered Analysis**: GPT-4 powered question answering
            - **Content Caching**: Save and reuse crawled content
            - **Real-time Progress**: Track crawling progress with live updates
            - **Configurable Responses**: Choose answer detail level
            """)
        
        with col2:
            st.markdown("""
            #### 💡 Example Questions
            - "What products does this company offer?"
            - "What are the main topics covered?"
            - "How can I contact them?"
            - "What are their pricing options?"
            - "What services are available?"
            """)
        
        st.markdown("---")
        st.info("👆 Enter a website URL in the sidebar to get started!")
    
    def render_analysis_interface(self):
        """Render analysis interface"""
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            padding: 1.5rem;
            border-radius: 8px;
            margin-bottom: 1.5rem;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        ">
            <h2 style="
                color: white;
                font-size: 1.8rem;
                font-weight: 600;
                margin: 0;
                text-shadow: 1px 1px 3px rgba(0,0,0,0.3);
            ">
                💬 Ask Questions about {st.session_state.current_domain}
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.rag_engine:
            suggested_questions = st.session_state.rag_engine.suggest_questions()
            
            with st.expander("💡 Suggested Questions"):
                for question in suggested_questions:
                    if st.button(f"❓ {question}", key=f"suggested_{hash(question)}"):
                        st.session_state.current_question = question
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("**Your question:**")
            question = st.text_area(
                "Your question:",
                placeholder="Ask anything about the website content...",
                height=80,
                label_visibility="collapsed"
            )
        with col2:
            st.markdown("**Answer Style:**")
            verbosity = st.selectbox(
                "Response level:",
                options=['concise', 'balanced', 'comprehensive'],
                index=['concise', 'balanced', 'comprehensive'].index(st.session_state.answer_verbosity),
                format_func=lambda x: {
                    'concise': '🎯 Concise',
                    'balanced': '⚖️ Balanced', 
                    'comprehensive': '📖 Detailed'
                }[x],
                help="Choose how detailed you want the answers to be",
                label_visibility="collapsed"
            )
            
            if verbosity != st.session_state.answer_verbosity:
                st.session_state.answer_verbosity = verbosity
        
        if hasattr(st.session_state, 'current_question'):
            question = st.session_state.current_question
            delattr(st.session_state, 'current_question')
        
        if question and st.session_state.rag_engine:
            with st.spinner("Analyzing content..."):
                result = st.session_state.rag_engine.analyze_content(question, verbosity=verbosity)
                
                verbosity_indicators = {
                    'concise': '🎯 Concise',
                    'balanced': '⚖️ Balanced',
                    'comprehensive': '📖 Detailed'
                }
                
                st.markdown(f"### Answer ({verbosity_indicators[verbosity]})")
                st.markdown(result['answer'])
                
                # Display confidence and sources
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if 'sources' in result and result['sources']:
                        with st.expander("📚 Sources"):
                            for source in result['sources']:
                                relevance_text = f" (Relevance: {source.get('relevance', 'N/A')})" if 'relevance' in source else ""
                                st.markdown(f"- **{source['title']}**{relevance_text}")
                                st.markdown(f"  {source['url']}")
                
                with col2:
                    if 'confidence' in result:
                        confidence = result['confidence']
                        if confidence >= 0.8:
                            confidence_text = "Very Reliable"
                            confidence_color = "green"
                        elif confidence >= 0.6:
                            confidence_text = "Mostly Reliable"  
                            confidence_color = "orange"
                        elif confidence >= 0.4:
                            confidence_text = "Moderately Reliable"
                            confidence_color = "orange"
                        else:
                            confidence_text = "Limited Reliability"
                            confidence_color = "red"
                        
                        st.markdown(f"**Confidence:** :{confidence_color}[{confidence_text}]")
                        
                        if 'chunks_used' in result:
                            st.markdown(f"**Chunks analyzed:** {result['chunks_used']}")
    
    def run(self):
        """Run the application"""
        if not self.check_openai_key():
            st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
            st.stop()
        
        self.render_sidebar()
        self.render_main_content()


def main():
    """Main entry point"""
    app = WebContentAnalyzer()
    app.run()


if __name__ == "__main__":
    main()