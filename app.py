"""
Web Content Analyzer - Temporary Version Without Pandas
A comprehensive Streamlit application for crawling and analyzing web content using AI-powered RAG
"""
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
from datetime import datetime
from urllib.parse import urlparse
from typing import Dict, List, Any, Optional

# Import refactored modules
from config.settings import DEFAULT_VERBOSITY, DEFAULT_MAX_PAGES, DEFAULT_DELAY
from src.core.crawler import WebCrawler
from src.core.rag_engine import WebRAGEngine
from src.utils.cache_manager import CacheManager

# Configure Streamlit page
st.set_page_config(
    page_title="Web Content Analyzer",
    page_icon="🕷️",
    layout="wide",
    initial_sidebar_state="expanded"
)


class WebContentAnalyzer:
    """Main application class for the Web Content Analyzer"""
    
    def __init__(self):
        self.cache_manager = CacheManager()
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize session state variables"""
        defaults = {
            'crawler': None,
            'rag_engine': None,
            'crawled_content': [],
            'crawl_stats': {},
            'analysis_results': {},
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
        try:
            from config.settings import OPENAI_API_KEY
            return bool(OPENAI_API_KEY)
        except:
            return False

    def render_sidebar(self):
        """Render the collapsible sidebar interface"""
        with st.sidebar:
            st.markdown("### Navigation")
            
            # Navigation buttons for drawers
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
        """Render the web crawler interface"""
        st.markdown("#### 🕷️ Web Crawler")
        
        url_input = st.text_input(
            "Website URL",
            placeholder="https://example.com",
            help="Enter the URL of the website you want to analyze"
        )
        
        with st.expander("⚙️ Crawling Settings"):
            max_pages = st.slider("Maximum pages to crawl", 1, 100, DEFAULT_MAX_PAGES)
            delay = st.slider("Delay between requests (seconds)", 0.5, 5.0, DEFAULT_DELAY, 0.5)
            st.info("Higher delays are more respectful to websites but take longer")
        
        if st.button("🔍 Start Crawling", disabled=st.session_state.crawl_in_progress):
            if url_input:
                self.start_crawling(url_input, max_pages, delay)
            else:
                st.warning("Please enter a valid URL")

    def render_cache_drawer(self):
        """Render the cache management interface"""
        st.markdown("#### 💾 Cache Management")
        
        cache_files = self.cache_manager.get_cache_files()
        
        if cache_files:
            selected_cache = st.selectbox(
                "Load from cache:",
                options=[None] + [f['filename'] for f in cache_files],
                format_func=lambda x: self.cache_manager.format_cache_name(x) if x else "Select cache file..."
            )
            
            if st.button("📂 Load Cache") and selected_cache:
                self.load_cached_content(selected_cache)
        else:
            st.info("No cache files found")

    def render_overview_drawer(self):
        """Render the content overview interface"""
        if st.session_state.crawled_content:
            st.markdown("#### 📊 Content Overview")
            
            stats = st.session_state.crawl_stats
            if stats:
                st.metric("Total Pages", stats.get('total_pages', 0))
                st.metric("Total Words", f"{stats.get('total_words', 0):,}")
                st.metric("Avg Words/Page", f"{stats.get('average_words_per_page', 0):.0f}")
                
                if st.session_state.rag_engine:
                    summary = st.session_state.rag_engine.get_content_summary()
                    st.markdown("**Content Chunks:** " + str(summary.get('total_chunks', 0)))
        else:
            st.info("No content loaded yet")

    def start_crawling(self, url: str, max_pages: int, delay: float):
        """Start the crawling process"""
        st.session_state.crawl_in_progress = True
        domain = urlparse(url).netloc
        st.session_state.current_domain = domain
        
        with st.sidebar:
            st.markdown("---")
            st.markdown(f"### 🕷️ Crawling Progress")
            
            pages_visited_placeholder = st.empty()
            pages_extracted_placeholder = st.empty()
            progress_bar_placeholder = st.empty()
            current_page_placeholder = st.empty()
            eta_placeholder = st.empty()
            
            start_time = datetime.now()
        
        crawler = WebCrawler(max_pages=max_pages, delay=delay)
        
        def progress_callback(visited, extracted, current_url=None, page_title=None):
            pages_visited_placeholder.metric("📄 Pages Visited", visited)
            pages_extracted_placeholder.metric("✅ Content Extracted", extracted)
            
            progress = min(visited / max_pages, 1.0)
            progress_bar_placeholder.progress(progress, text=f"{progress:.0%} Complete")
            
            if current_url and page_title:
                title_display = page_title[:40] + "..." if len(page_title) > 40 else page_title
                current_page_placeholder.info(f"🔍 **Current:** {title_display}")
            
            elapsed = (datetime.now() - start_time).total_seconds()
            if visited > 0 and elapsed > 0:
                rate = visited / (elapsed / 60)
                remaining = max_pages - visited
                eta_minutes = remaining / rate if rate > 0 else 0
                
                if eta_minutes < 1:
                    eta_display = "< 1 min"
                elif eta_minutes < 60:
                    eta_display = f"{eta_minutes:.0f} min"
                else:
                    hours = eta_minutes / 60
                    eta_display = f"{hours:.1f} hrs"
                
                eta_placeholder.info(f"⏱️ **ETA:** {eta_display}")
        
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
            
            # AI Analysis
            with st.spinner("Processing content for AI analysis..."):
                rag_engine = WebRAGEngine(collection_name=f"web_{domain.replace('.', '_')}")
                rag_engine.process_web_content(content, domain, use_cached_embeddings=False)
                st.session_state.rag_engine = rag_engine
            
            with st.spinner("Saving to cache..."):
                cache_file = self.cache_manager.save_cache(content, domain)
                st.info(f"Cache saved: {cache_file}")
            
            st.success(f"Successfully analyzed {len(content)} pages!")
        else:
            st.error("Failed to crawl website. Please check the URL and try again.")
        
        st.session_state.crawl_in_progress = False

    def load_cached_content(self, filename: str):
        """Load content from cache file"""
        with st.spinner("Loading cached content..."):
            try:
                cache_data = self.cache_manager.load_cache(filename)
                
                if cache_data and cache_data.get('content'):
                    content = cache_data['content']
                    st.session_state.crawled_content = content
                    domain = cache_data.get('domain', urlparse(content[0]['url']).netloc)
                    st.session_state.current_domain = domain
                    
                    if content:
                        total_words = sum(page.get('word_count', 0) for page in content)
                        avg_words = total_words / len(content) if content else 0
                        
                        stats = {
                            'total_pages': len(content),
                            'total_words': total_words,
                            'average_words_per_page': avg_words,
                            'domain': domain,
                            'source': 'cache'
                        }
                        
                        st.session_state.crawl_stats = stats
                    
                    rag_engine = WebRAGEngine(collection_name=f"web_{domain.replace('.', '_')}")
                    rag_engine.process_web_content(content, domain, use_cached_embeddings=True)
                    st.session_state.rag_engine = rag_engine
                    
                    st.success(f"Loaded {len(content)} pages from cache!")
                else:
                    st.error("Cache file is empty or corrupted")
            except Exception as e:
                st.error(f"Error loading cache: {str(e)}")

    def render_main_content(self):
        """Render the main content area"""
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
        """Render the welcome screen when no content is loaded"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### 🚀 Key Features
            - **Intelligent Web Crawling**: Respects robots.txt and rate limits
            - **AI-Powered Analysis**: GPT-4 powered question answering
            - **Semantic Search**: Find relevant content using natural language
            - **Visual Analytics**: Interactive charts and network graphs
            - **Embedding Cache**: Reuse AI processing for faster analysis
            """)
        
        with col2:
            st.markdown("""
            #### 💡 Example Questions
            - "What products does this company offer?"
            - "What are the main topics covered?"
            - "How can I contact them?"
            - "What are their pricing options?"
            """)
        
        st.markdown("---")
        st.info("👆 Enter a website URL in the sidebar to get started!")

    def render_analysis_interface(self):
        """Render the main analysis interface"""
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
                for question in suggested_questions[:6]:
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

    def run(self):
        """Run the main application"""
        if not self.check_openai_key():
            st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
            st.stop()
        
        self.render_sidebar()
        self.render_main_content()


def main():
    """Main application entry point"""
    app = WebContentAnalyzer()
    app.run()


if __name__ == "__main__":
    main()