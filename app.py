"""
Web Content Analyzer - Refactored Main Application
A comprehensive Streamlit application for crawling and analyzing web content using AI-powered RAG
"""
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
import pandas as pd
from datetime import datetime
from urllib.parse import urlparse
from typing import Dict, List, Any, Optional

# Import refactored modules
from config.settings import DEFAULT_VERBOSITY, DEFAULT_MAX_PAGES, DEFAULT_DELAY
from src.core.crawler import WebCrawler
from src.core.rag_engine import WebRAGEngine
from src.utils.cache_manager import CacheManager
from src.ui.visualizations import (
    create_content_visualization, 
    create_improved_network_graph,
    display_source_analysis,
    display_reliability_guide,
    display_confidence_explanation
)

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
            'active_drawer': None  # Options: 'crawler', 'cache', 'overview'
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
            
            # Web Crawler Drawer
            if st.session_state.active_drawer == 'crawler':
                self.render_crawler_drawer()
            
            # Cache Management Drawer
            elif st.session_state.active_drawer == 'cache':
                self.render_cache_drawer()
            
            # Content Overview Drawer
            elif st.session_state.active_drawer == 'overview':
                self.render_overview_drawer()

    def render_crawler_drawer(self):
        """Render the web crawler interface"""
        st.markdown("#### 🕷️ Web Crawler")
        
        # URL input
        url_input = st.text_input(
            "Website URL",
            placeholder="https://example.com",
            help="Enter the URL of the website you want to analyze"
        )
        
        # Crawling parameters
        with st.expander("⚙️ Crawling Settings"):
            max_pages = st.slider("Maximum pages to crawl", 1, 100, DEFAULT_MAX_PAGES)
            delay = st.slider("Delay between requests (seconds)", 0.5, 5.0, DEFAULT_DELAY, 0.5)
            st.info("Higher delays are more respectful to websites but take longer")
        
        # Crawl button
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
                
                # Coverage percentage if available
                if stats.get('coverage_percentage') is not None:
                    coverage = stats['coverage_percentage']
                    
                    # Color code coverage
                    if coverage >= 80:
                        coverage_color = "green"
                    elif coverage >= 50:
                        coverage_color = "orange"
                    else:
                        coverage_color = "red"
                    
                    st.markdown(f"**Website Coverage:** :{coverage_color}[{coverage:.1f}%]")
                    st.caption(f"Crawled {stats['total_pages']} of {stats.get('estimated_total_pages', 0)} total pages")
                    
                    # Show source of size estimation with accurate details
                    size_source = stats.get('size_source', 'unknown')
                    size_details = stats.get('size_details', '')
                    
                    if size_source == 'sitemap':
                        st.caption("📋 Size based on website sitemap analysis")
                    elif size_source == 'robots':
                        st.caption("🤖 Size estimated from robots.txt sitemap references")
                    elif size_source == 'dynamic':
                        st.caption("🔍 Size estimated through dynamic link discovery")
                    elif size_source == 'third_party':
                        service_name = size_details.split(':')[0] if ':' in size_details else 'third-party service'
                        st.caption(f"🌐 Size estimated using {service_name}")
                    elif size_source == 'fallback':
                        st.caption("⚠️ Size estimation unavailable - showing crawled pages only")
                    else:
                        st.caption("❓ Size estimation method unknown")
                
                # Content summary
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
        
        # Create progress tracking containers
        progress_container = st.container()
        with progress_container:
            st.markdown(f"### 🕷️ Crawling {domain}")
            
            # Progress metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                pages_visited = st.metric("Pages Visited", 0)
            with col2:
                pages_extracted = st.metric("Content Extracted", 0)
            with col3:
                current_page_metric = st.metric("Current Page", "Starting...")
            with col4:
                eta_metric = st.metric("ETA", "Calculating...")
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Performance chart placeholder
            chart_placeholder = st.empty()
            
            # Initialize performance tracking
            performance_data = {
                'timestamps': [],
                'pages_per_minute': [],
                'cumulative_pages': []
            }
            start_time = datetime.now()
        
        crawler = WebCrawler(max_pages=max_pages, delay=delay)
        
        def progress_callback(visited, extracted, current_url=None, page_title=None):
            # Update metrics
            col1.metric("Pages Visited", visited)
            col2.metric("Content Extracted", extracted)
            
            if current_url:
                current_page_display = current_url.split('/')[-1][:30] + "..." if len(current_url.split('/')[-1]) > 30 else current_url.split('/')[-1]
                col3.metric("Current Page", current_page_display)
            
            # Calculate ETA
            elapsed = (datetime.now() - start_time).total_seconds()
            if visited > 0 and elapsed > 0:
                rate = visited / (elapsed / 60)  # pages per minute
                remaining = max_pages - visited
                eta_minutes = remaining / rate if rate > 0 else 0
                
                if eta_minutes < 1:
                    eta_display = "< 1 min"
                elif eta_minutes < 60:
                    eta_display = f"{eta_minutes:.0f} min"
                else:
                    hours = eta_minutes / 60
                    eta_display = f"{hours:.1f} hrs"
                
                col4.metric("ETA", eta_display)
                
                # Update performance data
                performance_data['timestamps'].append(elapsed / 60)
                performance_data['pages_per_minute'].append(rate)
                performance_data['cumulative_pages'].append(visited)
                
                # Update performance chart every 5 pages
                if visited % 5 == 0 and len(performance_data['timestamps']) > 1:
                    try:
                        import plotly.graph_objects as go
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=performance_data['timestamps'],
                            y=performance_data['pages_per_minute'],
                            mode='lines+markers',
                            name='Pages/Min',
                            line=dict(color='#1f77b4')
                        ))
                        fig.update_layout(
                            title="Crawling Performance",
                            xaxis_title="Time (minutes)",
                            yaxis_title="Pages per Minute",
                            height=200,
                            showlegend=False
                        )
                        chart_placeholder.plotly_chart(fig, use_container_width=True)
                    except:
                        pass
            
            # Update progress bar
            progress = min(visited / max_pages, 1.0)
            progress_bar.progress(progress)
            
            # Update status
            if page_title:
                title_display = page_title[:50] + "..." if len(page_title) > 50 else page_title
                status_text.text(f"Processing: {title_display}")
            else:
                status_text.text(f"Crawling page {visited}/{max_pages}")
        
        # Crawl the website
        content = crawler.crawl_website(url, progress_callback=progress_callback)
        
        # Clear progress display and show results
        progress_container.empty()
        
        if content:
            st.session_state.crawled_content = content
            
            # Calculate statistics
            stats = crawler.get_crawl_stats()
            
            # Get website size estimation for coverage
            estimation_result = crawler.estimate_total_pages(url)
            stats['estimation_result'] = estimation_result
            
            if estimation_result['total_pages'] is not None:
                estimated_total = estimation_result['total_pages']
                coverage_percentage = (len(content) / estimated_total * 100) if estimated_total > 0 else 100
                stats.update({
                    'estimated_total_pages': estimated_total,
                    'coverage_percentage': coverage_percentage,
                    'size_source': estimation_result['source'],
                    'size_details': estimation_result['details']
                })
            
            st.session_state.crawl_stats = stats
            
            # Initialize RAG engine and process content
            with st.spinner("Processing content for AI analysis..."):
                rag_engine = WebRAGEngine(collection_name=f"web_{domain.replace('.', '_')}")
                rag_engine.process_web_content(content, domain, use_cached_embeddings=False)
                st.session_state.rag_engine = rag_engine
            
            # Save cache with embeddings
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
                    
                    # Calculate coverage for cached content
                    crawler = WebCrawler()
                    if content:
                        base_url = content[0]['url']
                        estimation_result = crawler.estimate_total_pages(base_url)
                        
                        # Calculate content statistics
                        total_words = sum(page.get('word_count', 0) for page in content)
                        avg_words = total_words / len(content) if content else 0
                        
                        # Store stats for display
                        stats = {
                            'total_pages': len(content),
                            'total_words': total_words,
                            'average_words_per_page': avg_words,
                            'domain': domain,
                            'source': 'cache',
                            'estimation_result': estimation_result
                        }
                        
                        # Only calculate coverage if we have reliable website size data
                        if estimation_result['total_pages'] is not None:
                            estimated_total = estimation_result['total_pages']
                            coverage_percentage = (len(content) / estimated_total * 100) if estimated_total > 0 else 100
                            stats.update({
                                'estimated_total_pages': estimated_total,
                                'coverage_percentage': coverage_percentage,
                                'size_source': estimation_result['source'],
                                'size_details': estimation_result['details']
                            })
                        
                        st.session_state.crawl_stats = stats
                    
                    # Initialize RAG engine with cached embeddings
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
        # Display protocol disclaimer
        st.info("""
        **Crawling Protocol Notice:** This tool respects website robots.txt files and implements rate limiting. 
        Only publicly accessible pages are crawled (no authentication required). We crawl both static and dynamic content 
        with a 1-5 second delay between requests and 10-second page load timeout to be respectful to server resources.
        """)
        
        if not st.session_state.crawled_content:
            self.render_welcome_screen()
        else:
            self.render_analysis_interface()

    def render_welcome_screen(self):
        """Render the welcome screen when no content is loaded"""
        st.title("🕷️ Web Content Analyzer")
        st.markdown("### Transform any website into intelligent, searchable knowledge")
        
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
        st.header(f"💬 Ask Questions about {st.session_state.current_domain}")
        
        # Suggested questions
        if st.session_state.rag_engine:
            suggested_questions = st.session_state.rag_engine.suggest_questions()
            
            with st.expander("💡 Suggested Questions"):
                for question in suggested_questions[:6]:
                    if st.button(f"❓ {question}", key=f"suggested_{hash(question)}"):
                        st.session_state.current_question = question
        
        # Question input and verbosity selector
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
            
            # Update session state if changed
            if verbosity != st.session_state.answer_verbosity:
                st.session_state.answer_verbosity = verbosity
        
        # Use suggested question if set
        if hasattr(st.session_state, 'current_question'):
            question = st.session_state.current_question
            delattr(st.session_state, 'current_question')
        
        # Process question if provided
        if question and st.session_state.rag_engine:
            self.process_question(question, verbosity)
        
        # Analytics tabs
        self.render_analytics_tabs()

    def process_question(self, question: str, verbosity: str):
        """Process a user question and display results"""
        with st.spinner("Analyzing content..."):
            result = st.session_state.rag_engine.analyze_content(question, verbosity=verbosity)
            
            # Display answer with verbosity indicator
            verbosity_indicators = {
                'concise': '🎯 Concise',
                'balanced': '⚖️ Balanced',
                'comprehensive': '📖 Detailed'
            }
            
            st.markdown(f"### Answer ({verbosity_indicators[verbosity]})")
            st.markdown(result['answer'])
            
            # Display confidence and explanation
            confidence = result.get('confidence', 0)
            display_confidence_explanation(confidence)
            
            # Sources
            if result.get('sources'):
                with st.expander(f"📚 Sources ({len(result['sources'])} unique pages)"):
                    display_source_analysis(result['sources'])
                
                # Reliability guide after sources
                display_reliability_guide()

    def render_analytics_tabs(self):
        """Render the analytics section with tabs"""
        st.markdown("---")
        tab1, tab2, tab3 = st.tabs(["📈 Analytics", "🔍 Search", "🗂️ Content"])
        
        with tab1:
            self.render_analytics_tab()
        
        with tab2:
            self.render_search_tab()
        
        with tab3:
            self.render_content_tab()

    def render_analytics_tab(self):
        """Render the analytics tab"""
        st.header("Content Analytics")
        
        # Visualizations side by side
        fig_words, fig_depth = create_content_visualization(st.session_state.crawled_content)
        
        if fig_words and fig_depth:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_words, use_container_width=True)
            with col2:
                st.plotly_chart(fig_depth, use_container_width=True)
        elif fig_words:
            st.plotly_chart(fig_words, use_container_width=True)
        elif fig_depth:
            st.plotly_chart(fig_depth, use_container_width=True)
        
        # Network graph
        st.subheader("Content Relationship Network")
        st.caption("Interactive visualization showing how pages relate to each other based on content similarity")
        
        try:
            network_result = create_improved_network_graph(st.session_state.crawled_content)
            if network_result:
                st.info("💡 Larger nodes = more content. Thicker connections = stronger similarity. Click and drag to explore!")
            else:
                st.info("Network requires at least 2 pages to display relationships")
        except Exception as e:
            st.info("Network visualization temporarily unavailable")

    def render_search_tab(self):
        """Render the semantic search tab"""
        st.header("Semantic Search")
        
        # Explanation of the difference
        with st.expander("What's the difference between Questions and Search?"):
            st.markdown("""
            **User Questions (AI Analysis):**
            - Uses AI to understand and analyze your question
            - Combines information from multiple sources
            - Provides comprehensive answers with context
            - Best for: "What are the main features?" or "How much does it cost?"
            
            **Semantic Search (Direct Lookup):**
            - Finds pages that contain similar content to your search terms
            - Shows you the actual pages without interpretation
            - Returns ranked list of matching content
            - Best for: Finding specific pages, exploring content, or getting raw information
            """)
        
        search_query = st.text_input("Search content:", placeholder="Enter search terms...")
        
        if search_query and st.session_state.rag_engine:
            results = st.session_state.rag_engine.search_content(search_query, k=10)
            
            st.write(f"Found {len(results)} relevant results:")
            
            for i, result in enumerate(results[:5], 1):
                similarity = result.get('similarity_score', 0)
                
                # Color code search relevance
                if similarity >= 0.7:
                    relevance_icon = "🟢"
                elif similarity >= 0.5:
                    relevance_icon = "🟡"
                else:
                    relevance_icon = "🟠"
                
                with st.expander(f"{i}. {result['title']} {relevance_icon} {similarity:.1%}"):
                    st.markdown(f"**URL:** {result['url']}")
                    st.markdown(f"**Content Preview:**")
                    st.text(result['content'][:300] + "..." if len(result['content']) > 300 else result['content'])

    def render_content_tab(self):
        """Render the raw content tab"""
        st.header("Raw Content")
        
        if st.session_state.crawled_content:
            # Content table
            content_df = pd.DataFrame([
                {
                    'Title': page.get('title', 'Untitled')[:50] + "..." if len(page.get('title', '')) > 50 else page.get('title', 'Untitled'),
                    'URL': page.get('url', ''),
                    'Words': page.get('word_count', 0),
                    'Headings': len(page.get('headings', [])),
                    'Has Description': bool(page.get('description', ''))
                }
                for page in st.session_state.crawled_content
            ])
            
            st.dataframe(content_df, use_container_width=True)
            
            # Individual page details
            selected_page = st.selectbox(
                "View page details:",
                options=range(len(st.session_state.crawled_content)),
                format_func=lambda x: f"{x+1}. {st.session_state.crawled_content[x].get('title', 'Untitled')}"
            )
            
            if selected_page is not None:
                page = st.session_state.crawled_content[selected_page]
                
                st.markdown(f"**Title:** {page.get('title', 'N/A')}")
                st.markdown(f"**URL:** {page.get('url', 'N/A')}")
                st.markdown(f"**Description:** {page.get('description', 'N/A')}")
                st.markdown(f"**Word Count:** {page.get('word_count', 0)}")
                
                if page.get('headings'):
                    st.markdown("**Headings:**")
                    for heading in page['headings'][:10]:
                        st.markdown(f"• {heading}")
                
                with st.expander("View full content"):
                    st.text(page.get('content', 'No content available'))
        else:
            st.info("No content available. Crawl a website first.")

    def run(self):
        """Run the main application"""
        # Check for OpenAI API key
        if not self.check_openai_key():
            st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
            st.stop()
        
        # Render the application
        self.render_sidebar()
        self.render_main_content()


def main():
    """Main application entry point"""
    app = WebContentAnalyzer()
    app.run()


if __name__ == "__main__":
    main()