import streamlit as st
import time
import os
from datetime import datetime
from web_crawler import WebCrawler
from web_rag_engine import WebRAGEngine
from urllib.parse import urlparse
import json
import plotly.express as px
import plotly.graph_objects as go
from streamlit_agraph import agraph, Node, Edge, Config
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Web Content Analyzer",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        'crawler': None,
        'rag_engine': None,
        'crawled_content': [],
        'crawl_stats': {},
        'analysis_results': {},
        'cache_files': [],
        'current_domain': None,
        'crawl_in_progress': False
    }
    
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

def check_openai_key():
    """Check if OpenAI API key is available"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("🔑 OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        st.stop()
    return api_key

def get_cache_files():
    """Get list of available cache files"""
    cache_files = []
    for file in os.listdir('.'):
        if file.startswith('cache_') and file.endswith('.json'):
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    cache_files.append({
                        'filename': file,
                        'domain': file.split('_')[1] if len(file.split('_')) > 1 else 'unknown',
                        'crawled_at': data.get('crawled_at', 'Unknown'),
                        'total_pages': data.get('total_pages', 0)
                    })
            except:
                continue
    return cache_files

def crawl_website(url, max_pages, delay):
    """Crawl website and return content"""
    crawler = WebCrawler(max_pages=max_pages, delay=delay)
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def progress_callback(visited, extracted):
        progress = min(visited / max_pages, 1.0)
        progress_bar.progress(progress)
        status_text.text(f"Crawling... Visited: {visited}, Extracted: {extracted} pages")
    
    try:
        content = crawler.crawl_website(url, progress_callback)
        progress_bar.progress(1.0)
        status_text.text(f"✅ Crawling completed! Extracted {len(content)} pages")
        
        # Save cache
        cache_file = crawler.save_cache()
        
        return content, crawler.get_crawl_stats(), cache_file
    
    except Exception as e:
        st.error(f"Error during crawling: {str(e)}")
        return [], {}, None

def create_content_visualization(content):
    """Create visualizations for crawled content"""
    if not content:
        return None, None
    
    # Word count distribution
    word_counts = [page.get('word_count', 0) for page in content]
    fig_words = px.histogram(
        x=word_counts,
        nbins=20,
        title="Distribution of Page Word Counts",
        labels={'x': 'Word Count', 'y': 'Number of Pages'}
    )
    
    # Pages by URL path depth
    path_depths = []
    for page in content:
        url = page.get('url', '')
        if url:
            path = urlparse(url).path
            depth = len([p for p in path.split('/') if p])
            path_depths.append(depth)
    
    if path_depths:
        fig_depth = px.bar(
            x=list(range(max(path_depths) + 1)),
            y=[path_depths.count(i) for i in range(max(path_depths) + 1)],
            title="Pages by URL Depth",
            labels={'x': 'URL Depth', 'y': 'Number of Pages'}
        )
    else:
        fig_depth = None
    
    return fig_words, fig_depth

def create_network_graph(content):
    """Create a network graph of content relationships"""
    if len(content) < 2:
        return None
    
    nodes = []
    edges = []
    
    # Create nodes for each page
    for i, page in enumerate(content[:20]):  # Limit to 20 for performance
        title = page.get('title', f"Page {i+1}")
        word_count = page.get('word_count', 0)
        
        # Node size based on word count
        size = max(10, min(30, word_count / 100))
        
        nodes.append(Node(
            id=str(i),
            label=title[:30] + "..." if len(title) > 30 else title,
            size=size,
            color="#FF6B6B" if word_count > 1000 else "#4ECDC4"
        ))
    
    # Create edges based on content similarity (simplified)
    for i in range(min(len(content), 20)):
        for j in range(i+1, min(len(content), 20)):
            # Simple similarity based on common words in titles
            title1 = content[i].get('title', '').lower().split()
            title2 = content[j].get('title', '').lower().split()
            common_words = set(title1) & set(title2)
            
            if len(common_words) > 1:
                edges.append(Edge(
                    source=str(i),
                    target=str(j),
                    width=len(common_words)
                ))
    
    config = Config(
        width=700,
        height=400,
        directed=False,
        physics=True,
        hierarchical=False
    )
    
    return agraph(nodes=nodes, edges=edges, config=config)

def main():
    # Initialize session state
    initialize_session_state()
    
    # Check for OpenAI API key
    check_openai_key()
    
    # Header
    st.title("🌐 Web Content Analyzer")
    st.markdown("Transform any website into intelligent, searchable knowledge using AI-powered content analysis.")
    
    # Sidebar for crawling controls
    with st.sidebar:
        st.header("🕷️ Web Crawler")
        
        # URL input
        url_input = st.text_input(
            "Website URL",
            placeholder="https://example.com",
            help="Enter the URL of the website you want to analyze"
        )
        
        # Crawling parameters
        with st.expander("⚙️ Crawling Settings"):
            max_pages = st.slider("Maximum pages to crawl", 1, 100, 25)
            delay = st.slider("Delay between requests (seconds)", 0.5, 5.0, 1.0, 0.5)
            st.info("Higher delays are more respectful to websites but take longer")
        
        # Crawl button
        if st.button("🚀 Start Crawling", disabled=st.session_state.crawl_in_progress):
            if url_input:
                st.session_state.crawl_in_progress = True
                domain = urlparse(url_input).netloc
                st.session_state.current_domain = domain
                
                with st.spinner(f"Crawling {domain}..."):
                    content, stats, cache_file = crawl_website(url_input, max_pages, delay)
                    
                    if content:
                        st.session_state.crawled_content = content
                        st.session_state.crawl_stats = stats
                        
                        # Initialize RAG engine
                        with st.spinner("Processing content for AI analysis..."):
                            rag_engine = WebRAGEngine(collection_name=f"web_{domain.replace('.', '_')}")
                            rag_engine.process_web_content(content, domain)
                            st.session_state.rag_engine = rag_engine
                        
                        st.success(f"✅ Successfully analyzed {len(content)} pages!")
                    else:
                        st.error("Failed to crawl website. Please check the URL and try again.")
                
                st.session_state.crawl_in_progress = False
            else:
                st.warning("Please enter a valid URL")
        
        # Cache management
        st.markdown("### 💾 Cache Management")
        cache_files = get_cache_files()
        st.session_state.cache_files = cache_files
        
        if cache_files:
            selected_cache = st.selectbox(
                "Load from cache:",
                options=[None] + [f['filename'] for f in cache_files],
                format_func=lambda x: "Select cache file..." if x is None else f"{x} ({next(f['total_pages'] for f in cache_files if f['filename'] == x)} pages)"
            )
            
            if st.button("📂 Load Cache") and selected_cache:
                with st.spinner("Loading cached content..."):
                    try:
                        with open(selected_cache, 'r') as f:
                            cache_data = json.load(f)
                        
                        content = cache_data.get('content', [])
                        if content:
                            st.session_state.crawled_content = content
                            domain = cache_data.get('domain', urlparse(content[0]['url']).netloc)
                            st.session_state.current_domain = domain
                            
                            # Initialize RAG engine
                            rag_engine = WebRAGEngine(collection_name=f"web_{domain.replace('.', '_')}")
                            rag_engine.process_web_content(content, domain)
                            st.session_state.rag_engine = rag_engine
                            
                            st.success(f"✅ Loaded {len(content)} pages from cache!")
                        else:
                            st.error("Cache file is empty or corrupted")
                    except Exception as e:
                        st.error(f"Error loading cache: {str(e)}")
        else:
            st.info("No cache files found")
    
    # Main content area
    if not st.session_state.crawled_content:
        # Welcome screen with features
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 Key Features
            
            **🕷️ Intelligent Web Crawling**
            - Respects robots.txt and rate limits
            - Extracts clean, readable content
            - Handles dynamic content and navigation
            
            **🧠 AI-Powered Analysis**
            - Semantic search across all content
            - Context-aware question answering
            - Content summarization and insights
            
            **📊 Visual Analytics**
            - Content distribution charts
            - Site structure visualization
            - Word frequency analysis
            """)
        
        with col2:
            st.markdown("""
            ### 🚀 How It Works
            
            1. **Enter a URL** - Any publicly accessible website
            2. **Configure crawling** - Set limits and delays
            3. **Start analysis** - AI processes all content
            4. **Ask questions** - Get intelligent answers
            5. **Explore insights** - Visual analytics and summaries
            
            ### 💡 Example Questions
            - "What products does this company offer?"
            - "What are the main topics covered?"
            - "How can I contact them?"
            - "What are their pricing options?"
            """)
        
        st.markdown("---")
        st.info("👆 Enter a website URL in the sidebar to get started!")
    
    else:
        # Content analysis interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header(f"💬 Ask Questions about {st.session_state.current_domain}")
            
            # Suggested questions
            if st.session_state.rag_engine:
                suggested_questions = st.session_state.rag_engine.suggest_questions()
                
                with st.expander("💡 Suggested Questions"):
                    for question in suggested_questions[:6]:
                        if st.button(f"❓ {question}", key=f"suggested_{hash(question)}"):
                            st.session_state.current_question = question
            
            # Question input
            question = st.text_area(
                "Your question:",
                placeholder="Ask anything about the website content...",
                height=80
            )
            
            # Use suggested question if set
            if hasattr(st.session_state, 'current_question'):
                question = st.session_state.current_question
                delattr(st.session_state, 'current_question')
            
            if question and st.session_state.rag_engine:
                with st.spinner("Analyzing content..."):
                    result = st.session_state.rag_engine.analyze_content(question)
                    
                    # Display answer
                    st.markdown("### 🤖 Analysis Result")
                    st.markdown(result['answer'])
                    
                    # Confidence indicator
                    confidence = result.get('confidence', 0)
                    if confidence > 0.8:
                        confidence_color = "green"
                        confidence_text = "High"
                    elif confidence > 0.6:
                        confidence_color = "orange"
                        confidence_text = "Medium"
                    else:
                        confidence_color = "red"
                        confidence_text = "Low"
                    
                    st.markdown(f"**Confidence:** :{confidence_color}[{confidence_text} ({confidence:.1%})]")
                    
                    # Sources
                    if result.get('sources'):
                        with st.expander(f"📚 Sources ({len(result['sources'])} pages)"):
                            for i, source in enumerate(result['sources'][:5], 1):
                                similarity = source.get('similarity_score', 0)
                                st.markdown(f"""
                                **{i}. {source.get('title', 'Untitled')}**  
                                🔗 [{source.get('url', 'No URL')}]({source.get('url', '#')})  
                                📊 Relevance: {similarity:.1%}
                                """)
        
        with col2:
            st.header("📊 Content Overview")
            
            # Quick stats
            stats = st.session_state.crawl_stats
            if stats:
                st.metric("Total Pages", stats.get('total_pages', 0))
                st.metric("Total Words", f"{stats.get('total_words', 0):,}")
                st.metric("Avg Words/Page", f"{stats.get('average_words_per_page', 0):.0f}")
            
            # Content summary
            if st.session_state.rag_engine:
                summary = st.session_state.rag_engine.get_content_summary()
                
                st.markdown("**Content Chunks:** " + str(summary.get('total_chunks', 0)))
                
                if summary.get('sample_titles'):
                    with st.expander("📄 Sample Pages"):
                        for title in summary['sample_titles']:
                            st.markdown(f"• {title}")
        
        # Analytics tabs
        st.markdown("---")
        tab1, tab2, tab3 = st.tabs(["📈 Analytics", "🔍 Search", "🗂️ Content"])
        
        with tab1:
            st.header("Content Analytics")
            
            # Visualizations
            fig_words, fig_depth = create_content_visualization(st.session_state.crawled_content)
            
            if fig_words:
                st.plotly_chart(fig_words, use_container_width=True)
            
            if fig_depth:
                st.plotly_chart(fig_depth, use_container_width=True)
            
            # Network graph
            st.subheader("Content Relationship Network")
            try:
                network_result = create_network_graph(st.session_state.crawled_content)
                if network_result:
                    st.info("Node size represents word count. Connections show content similarity.")
            except Exception as e:
                st.info("Network visualization not available for this content")
        
        with tab2:
            st.header("Semantic Search")
            
            search_query = st.text_input("Search content:", placeholder="Enter search terms...")
            
            if search_query and st.session_state.rag_engine:
                results = st.session_state.rag_engine.search_content(search_query, k=10)
                
                st.write(f"Found {len(results)} relevant results:")
                
                for i, result in enumerate(results[:5], 1):
                    with st.expander(f"{i}. {result['title']} (Relevance: {result['similarity_score']:.1%})"):
                        st.markdown(f"**URL:** {result['url']}")
                        st.markdown(f"**Content Preview:**")
                        st.text(result['content'][:300] + "..." if len(result['content']) > 300 else result['content'])
        
        with tab3:
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

if __name__ == "__main__":
    main()