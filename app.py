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
    
    # Estimate total pages before crawling
    with st.spinner("Analyzing website structure..."):
        estimated_total = crawler.estimate_total_pages(url)
        crawler.estimated_total_pages = estimated_total
    
    # Show estimation to user
    if estimated_total > max_pages:
        st.info(f"📊 Website has approximately {estimated_total} pages. You'll crawl {max_pages} pages ({(max_pages/estimated_total)*100:.1f}% coverage)")
    else:
        st.info(f"📊 Website has approximately {estimated_total} pages. You'll crawl up to {max_pages} pages (potentially 100% coverage)")
    
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

def create_improved_network_graph(content):
    """Create an improved network graph of content relationships"""
    if len(content) < 2:
        return None
    
    nodes = []
    edges = []
    
    # Limit to 15 nodes to reduce clutter
    content_subset = content[:15]
    
    # Create nodes for each page with better spacing
    for i, page in enumerate(content_subset):
        title = page.get('title', f"Page {i+1}")
        word_count = page.get('word_count', 0)
        url_path = page.get('url', '').split('/')[-1] or 'home'
        
        # Better node sizing - more reasonable range
        size = max(15, min(35, word_count / 50))
        
        # Color coding based on content type
        if 'home' in title.lower() or url_path == '' or 'index' in url_path:
            color = "#FF6B6B"  # Red for home/main pages
        elif word_count > 1000:
            color = "#4ECDC4"  # Teal for content-heavy pages
        elif any(word in title.lower() for word in ['product', 'service', 'plan']):
            color = "#45B7D1"  # Blue for product pages
        else:
            color = "#96CEB4"  # Green for other pages
        
        # Shorter, cleaner labels
        clean_title = title.replace('|', '-').replace('  ', ' ')
        label = clean_title[:25] + "..." if len(clean_title) > 25 else clean_title
        
        nodes.append(Node(
            id=str(i),
            label=label,
            size=size,
            color=color,
            title=f"{title}\nWords: {word_count}\nURL: {page.get('url', 'N/A')}"  # Tooltip
        ))
    
    # Create edges with better similarity calculation
    for i in range(len(content_subset)):
        for j in range(i+1, len(content_subset)):
            # Enhanced similarity based on multiple factors
            title1 = content_subset[i].get('title', '').lower()
            title2 = content_subset[j].get('title', '').lower()
            content1 = content_subset[i].get('content', '')[:500].lower()
            content2 = content_subset[j].get('content', '')[:500].lower()
            
            # Title similarity
            title1_words = set(title1.split())
            title2_words = set(title2.split())
            title_similarity = len(title1_words & title2_words)
            
            # Content similarity (simplified)
            content1_words = set(content1.split())
            content2_words = set(content2.split())
            content_similarity = len(content1_words & content2_words)
            
            # Combined similarity score
            total_similarity = title_similarity * 2 + min(content_similarity / 10, 5)
            
            # Only create edge if similarity is meaningful
            if total_similarity >= 3:
                # Cap edge width for better visualization
                edge_width = min(total_similarity / 2, 5)
                
                edges.append(Edge(
                    source=str(i),
                    target=str(j),
                    width=edge_width,
                    color="#E0E0E0"
                ))
    
    # Improved configuration for better layout
    config = Config(
        width=800,
        height=500,
        directed=False,
        physics=True,
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor="#F0F0F0",
        linkHighlightBehavior=True,
        maxZoom=3,
        minZoom=0.3
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
            def format_cache_name(filename):
                if filename is None:
                    return "Select cache file..."
                
                # Extract domain and timestamp from filename
                parts = filename.split('_')
                if len(parts) >= 3:
                    domain = parts[1].replace('www.', '')
                    timestamp = parts[2].replace('.json', '')
                    try:
                        # Parse timestamp and make it user-friendly
                        from datetime import datetime
                        dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
                        date_str = dt.strftime("%B %d, %Y at %I:%M %p")
                        pages = next(f['total_pages'] for f in cache_files if f['filename'] == filename)
                        return f"{domain} - {date_str} ({pages} pages)"
                    except:
                        pass
                
                # Fallback to original format
                pages = next(f['total_pages'] for f in cache_files if f['filename'] == filename)
                return f"{filename} ({pages} pages)"
            
            selected_cache = st.selectbox(
                "Load from cache:",
                options=[None] + [f['filename'] for f in cache_files],
                format_func=format_cache_name
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
        
        # Content Overview in sidebar
        if st.session_state.crawled_content:
            st.markdown("### 📊 Content Overview")
            
            # Quick stats
            stats = st.session_state.crawl_stats
            if stats:
                st.metric("Total Pages", stats.get('total_pages', 0))
                st.metric("Total Words", f"{stats.get('total_words', 0):,}")
                st.metric("Avg Words/Page", f"{stats.get('average_words_per_page', 0):.0f}")
                
                # Coverage percentage if available
                if stats.get('coverage_percentage') is not None:
                    coverage = stats['coverage_percentage']
                    estimated_total = stats.get('estimated_total_pages', 0)
                    
                    # Color code coverage
                    if coverage >= 80:
                        coverage_color = "green"
                    elif coverage >= 50:
                        coverage_color = "orange"
                    else:
                        coverage_color = "red"
                    
                    st.markdown(f"**Website Coverage:** :{coverage_color}[{coverage:.1f}%]")
                    st.caption(f"Crawled {stats['total_pages']} of ~{estimated_total} total pages")
            
            # Content summary
            if st.session_state.rag_engine:
                summary = st.session_state.rag_engine.get_content_summary()
                st.markdown("**Content Chunks:** " + str(summary.get('total_chunks', 0)))
    
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
        # Content analysis interface - full width
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
                
                # User-friendly confidence indicator
                confidence = result.get('confidence', 0)
                
                if confidence >= 0.8:
                    confidence_emoji = "🟢"
                    confidence_text = "Very Reliable"
                    confidence_desc = "High quality sources with strong relevance"
                elif confidence >= 0.6:
                    confidence_emoji = "🟡" 
                    confidence_text = "Mostly Reliable"
                    confidence_desc = "Good sources with decent relevance"
                elif confidence >= 0.4:
                    confidence_emoji = "🟠"
                    confidence_text = "Somewhat Reliable"
                    confidence_desc = "Some relevant information found"
                else:
                    confidence_emoji = "🔴"
                    confidence_text = "Limited Reliability"
                    confidence_desc = "Limited relevant information available"
                
                col_conf, col_info = st.columns([3, 1])
                with col_conf:
                    st.markdown(f"**Answer Quality:** {confidence_emoji} {confidence_text}")
                    st.caption(confidence_desc)
                with col_info:
                    with st.expander("ℹ️ About Quality"):
                        st.markdown("""
                        **Quality scoring is based on:**
                        - How well the content matches your question
                        - Number of relevant sources found
                        - Content overlap and consistency
                        
                        Higher scores mean the answer is more reliable and well-supported by the website content.
                        """)
                
                # Sources
                if result.get('sources'):
                    with st.expander(f"📚 Sources ({len(result['sources'])} unique pages)"):
                        st.caption("Each source shows how well that page content matches your question")
                        for i, source in enumerate(result['sources'][:5], 1):
                            similarity = source.get('similarity_score', 0)
                            
                            # Color code relevance
                            if similarity >= 0.7:
                                relevance_icon = "🟢"
                                relevance_text = "Highly Relevant"
                            elif similarity >= 0.5:
                                relevance_icon = "🟡"
                                relevance_text = "Moderately Relevant"
                            else:
                                relevance_icon = "🟠"
                                relevance_text = "Somewhat Relevant"
                            
                            st.markdown(f"""
                            **{i}. {source.get('title', 'Untitled')}**  
                            🔗 [{source.get('url', 'No URL')}]({source.get('url', '#')})  
                            {relevance_icon} {relevance_text} ({similarity:.1%})
                            """)
        
        # Analytics tabs
        st.markdown("---")
        tab1, tab2, tab3 = st.tabs(["📈 Analytics", "🔍 Search", "🗂️ Content"])
        
        with tab1:
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
            
            # Improved Network graph
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
        
        with tab2:
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
            else:
                st.info("No content available. Crawl a website first.")
        
        # Reliability Tips Section
        st.markdown("---")
        with st.expander("💡 How to Improve Answer Reliability"):
            st.markdown("""
            **To get the most reliable answers from the Web Content Analyzer:**
            
            **1. Crawl More Pages (15-100 recommended)**
            - More content = better context for AI analysis
            - Diverse pages provide comprehensive coverage
            - Current crawl limit can be adjusted in the sidebar
            - Check the "Website Coverage" percentage in the sidebar - aim for 50%+ when possible
            
            **2. Ask Specific Questions**
            - Instead of: "Tell me about this site"
            - Try: "What are the main product features?" or "What pricing options are available?"
            
            **3. Use Keywords in Search**
            - For semantic search, use specific terms from the website
            - Multiple related keywords work better than single words
            
            **4. Check Source Quality**
            - Look at the sources shown with each answer
            - Higher similarity scores (70%+) indicate better matches
            - Multiple relevant sources increase confidence
            
            **5. Compare Question vs Search Results**
            - Use questions for analysis and summaries
            - Use semantic search to verify specific details
            
            **6. Website Content Quality Matters**
            - Well-structured sites with clear content work better
            - Sites with lots of text content provide richer analysis
            - Technical documentation and product pages are ideal
            
            **Current Quality Indicators:**
            - 🟢 Very Reliable (80%+): High confidence, multiple good sources
            - 🟡 Mostly Reliable (60-80%): Good sources, decent coverage
            - 🟠 Somewhat Reliable (40-60%): Limited relevant information
            - 🔴 Limited Reliability (<40%): Insufficient matching content
            """)

if __name__ == "__main__":
    main()