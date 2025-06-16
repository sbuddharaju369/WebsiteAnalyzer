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
        'crawl_in_progress': False,
        'answer_verbosity': 'concise',  # Options: 'concise', 'balanced', 'comprehensive'
        'active_drawer': None  # Options: 'crawler', 'cache', 'overview'
    }
    
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

def update_progress_chart(chart_placeholder, performance_data):
    """Update the real-time progress chart"""
    if len(performance_data) < 2:
        return
    
    df = pd.DataFrame(performance_data)
    
    # Create dual-axis chart
    fig = go.Figure()
    
    # Pages crawled over time
    fig.add_trace(go.Scatter(
        x=df['time'],
        y=df['visited'],
        mode='lines+markers',
        name='Pages Visited',
        line=dict(color='#1f77b4', width=3),
        yaxis='y'
    ))
    
    # Success rate over time
    fig.add_trace(go.Scatter(
        x=df['time'],
        y=df['success_rate'],
        mode='lines+markers',
        name='Success Rate (%)',
        line=dict(color='#ff7f0e', width=2),
        yaxis='y2'
    ))
    
    # Layout with dual y-axes
    fig.update_layout(
        title="Real-time Crawling Performance",
        xaxis_title="Time (seconds)",
        yaxis=dict(
            title="Pages Visited",
            side="left"
        ),
        yaxis2=dict(
            title="Success Rate (%)",
            side="right",
            overlaying="y",
            range=[0, 100]
        ),
        height=300,
        showlegend=True,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    chart_placeholder.plotly_chart(fig, use_container_width=True)

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
        # Support both old format (cache_domain_timestamp.json) and new format (domain_date_time_pages.json)
        if (file.startswith('cache_') or file.endswith('pages.json')) and file.endswith('.json'):
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    
                    # Extract domain from filename
                    if file.startswith('cache_'):
                        # Old format: cache_www.domain.com_timestamp.json
                        domain = file.split('_')[1] if len(file.split('_')) > 1 else 'unknown'
                        domain = domain.replace('www.', '')
                    else:
                        # New format: domain_date_time_pages.json
                        domain = file.split('_')[0]
                    
                    cache_files.append({
                        'filename': file,
                        'domain': domain,
                        'crawled_at': data.get('crawled_at', 'Unknown'),
                        'total_pages': data.get('total_pages', 0)
                    })
            except:
                continue
    
    # Sort by creation time (newest first)
    cache_files.sort(key=lambda x: x['crawled_at'], reverse=True)
    return cache_files

def crawl_website(url, max_pages, delay):
    """Crawl website and return content"""
    crawler = WebCrawler(max_pages=max_pages, delay=delay)
    
    # Estimate total pages before crawling
    with st.spinner("Analyzing website structure..."):
        estimation_result = crawler.estimate_total_pages(url)
        
    # Show estimation to user with proper source attribution
    if estimation_result['total_pages'] is not None:
        estimated_total = estimation_result['total_pages']
        source_info = {
            'sitemap': f"Based on website sitemap analysis: {estimation_result['details']}",
            'third_party': f"Estimated using {estimation_result.get('service_name', 'third-party service')}: {estimation_result['details']}"
        }.get(estimation_result['source'], estimation_result['details'])
        
        if estimated_total > max_pages:
            st.info(f"📊 Website has {estimated_total} pages. You'll crawl {max_pages} pages ({(max_pages/estimated_total)*100:.1f}% coverage)")
        else:
            st.info(f"📊 Website has {estimated_total} pages. You'll crawl up to {max_pages} pages (potentially 100% coverage)")
        st.caption(source_info)
        
        # Store estimation data for later use
        crawler.estimation_result = estimation_result
    else:
        st.warning(f"⚠️ {estimation_result['details']}")
        st.caption("Coverage percentage cannot be calculated without reliable website size information")
        crawler.estimation_result = estimation_result
    
    # Interactive Progress tracking with metrics
    progress_container = st.container()
    
    with progress_container:
        st.markdown("### 🚀 Crawling Progress")
        
        # Progress bar with custom styling
        progress_bar = st.progress(0)
        
        # Real-time metrics - each on its own line for sidebar
        visited_metric = st.metric("Pages Visited", "0")
        extracted_metric = st.metric("Content Extracted", "0")
        success_rate_metric = st.metric("Success Rate", "0%")
        eta_metric = st.metric("Est. Time Left", "Calculating...")
        
        # Status and current page display
        status_container = st.container()
        with status_container:
            current_page_text = st.empty()
            detailed_status = st.empty()
        
        # Performance chart placeholder
        chart_placeholder = st.empty()
    
    # Track crawling performance
    crawl_start_time = time.time()
    performance_data = []
    
    def progress_callback(visited, extracted, current_url=None, page_title=None):
        current_time = time.time()
        elapsed_time = current_time - crawl_start_time
        
        # Update progress bar based on max_pages limit
        progress = min(visited / max_pages, 1.0)
        progress_bar.progress(progress)
        
        # Calculate metrics
        success_rate = (extracted / visited * 100) if visited > 0 else 0
        pages_per_minute = (visited / elapsed_time * 60) if elapsed_time > 0 else 0
        
        # Estimate time remaining based on crawl limit
        if pages_per_minute > 0 and visited < max_pages:
            remaining_pages = max_pages - visited
            eta_seconds = remaining_pages / pages_per_minute * 60
            eta_text = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
        else:
            eta_text = "Almost done!"
            
        visited_metric.metric("Pages Visited", f"{visited}/{max_pages}")
        extracted_metric.metric("Content Extracted", str(extracted))
        success_rate_metric.metric("Success Rate", f"{success_rate:.1f}%")
        eta_metric.metric("Time Left", eta_text)
        
        # Update status text with current page details
        if current_url and page_title:
            # Truncate long URLs and titles for display
            display_url = current_url if len(current_url) <= 60 else current_url[:57] + "..."
            display_title = page_title if len(page_title) <= 50 else page_title[:47] + "..."
            
            current_page_text.markdown(f"**Currently crawling:** {display_title}")
            detailed_status.markdown(f"🔗 **URL:** {display_url}")
        else:
            current_page_text.markdown(f"**Processing:** Page {visited}")
            detailed_status.markdown(f"⏱️ **Elapsed:** {int(elapsed_time)}s | 📈 **Speed:** {pages_per_minute:.1f} pages/min")
        
        # Store performance data for chart
        performance_data.append({
            'time': elapsed_time,
            'visited': visited,
            'extracted': extracted,
            'success_rate': success_rate
        })
        
        # Update performance chart every 5 pages
        if len(performance_data) > 1 and visited % 5 == 0:
            update_progress_chart(chart_placeholder, performance_data)
    
    try:
        content = crawler.crawl_website(url, progress_callback)
        progress_bar.progress(1.0)
        
        # Final completion message
        current_page_text.markdown("**✅ Crawling completed successfully!**")
        detailed_status.markdown(f"🎉 **Final Result:** Extracted {len(content)} pages with content")
        
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
    
    # Disclaimer
    st.info("""
    **Responsible Crawling Notice:** We respect website policies including robots.txt, sitemap.xml, and other crawling directives. 
    Only publicly accessible pages are crawled (no authentication required). We crawl both static and dynamic content 
    with a 1-5 second delay between requests and 10-second page load timeout to be respectful to server resources.
    """)
    
    # Sidebar with collapsible sections
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
            st.markdown("#### 🕷️ Web Crawler")
            
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
            if st.button("🔍 Start Crawling", disabled=st.session_state.crawl_in_progress):
                if url_input:
                    st.session_state.crawl_in_progress = True
                    domain = urlparse(url_input).netloc
                    st.session_state.current_domain = domain
                    
                    with st.spinner(f"Crawling {domain}..."):
                        content, stats, cache_file = crawl_website(url_input, max_pages, delay)
                        
                        if content:
                            st.session_state.crawled_content = content
                            st.session_state.crawl_stats = stats
                            
                            # Initialize RAG engine and process content (creates embeddings)
                            with st.spinner("Processing content for AI analysis..."):
                                rag_engine = WebRAGEngine(collection_name=f"web_{domain.replace('.', '_')}")
                                rag_engine.process_web_content(content, domain, use_cached_embeddings=False)
                                st.session_state.rag_engine = rag_engine
                            
                            # Re-save cache with embeddings included
                            with st.spinner("Saving embeddings to cache..."):
                                crawler = WebCrawler()
                                crawler.scraped_content = content  # Set the content with embeddings
                                updated_cache_file = crawler.save_cache()
                                st.info(f"Cache updated with embeddings: {updated_cache_file}")
                            
                            st.success(f"✅ Successfully analyzed {len(content)} pages!")
                        else:
                            st.error("Failed to crawl website. Please check the URL and try again.")
                    
                    st.session_state.crawl_in_progress = False
                else:
                    st.warning("Please enter a valid URL")
        
        # Cache Management Drawer
        elif st.session_state.active_drawer == 'cache':
            st.markdown("#### 💾 Cache Management")
            cache_files = get_cache_files()
            st.session_state.cache_files = cache_files
            
            if cache_files:
                def format_cache_name(filename):
                    if filename is None:
                        return "Select cache file..."
                    
                    # Handle new readable format (domain_date_time_pages.json)
                    if filename.endswith('pages.json'):
                        parts = filename.replace('.json', '').split('_')
                        if len(parts) >= 4:
                            domain = parts[0]
                            date_part = parts[1]  # e.g., "Dec-15-2025"
                            time_part = parts[2]  # e.g., "2-28pm"
                            pages_part = parts[3]  # e.g., "48pages"
                            
                            # Convert to more readable format
                            try:
                                # Parse date: "Dec-15-2025" -> "December 15, 2025"
                                date_obj = datetime.strptime(date_part, "%b-%d-%Y")
                                readable_date = date_obj.strftime("%B %d, %Y")
                                
                                # Parse time: "2-28pm" -> "2:28 PM"
                                time_clean = time_part.replace('pm', ' PM').replace('am', ' AM')
                                time_clean = time_clean.replace('-', ':')
                                
                                # Extract page count
                                page_count = pages_part.replace('pages', '')
                                
                                return f"{domain} - {readable_date} at {time_clean} ({page_count} pages)"
                            except:
                                pass
                    
                    # Handle old format (cache_domain_timestamp.json)
                    elif filename.startswith('cache_'):
                        parts = filename.split('_')
                        if len(parts) >= 3:
                            domain = parts[1].replace('www.', '')
                            timestamp = parts[2].replace('.json', '')
                            try:
                                # Parse timestamp and make it user-friendly
                                dt = datetime.strptime(timestamp, "%Y%m%d%H%M%S")
                                date_str = dt.strftime("%B %d, %Y at %I:%M %p")
                                pages = next(f['total_pages'] for f in cache_files if f['filename'] == filename)
                                return f"{domain} - {date_str} ({pages} pages)"
                            except:
                                pass
                    
                    # Fallback to original format
                    try:
                        pages = next(f['total_pages'] for f in cache_files if f['filename'] == filename)
                        return f"{filename} ({pages} pages)"
                    except:
                        return filename
                
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
                                
                                st.success(f"✅ Loaded {len(content)} pages from cache!")
                            else:
                                st.error("Cache file is empty or corrupted")
                        except Exception as e:
                            st.error(f"Error loading cache: {str(e)}")
            else:
                st.info("No cache files found")
        
        # Content Overview Drawer
        elif st.session_state.active_drawer == 'overview':
            if st.session_state.crawled_content:
                st.markdown("#### 📊 Content Overview")
                
                # Quick stats
                stats = st.session_state.crawl_stats
                if stats:
                    st.metric("Total Pages", stats.get('total_pages', 0))
                    st.metric("Total Words", f"{stats.get('total_words', 0):,}")
                    st.metric("Avg Words/Page", f"{stats.get('average_words_per_page', 0):.0f}")
                    
                    # Coverage percentage if available with source attribution
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
                        st.caption(f"Crawled {stats['total_pages']} of {estimated_total} total pages")
                        
                        # Show source of size estimation
                        size_source = stats.get('size_source', 'unknown')
                        if size_source == 'sitemap':
                            st.caption(f"📋 Size based on website sitemap analysis")
                        elif size_source == 'third_party':
                            service_name = stats.get('size_details', '').split(':')[0] if ':' in stats.get('size_details', '') else 'third-party service'
                            st.caption(f"🔍 Size estimated using {service_name}")
                        
                    else:
                        # Check if we have estimation result but no coverage
                        estimation_result = stats.get('estimation_result', {})
                        if estimation_result.get('source') == 'unavailable':
                            st.markdown("**Website Coverage:** :gray[Cannot be calculated]")
                            st.caption("⚠️ Website does not provide sitemap or reliable page count information")
                        elif estimation_result.get('source') == 'error':
                            st.markdown("**Website Coverage:** :gray[Cannot be calculated]")
                            st.caption("❌ Error analyzing website structure")
                        else:
                            st.markdown("**Website Coverage:** :gray[Unknown]")
                            st.caption("📊 Website size analysis incomplete")
                
                # Content summary
                if st.session_state.rag_engine:
                    summary = st.session_state.rag_engine.get_content_summary()
                    st.markdown("**Content Chunks:** " + str(summary.get('total_chunks', 0)))
            else:
                st.info("No content loaded yet")
    
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
        
        # Answer verbosity selector
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
        
        if question and st.session_state.rag_engine:
            with st.spinner("Analyzing content..."):
                result = st.session_state.rag_engine.analyze_content(question, verbosity=st.session_state.answer_verbosity)
                
                # Display answer with verbosity indicator
                verbosity_indicators = {
                    'concise': '🎯 Concise',
                    'balanced': '⚖️ Balanced',
                    'comprehensive': '📖 Detailed'
                }
                st.markdown(f"### 🤖 Analysis Result ({verbosity_indicators[st.session_state.answer_verbosity]})")
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
                
                # Reliability Tips Section - moved here after Sources
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
        


if __name__ == "__main__":
    main()