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
                
                # Always extract links even if content is minimal
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract title
                title_element = soup.find('title')
                title = title_element.get_text() if title_element else url
                
                if text_content and len(text_content.strip()) > 100:
                    page_data = {
                        'url': url,
                        'title': title.strip() if title else url,
                        'content': text_content,
                        'word_count': len(text_content.split()),
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    self.crawled_content.append(page_data)
                    pages_extracted += 1
                
                # Always look for links regardless of content extraction success
                links = soup.find_all('a', href=True)
                discovered_links = []
                
                for link in links:
                    href_attr = link.get('href')
                    if not href_attr:
                        continue
                    
                    # Handle href as string
                    href = str(href_attr).strip()
                    if not href:
                        continue
                        
                    # Convert relative URLs to absolute
                    if href.startswith('/'):
                        full_url = f"https://{base_domain}{href}"
                    elif href.startswith('http'):
                        # Only include URLs from the same domain
                        if base_domain in href:
                            full_url = href
                        else:
                            continue
                    else:
                        # Skip other types of links (mailto, javascript, etc.)
                        continue
                    
                    # Clean up URL fragments and parameters
                    clean_url = full_url.split('#')[0].split('?')[0]
                    
                    # Skip if already visited or queued
                    if clean_url not in visited_urls and clean_url not in to_visit and clean_url != url:
                        discovered_links.append(clean_url)
                
                # Add discovered links to queue
                to_visit.extend(discovered_links[:30])
                
                if progress_callback:
                    progress_callback(pages_visited, pages_extracted, url, title, len(discovered_links), len(to_visit))
                
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
    
    def estimate_total_pages(self, start_url: str) -> Dict[str, Any]:
        """Estimate total number of pages using authoritative sources"""
        try:
            parsed_url = urlparse(start_url)
            domain = parsed_url.netloc
            base_url = f"{parsed_url.scheme}://{domain}"
            
            # Try sitemap.xml first
            sitemap_urls = [
                f"{base_url}/sitemap.xml",
                f"{base_url}/sitemap_index.xml",
                f"{start_url.rstrip('/')}/sitemap.xml"
            ]
            
            for sitemap_url in sitemap_urls:
                try:
                    response = requests.get(sitemap_url, timeout=10)
                    if response.status_code == 200:
                        # Count URLs in sitemap
                        soup = BeautifulSoup(response.content, 'xml')
                        urls = soup.find_all(['url', 'sitemap'])
                        
                        if urls:
                            url_count = len(urls)
                            return {
                                'total_pages': url_count,
                                'source': 'sitemap',
                                'details': f"Found {url_count} URLs in sitemap.xml",
                                'confidence': 'high'
                            }
                except:
                    continue
            
            # Try robots.txt for sitemap references
            try:
                robots_url = f"{base_url}/robots.txt"
                response = requests.get(robots_url, timeout=10)
                if response.status_code == 200:
                    robots_content = response.text
                    sitemap_lines = [line for line in robots_content.split('\n') 
                                   if line.strip().lower().startswith('sitemap:')]
                    
                    if sitemap_lines:
                        # Try each sitemap found in robots.txt
                        for sitemap_line in sitemap_lines:
                            sitemap_url = sitemap_line.split(':', 1)[1].strip()
                            try:
                                response = requests.get(sitemap_url, timeout=10)
                                if response.status_code == 200:
                                    soup = BeautifulSoup(response.content, 'xml')
                                    urls = soup.find_all(['url', 'sitemap'])
                                    if urls:
                                        url_count = len(urls)
                                        return {
                                            'total_pages': url_count,
                                            'source': 'robots',
                                            'details': f"Found {url_count} URLs via robots.txt sitemap",
                                            'confidence': 'medium'
                                        }
                            except:
                                continue
            except:
                pass
            
            # Fallback: estimate based on crawled content
            crawled_count = len(self.crawled_content) if self.crawled_content else 1
            estimated = max(crawled_count * 2, 10)  # Conservative estimate
            
            return {
                'total_pages': estimated,
                'source': 'crawled',
                'details': f"Estimated based on {crawled_count} crawled pages",
                'confidence': 'low'
            }
            
        except Exception as e:
            return {
                'total_pages': None,
                'source': 'error',
                'details': f"Error estimating size: {str(e)}",
                'confidence': 'none'
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
                "sources": sources if relevant_chunks else [{"title": meta["title"], "url": meta["url"], "confidence": 0.3} for meta in self.chunk_metadata[:3]],
                "confidence": confidence,
                "chunks_used": len(relevant_chunks) if relevant_chunks else 3
            }
            
        except Exception as e:
            return {"answer": f"Error analyzing content: {str(e)}", "sources": [], "confidence": 0.0}
    
    def suggest_questions(self):
        """Generate context-sensitive questions based on crawled content"""
        if not self.chunks:
            return [
                "What are the main topics covered on this website?",
                "What products or services are offered?",
                "What are the key features mentioned?"
            ]
        
        try:
            # Get sample content for analysis
            sample_content = " ".join(self.chunks[:3])[:2000]  # First 3 chunks, max 2000 chars
            
            # Extract domain for context
            domain = ""
            if self.chunk_metadata:
                domain = self.chunk_metadata[0].get('domain', '')
            
            prompt = f"""Based on this website content from {domain}, generate 6 specific and relevant questions that users would want to ask about this particular website. The questions should be:

1. Specific to the actual content and services/products mentioned
2. Practical questions a visitor would have
3. Different from each other
4. Focused on key information available on the site

Website content sample:
{sample_content}

Return only the questions, one per line, without numbering or bullet points."""

            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.7
            )
            
            questions = response.choices[0].message.content.strip().split('\n')
            # Clean and filter questions
            filtered_questions = []
            for q in questions:
                q = q.strip()
                if q and len(q) > 10 and '?' in q:
                    filtered_questions.append(q)
            
            return filtered_questions[:6] if filtered_questions else [
                f"What services does {domain} offer?",
                f"What are the key features of {domain}?",
                f"How can I get started with {domain}?"
            ]
            
        except Exception as e:
            # Fallback to domain-specific generic questions
            domain_lower = domain.lower() if domain else ""
            
            if "verizon" in domain_lower:
                return [
                    "What wireless plans does Verizon offer?",
                    "What are Verizon's unlimited plan features?",
                    "How much do Verizon plans cost?",
                    "What devices work with Verizon?",
                    "Does Verizon offer international plans?",
                    "What are Verizon's prepaid options?"
                ]
            elif "amazon" in domain_lower:
                return [
                    "What products are available?",
                    "What are the shipping options?",
                    "What is the return policy?",
                    "How do I track my order?",
                    "What payment methods are accepted?",
                    "Are there any current deals or discounts?"
                ]
            else:
                return [
                    "What are the main products or services offered?",
                    "How can I contact customer support?",
                    "What are the pricing options?",
                    "What are the key features?",
                    "How do I get started?",
                    "What are the terms of service?"
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
            'current_page_title': None,
            'crawl_in_progress': False,
            'answer_verbosity': DEFAULT_VERBOSITY,
            'active_drawer': None,
            'current_question': "",
            'suggested_questions': []
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
                
                # Website coverage estimation
                if 'estimated_total_pages' in stats:
                    estimated_total = stats['estimated_total_pages']
                    coverage_percentage = stats.get('coverage_percentage', 0)
                    size_source = stats.get('size_source', 'unknown')
                    
                    st.markdown("**Website Coverage:**")
                    st.progress(coverage_percentage / 100)
                    st.caption(f"{coverage_percentage:.1f}% of estimated {estimated_total} total pages")
                    
                    if size_source == 'sitemap':
                        st.caption("📊 Size estimated from sitemap.xml")
                    elif size_source == 'robots':
                        st.caption("🤖 Size estimated from robots.txt")
                    elif size_source == 'crawled':
                        st.caption("⚠️ Size estimation unavailable - showing crawled pages only")
                    else:
                        st.caption("❓ Size estimation method unknown")
                else:
                    # Calculate website size estimation in background
                    if st.button("📊 Calculate Website Coverage", key="estimate_size"):
                        self.calculate_website_coverage()
                    else:
                        st.info("Click above to estimate total website size and coverage percentage")
                
                # Content summary
                if st.session_state.rag_engine:
                    summary = st.session_state.rag_engine.get_content_summary()
                    st.markdown("**Content Chunks:** " + str(summary.get('total_chunks', 0)))
        else:
            st.info("No content loaded yet")

    def calculate_website_coverage(self):
        """Calculate website size estimation in background"""
        if not st.session_state.crawled_content:
            return
        
        with st.spinner("Estimating total website size..."):
            try:
                # Get the base URL from crawled content
                base_url = st.session_state.crawled_content[0]['url']
                
                # Create crawler and estimate total pages
                crawler = SimpleWebCrawler()
                estimation_result = crawler.estimate_total_pages(base_url)
                
                # Update stats with estimation
                stats = st.session_state.crawl_stats.copy()
                stats['estimation_result'] = estimation_result
                
                if estimation_result['total_pages'] is not None:
                    estimated_total = estimation_result['total_pages']
                    coverage_percentage = (len(st.session_state.crawled_content) / estimated_total * 100) if estimated_total > 0 else 100
                    stats.update({
                        'estimated_total_pages': estimated_total,
                        'coverage_percentage': coverage_percentage,
                        'size_source': estimation_result['source'],
                        'size_details': estimation_result['details']
                    })
                    
                    st.session_state.crawl_stats = stats
                    st.success(f"Website size estimated: {estimated_total} total pages")
                else:
                    st.warning("Unable to estimate website size")
                    
            except Exception as e:
                st.error(f"Error estimating website size: {str(e)}")
    
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
            eta_placeholder = st.empty()
            performance_chart_placeholder = st.empty()
            
            # Add aggregated tracking placeholders
            links_found_placeholder = st.empty()
            queue_remaining_placeholder = st.empty()
            
            start_time = datetime.now()
            
            # Initialize performance tracking data
            performance_data = {
                'timestamps': [],
                'pages_visited': [],
                'content_extracted': [],
                'crawl_rate': []
            }
            
            # Initialize link tracking
            total_links_found = 0
            total_queue_remaining = 1  # Start with 1 (the initial URL)
        
        crawler = SimpleWebCrawler(max_pages=max_pages, delay=delay)
        
        def progress_callback(visited, extracted, current_url=None, page_title=None, new_links_count=0, queue_size=0):
            nonlocal total_links_found, total_queue_remaining
            
            pages_visited_placeholder.info(f"📄 **Pages Visited:** {visited}")
            pages_extracted_placeholder.info(f"✅ **Content Extracted:** {extracted}")
            
            progress = min(visited / max_pages, 1.0)
            progress_bar_placeholder.progress(progress, text=f"{progress:.0%} Complete")
            
            if current_url and page_title:
                title_display = page_title[:40] + "..." if len(page_title) > 40 else page_title
                current_page_placeholder.info(f"🔍 **Current:** {title_display}")
            
            # Update aggregated link tracking
            if new_links_count > 0:
                total_links_found += new_links_count
            total_queue_remaining = queue_size
            
            # Display aggregated link information
            links_found_placeholder.info(f"🔗 **Total Links Found:** {total_links_found}")
            queue_remaining_placeholder.info(f"📋 **Queue Remaining:** {total_queue_remaining}")
            
            # Calculate ETA and update performance tracking
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
                
                eta_placeholder.info(f"⏱️ **ETA:** {eta_display}")
                
                # Update performance data
                current_time = elapsed / 60  # in minutes
                performance_data['timestamps'].append(current_time)
                performance_data['pages_visited'].append(visited)
                performance_data['content_extracted'].append(extracted)
                performance_data['crawl_rate'].append(rate)
                
                # Update real-time performance chart
                if len(performance_data['timestamps']) > 1 and visited % 2 == 0:
                    try:
                        import plotly.graph_objects as go
                        
                        fig = go.Figure()
                        
                        # Add pages visited trace
                        fig.add_trace(go.Scatter(
                            x=performance_data['timestamps'],
                            y=performance_data['pages_visited'],
                            mode='lines+markers',
                            name='Pages Visited',
                            line=dict(color='#1f77b4', width=2),
                            marker=dict(size=4)
                        ))
                        
                        # Add content extracted trace
                        fig.add_trace(go.Scatter(
                            x=performance_data['timestamps'],
                            y=performance_data['content_extracted'],
                            mode='lines+markers',
                            name='Content Extracted',
                            line=dict(color='#ff7f0e', width=2),
                            marker=dict(size=4)
                        ))
                        
                        fig.update_layout(
                            title="Real-time Crawling Performance",
                            title_font_size=12,
                            xaxis_title="Time (minutes)",
                            yaxis_title="Pages",
                            height=180,
                            margin=dict(l=0, r=0, t=30, b=0),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            showlegend=True
                        )
                        
                        performance_chart_placeholder.plotly_chart(fig, use_container_width=True)
                    except ImportError:
                        pass  # Skip chart if plotly not available
        
        content = crawler.crawl_website(url, progress_callback=progress_callback)
        
        # Generate intelligent page title from content
        if content and len(content) > 0:
            st.session_state.current_page_title = self.generate_page_title(content, domain)
        else:
            st.session_state.current_page_title = domain
        
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
            
            # AI Analysis with detailed progress tracking
            with st.sidebar:
                st.markdown("---")
                st.markdown("### 🤖 AI Analysis Progress")
                
                analysis_progress = st.progress(0)
                analysis_status = st.empty()
                analysis_details = st.empty()
                
                analysis_status.info("🔍 **Step 1/4:** Initializing AI analysis engine...")
                analysis_progress.progress(0.1)
                time.sleep(0.3)
            
            # Initialize RAG engine with progress tracking
            try:
                with st.sidebar:
                    analysis_status.info("📝 **Step 2/4:** Processing content with smart chunking...")
                    analysis_details.text(f"Breaking {len(content)} pages into semantic chunks")
                    analysis_progress.progress(0.4)
                    time.sleep(0.3)
                
                rag_engine = SimpleRAGEngine()
                rag_engine.process_web_content(content, domain)
                
                with st.sidebar:
                    analysis_status.info("🧠 **Step 3/4:** Preparing semantic analysis capabilities...")
                    analysis_details.text(f"Created {len(rag_engine.chunks)} intelligent chunks for analysis")
                    analysis_progress.progress(0.7)
                    time.sleep(0.3)
                
                st.session_state.rag_engine = rag_engine
                # Clear previous suggested questions to generate new ones
                st.session_state.suggested_questions = []
                
                with st.sidebar:
                    analysis_status.success("✅ **Step 4/4:** AI analysis system ready!")
                    analysis_details.text("System optimized for intelligent question answering")
                    analysis_progress.progress(1.0)
                    
                    # Add detailed explanation
                    with st.expander("ℹ️ Why AI Analysis Takes Time"):
                        st.markdown("""
                        **Smart Chunking Process:**
                        - Breaks content into 1000-token chunks with 100-token overlap
                        - Respects paragraph and sentence boundaries for context
                        - Maintains source attribution for accurate citations
                        
                        **Semantic Preparation:**
                        - Prepares content for intelligent keyword-based search
                        - Creates rich metadata for source attribution
                        - Optimizes chunk structure for question-answering accuracy
                        
                        **Performance Optimization:**
                        - Caches processed chunks for faster repeat analysis
                        - Uses token-aware processing to maximize context
                        - Implements relevance scoring for better results
                        
                        This ensures high-quality, reliable answers with proper source attribution.
                        """)
                        
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
    
    def generate_page_title(self, content: List[Dict[str, Any]], domain: str) -> str:
        """Generate intelligent page title based on content analysis"""
        if not content:
            return domain
            
        try:
            # Collect sample text from multiple pages
            sample_texts = []
            for page in content[:5]:  # Use first 5 pages
                page_content = page.get('content', '')
                if page_content:
                    # Take first 200 chars from each page
                    sample_texts.append(page_content[:200])
            
            if not sample_texts:
                return domain
                
            combined_text = " ".join(sample_texts)[:1000]  # Limit to 1000 chars
            
            prompt = f"""Based on this website content, generate a concise, descriptive title (2-4 words max) that captures what this website is about. The title should be suitable for "Ask Questions about [Title]".

Website domain: {domain}
Content sample: {combined_text}

Examples of good titles:
- "Verizon Wireless Plans"
- "Amazon Shopping"
- "Netflix Streaming"
- "Apple Products"

Generate only the title, no explanations or quotes."""

            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0.3
            )
            
            generated_title = response.choices[0].message.content.strip().strip('"\'')
            
            # Validate the title (should be reasonable length and not contain weird characters)
            if generated_title and len(generated_title) <= 50 and not any(char in generated_title for char in ['<', '>', '{', '}', '[', ']']):
                return generated_title
            else:
                return domain
                
        except Exception as e:
            # Fallback to domain-based title
            domain_clean = domain.replace("www.", "").replace(".com", "").replace(".org", "").replace(".net", "")
            if "verizon" in domain_clean.lower():
                return "Verizon Services"
            elif "amazon" in domain_clean.lower():
                return "Amazon"
            elif "apple" in domain_clean.lower():
                return "Apple"
            elif "google" in domain_clean.lower():
                return "Google Services"
            else:
                return domain_clean.title()

    def get_cache_files(self) -> List[str]:
        """Get list of cache files"""
        cache_dir = Path("data/cache")
        if not cache_dir.exists():
            return []
        
        return sorted([f.name for f in cache_dir.glob("*.json")], reverse=True)
    
    def format_cache_name(self, filename: str) -> str:
        """Format cache filename for display"""
        if "_" in filename:
            parts = filename.replace(".json", "").split("_")
            if len(parts) >= 3:
                domain = parts[0]
                timestamp = parts[1] + "_" + parts[2] if len(parts) > 2 else parts[1]
                pages = parts[-1] if parts[-1].endswith("pages") else "pages"
                
                try:
                    # Handle new format: feb-19-2025_3-45pm
                    if "-" in timestamp and ("am" in timestamp.lower() or "pm" in timestamp.lower()):
                        # Parse new human-friendly format
                        date_part, time_part = timestamp.split("_")
                        month, day, year = date_part.split("-")
                        
                        # Convert month abbreviation to full name
                        month_names = {
                            "jan": "January", "feb": "February", "mar": "March", "apr": "April",
                            "may": "May", "jun": "June", "jul": "July", "aug": "August",
                            "sep": "September", "oct": "October", "nov": "November", "dec": "December"
                        }
                        full_month = month_names.get(month.lower(), month.capitalize())
                        
                        # Format time
                        time_formatted = time_part.replace("-", ":").upper()
                        
                        formatted_date = f"{full_month} {day}, {year} at {time_formatted}"
                    else:
                        # Handle old format: 20250219_144500
                        dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
                        formatted_date = dt.strftime("%B %d, %Y at %I:%M %p")
                    
                    return f"{domain} - {formatted_date} ({pages})"
                except:
                    pass
        
        return filename
    
    def save_cache(self, content: List[Dict[str, Any]], domain: str) -> str:
        """Save content to cache with human-friendly filename"""
        cache_dir = Path("data/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create human-friendly timestamp
        now = datetime.now()
        timestamp = now.strftime("%b-%d-%Y_%I-%M%p").lower()
        
        # Clean domain name for filename
        clean_domain = domain.replace("www.", "").replace(".com", "").replace(".org", "").replace(".net", "")
        
        filename = f"{clean_domain}_{timestamp}_{len(content)}pages.json"
        filepath = cache_dir / filename
        
        cache_data = {
            'domain': domain,
            'timestamp': now.isoformat(),
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
                💬 Ask Questions about {st.session_state.current_page_title or st.session_state.current_domain}
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.rag_engine:
            # Generate suggested questions only once per content
            if not st.session_state.suggested_questions:
                st.session_state.suggested_questions = st.session_state.rag_engine.suggest_questions()
            
            with st.expander("💡 Suggested Questions"):
                for i, question in enumerate(st.session_state.suggested_questions):
                    if st.button(f"❓ {question}", key=f"suggested_{i}"):
                        st.session_state.current_question = question
                        st.rerun()
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("**Your question:**")
            question = st.text_area(
                "Your question:",
                value=st.session_state.current_question,
                placeholder="Ask anything about the website content...",
                height=80,
                label_visibility="collapsed"
            )
            # Clear the current_question after displaying it
            if st.session_state.current_question:
                st.session_state.current_question = ""
        with col2:
            st.markdown("**Answer Style:**")
            verbosity = st.selectbox(
                "Response level:",
                options=['concise', 'balanced', 'comprehensive'],
                index=['concise', 'balanced', 'comprehensive'].index(st.session_state.answer_verbosity),
                format_func=lambda x: {
                    'concise': '🎯 Concise',
                    'balanced': '📊 Balanced', 
                    'comprehensive': '📚 Comprehensive'
                }[x],
                label_visibility="collapsed"
            )
            
            if verbosity != st.session_state.answer_verbosity:
                st.session_state.answer_verbosity = verbosity
        
        # Submit button for processing questions
        submit_col1, submit_col2 = st.columns([1, 4])
        with submit_col1:
            submit_button = st.button("🔍 Ask", type="primary", use_container_width=True)
        
        if question and st.session_state.rag_engine and submit_button:
            try:
                with st.spinner("🤔 Analyzing content..."):
                    result = st.session_state.rag_engine.analyze_content(question, verbosity=verbosity)
                
                if result.get('answer'):
                    st.markdown("### 💬 Answer")
                    st.markdown(result['answer'])
                    
                    # Enhanced reliability display
                    confidence = result.get('confidence', 0)
                    
                    # Convert confidence to reliability indicator
                    if confidence >= 0.8:
                        reliability_text = "Very Reliable"
                        reliability_color = "green"
                        reliability_icon = "🟢"
                    elif confidence >= 0.6:
                        reliability_text = "Mostly Reliable"
                        reliability_color = "lightgreen"
                        reliability_icon = "🟡"
                    elif confidence >= 0.4:
                        reliability_text = "Moderately Reliable"
                        reliability_color = "orange"
                        reliability_icon = "🟠"
                    else:
                        reliability_text = "Low Reliability"
                        reliability_color = "red"
                        reliability_icon = "🔴"
                    
                    st.markdown(f"### {reliability_icon} Reliability: {reliability_text} ({confidence:.0%})")
                    
                    # Reliability explanation
                    with st.expander("ℹ️ What does this reliability score mean?"):
                        st.markdown(f"""
                        **Reliability Score: {confidence:.0%}**
                        
                        This score indicates how confident the AI is in its answer based on:
                        - **Source Quality**: How well the found content matches your question
                        - **Content Clarity**: How clear and specific the source information is
                        - **Answer Completeness**: Whether sufficient information was available
                        
                        **Reliability Levels:**
                        - 🟢 **Very Reliable (80-100%)**: High confidence, comprehensive sources
                        - 🟡 **Mostly Reliable (60-79%)**: Good confidence, adequate sources  
                        - 🟠 **Moderately Reliable (40-59%)**: Some uncertainty, limited sources
                        - 🔴 **Low Reliability (0-39%)**: High uncertainty, insufficient sources
                        """)
                    
                    # Sources with confidence scores
                    if result.get('sources'):
                        st.markdown("### 📚 Sources")
                        for i, source in enumerate(result['sources'][:5], 1):
                            source_confidence = source.get('confidence', 0)
                            if source_confidence >= 0.7:
                                confidence_indicator = "🟢"
                            elif source_confidence >= 0.5:
                                confidence_indicator = "🟡"
                            else:
                                confidence_indicator = "🟠"
                            
                            st.markdown(f"{confidence_indicator} **{i}.** [{source['title']}]({source['url']}) *(Relevance: {source_confidence:.0%})*")
                    
                    # Reliability improvement tips
                    if confidence < 0.7:
                        with st.expander("💡 How to improve answer reliability"):
                            st.markdown("""
                            **To get more reliable answers:**
                            
                            1. **Be more specific**: Instead of "What services?" ask "What wireless plans are available?"
                            2. **Ask targeted questions**: Focus on specific topics covered on the website
                            3. **Try different phrasings**: Rephrase your question using different keywords
                            4. **Check if more content is needed**: Consider crawling more pages if the website is large
                            5. **Use exact terms**: Use terminology that appears on the website
                            
                            **Current issue may be:**
                            - Limited relevant content found for your specific question
                            - Question too broad or general for available content
                            - Key information may be on pages not yet crawled
                            """)
                else:
                    st.error("Could not generate an answer. Please try rephrasing your question.")
                    
            except Exception as e:
                st.error(f"Error processing question: {str(e)}")
        else:
            if not question:
                st.info("💡 Enter a question above or click on a suggested question to get started!")
            elif not st.session_state.rag_engine:
                st.warning("⚠️ Please crawl a website first to enable question answering.")
                            confidence_text = "Moderately Reliable"
                            confidence_color = "orange"
                        else:
                            confidence_text = "Limited Reliability"
                            confidence_color = "red"
                        
                        st.markdown(f"**Confidence:** :{confidence_color}[{confidence_text}]")
                        
                        if 'chunks_used' in result:
                            st.markdown(f"**Chunks analyzed:** {result['chunks_used']}")
        
        # Add comprehensive analytics section
        st.markdown("---")
        st.markdown("### 📊 Content Analytics & Tools")
        
        # Main tabs for different functionalities
        tab1, tab2, tab3 = st.tabs(["📊 Analytics", "🔍 Semantic Search", "📄 Raw Content"])
        
        with tab1:
            self.render_analytics_tab()
        
        with tab2:
            self.render_search_tab()
            
        with tab3:
            self.render_content_tab()
    
    def render_analytics_tab(self):
        """Render comprehensive analytics with visualizations"""
        if not st.session_state.crawled_content:
            st.info("No content available for analysis")
            return
            
        stats = st.session_state.crawl_stats
        content = st.session_state.crawled_content
        
        # Key metrics overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Pages", stats.get('total_pages', 0))
        with col2:
            st.metric("Total Words", f"{stats.get('total_words', 0):,}")
        with col3:
            st.metric("Avg Words/Page", f"{stats.get('average_words_per_page', 0):.0f}")
        with col4:
            if st.session_state.rag_engine:
                summary = st.session_state.rag_engine.get_content_summary()
                st.metric("Content Chunks", summary.get('total_chunks', 0))
        
        st.markdown("---")
        
        # Advanced analytics with visualizations
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            
            # Content analysis
            word_counts = [page.get('word_count', 0) for page in content]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Word count distribution
                fig_hist = px.histogram(
                    x=word_counts,
                    title="📊 Word Count Distribution",
                    labels={'x': 'Words per Page', 'y': 'Number of Pages'},
                    nbins=min(20, len(set(word_counts))),
                    color_discrete_sequence=['#1f77b4']
                )
                fig_hist.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig_hist, use_container_width=True)
                
                # Top pages by content
                top_pages = sorted(content, key=lambda x: x.get('word_count', 0), reverse=True)[:5]
                st.markdown("**📈 Top Pages by Content:**")
                for i, page in enumerate(top_pages):
                    st.write(f"{i+1}. **{page.get('title', 'Untitled')[:40]}...** ({page.get('word_count', 0)} words)")
            
            with col2:
                # URL depth analysis
                url_depths = []
                for page in content:
                    url = page.get('url', '')
                    depth = len([p for p in url.split('/') if p and p not in ['http:', 'https:']]) - 1
                    url_depths.append(max(0, depth))
                
                fig_depth = px.histogram(
                    x=url_depths,
                    title="🌐 Page Depth Distribution",
                    labels={'x': 'URL Depth Level', 'y': 'Number of Pages'},
                    nbins=max(1, max(url_depths) if url_depths else 1),
                    color_discrete_sequence=['#ff7f0e']
                )
                fig_depth.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig_depth, use_container_width=True)
                
                # Content quality insights
                st.markdown("**📋 Content Quality Insights:**")
                if word_counts:
                    avg_words = sum(word_counts) / len(word_counts)
                    high_quality = len([w for w in word_counts if w > avg_words * 1.5])
                    low_quality = len([w for w in word_counts if w < avg_words * 0.5])
                    
                    st.write(f"• **High-content pages:** {high_quality} ({high_quality/len(content)*100:.1f}%)")
                    st.write(f"• **Low-content pages:** {low_quality} ({low_quality/len(content)*100:.1f}%)")
                    st.write(f"• **Average depth:** {sum(url_depths)/len(url_depths):.1f} levels")
            
            # Network visualization
            if len(content) <= 15:
                st.markdown("### 🕸️ Page Relationship Network")
                self.create_simple_network_visualization(content)
                
        except ImportError:
            # Fallback for when plotly is not available
            st.info("📊 **Basic Analytics** (Install plotly for advanced visualizations)")
            
            # Simple text-based analytics
            word_counts = [page.get('word_count', 0) for page in content]
            if word_counts:
                st.write(f"**Content Statistics:**")
                st.write(f"• Longest page: {max(word_counts)} words")
                st.write(f"• Shortest page: {min(word_counts)} words")
                st.write(f"• Median words: {sorted(word_counts)[len(word_counts)//2]} words")
                
                # Show top pages
                top_pages = sorted(content, key=lambda x: x.get('word_count', 0), reverse=True)[:3]
                st.write("**Top Pages by Content:**")
                for i, page in enumerate(top_pages):
                    st.write(f"{i+1}. {page.get('title', 'Untitled')[:50]}... ({page.get('word_count', 0)} words)")
    
    def create_simple_network_visualization(self, content):
        """Create a simple network visualization using available tools"""
        try:
            from streamlit_agraph import agraph, Node, Edge, Config
            
            nodes = []
            edges = []
            
            # Create nodes for each page (limit to 10 for clarity)
            for i, page in enumerate(content[:10]):
                title = page.get('title', f'Page {i+1}')
                short_title = title[:15] + "..." if len(title) > 15 else title
                word_count = page.get('word_count', 100)
                
                # Node size based on content length
                size = max(15, min(40, word_count / 20))
                
                nodes.append(Node(
                    id=str(i),
                    label=short_title,
                    size=size,
                    color="#1f77b4" if word_count > 200 else "#ff7f0e"
                ))
            
            # Create edges based on URL similarity
            for i, page1 in enumerate(content[:10]):
                for j, page2 in enumerate(content[:10]):
                    if i < j:  # Avoid duplicate edges
                        url1_parts = set(page1.get('url', '').split('/'))
                        url2_parts = set(page2.get('url', '').split('/'))
                        
                        # Connect pages with similar URL structures
                        shared_parts = url1_parts.intersection(url2_parts)
                        if len(shared_parts) > 2:  # More than just protocol and domain
                            edges.append(Edge(source=str(i), target=str(j), type="CURVE_SMOOTH"))
            
            config = Config(
                width=700,
                height=400,
                directed=False,
                physics=True,
                hierarchical=False,
                nodeHighlightBehavior=True,
                highlightColor="#F7A7A6"
            )
            
            if nodes:
                agraph(nodes=nodes, edges=edges, config=config)
            else:
                st.info("Network visualization requires page data")
                
        except ImportError:
            st.info("Network visualization requires streamlit-agraph package")
    
    def render_search_tab(self):
        """Render comprehensive semantic search interface"""
        st.markdown("### 🔍 Semantic Search")
        
        # Explanation section
        with st.expander("ℹ️ How Semantic Search Works"):
            st.markdown("""
            **Semantic Search vs AI Questions:**
            - **Semantic Search**: Finds relevant content chunks using keyword matching and relevance scoring
            - **AI Questions**: Uses GPT-4 to analyze content and provide intelligent answers
            
            **Search Features:**
            - Searches across all content chunks for maximum coverage
            - Ranks results by relevance score (word overlap + proximity)
            - Shows exact content snippets with source attribution
            - No AI interpretation - pure content retrieval
            """)
        
        # Search interface
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_query = st.text_input(
                "Search query:",
                placeholder="Enter keywords, phrases, or topics to find relevant content...",
                help="Use specific terms for better results. Multiple keywords will be searched across all content."
            )
        
        with col2:
            max_results = st.selectbox("Max results:", [5, 10, 15, 20], index=1)
        
        if search_query and st.session_state.rag_engine:
            with st.spinner("Searching through content chunks..."):
                # Perform semantic search
                results = st.session_state.rag_engine._semantic_search(search_query, k=max_results)
                
                if results:
                    st.success(f"Found {len(results)} relevant results (sorted by relevance)")
                    
                    # Display results with enhanced formatting
                    for i, result in enumerate(results):
                        relevance = result['relevance']
                        metadata = result['metadata']
                        chunk = result['chunk']
                        
                        # Color code relevance
                        if relevance >= 0.3:
                            relevance_color = "🟢"
                            relevance_text = "High"
                        elif relevance >= 0.15:
                            relevance_color = "🟡"
                            relevance_text = "Medium"
                        else:
                            relevance_color = "🟠"
                            relevance_text = "Low"
                        
                        with st.expander(f"{relevance_color} **Result {i+1}**: {metadata['title'][:60]}... | Relevance: {relevance_text} ({relevance:.3f})"):
                            col1, col2 = st.columns([2, 1])
                            
                            with col2:
                                st.markdown("**Source Information:**")
                                st.write(f"**Page:** {metadata['title']}")
                                st.write(f"**URL:** {metadata['url']}")
                                st.write(f"**Relevance:** {relevance:.3f}")
                                st.write(f"**Word Count:** {metadata.get('word_count', 'N/A')}")
                                
                                # Highlight matching terms
                                query_words = set(search_query.lower().split())
                                chunk_words = set(chunk.lower().split())
                                matches = query_words.intersection(chunk_words)
                                
                                if matches:
                                    st.write(f"**Matching terms:** {', '.join(matches)}")
                            
                            with col1:
                                st.markdown("**Content Preview:**")
                                
                                # Highlight search terms in preview
                                preview = chunk[:800] + "..." if len(chunk) > 800 else chunk
                                
                                # Simple highlighting
                                highlighted_preview = preview
                                for word in search_query.split():
                                    if len(word) > 2:
                                        highlighted_preview = highlighted_preview.replace(
                                            word, f"**{word}**"
                                        )
                                        highlighted_preview = highlighted_preview.replace(
                                            word.lower(), f"**{word.lower()}**"
                                        )
                                        highlighted_preview = highlighted_preview.replace(
                                            word.capitalize(), f"**{word.capitalize()}**"
                                        )
                                
                                st.markdown(highlighted_preview)
                                
                                # Full content toggle
                                if st.button(f"Show full content", key=f"full_{i}"):
                                    st.text_area("Full chunk content:", chunk, height=200, key=f"content_{i}")
                    
                    # Search statistics
                    st.markdown("---")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        avg_relevance = sum(r['relevance'] for r in results) / len(results)
                        st.metric("Average Relevance", f"{avg_relevance:.3f}")
                    
                    with col2:
                        high_relevance = len([r for r in results if r['relevance'] >= 0.2])
                        st.metric("High Relevance Results", high_relevance)
                    
                    with col3:
                        total_chunks = len(st.session_state.rag_engine.chunks)
                        coverage = (len(results) / total_chunks * 100) if total_chunks > 0 else 0
                        st.metric("Search Coverage", f"{coverage:.1f}%")
                        
                else:
                    st.warning("No relevant content found. Try different keywords or broader search terms.")
                    
                    # Search suggestions
                    if st.session_state.rag_engine and hasattr(st.session_state.rag_engine, 'chunks'):
                        st.info("**Search Tips:**")
                        st.write("• Try broader terms or synonyms")
                        st.write("• Use multiple related keywords")
                        st.write("• Check spelling and try variations")
        
        elif search_query and not st.session_state.rag_engine:
            st.error("Search functionality requires content to be loaded and processed first.")
    
    def render_content_tab(self):
        """Render the raw content tab with detailed page information"""
        st.markdown("### 📄 Raw Content Browser")
        
        if not st.session_state.crawled_content:
            st.info("No content available")
            return
            
        # Content selector
        content = st.session_state.crawled_content
        page_options = [f"{i+1}. {page.get('title', 'Untitled')[:50]}..." for i, page in enumerate(content)]
        
        selected_page_idx = st.selectbox("Select page to view:", range(len(page_options)), format_func=lambda x: page_options[x])
        
        if selected_page_idx is not None:
            page = content[selected_page_idx]
            
            col1, col2 = st.columns([2, 1])
            
            with col2:
                st.markdown("**📋 Page Information:**")
                st.write(f"**Title:** {page.get('title', 'N/A')}")
                st.write(f"**URL:** {page.get('url', 'N/A')}")
                st.write(f"**Word Count:** {page.get('word_count', 'N/A')}")
                
                if 'timestamp' in page:
                    st.write(f"**Crawled:** {page['timestamp']}")
                
                if st.session_state.rag_engine and hasattr(st.session_state.rag_engine, 'chunks'):
                    # Show chunks for this page
                    page_chunks = [i for i, meta in enumerate(st.session_state.rag_engine.chunk_metadata) 
                                 if meta['url'] == page.get('url')]
                    st.write(f"**Content Chunks:** {len(page_chunks)}")
                    
                    if page_chunks:
                        chunk_sizes = [len(st.session_state.rag_engine.chunks[i].split()) for i in page_chunks]
                        st.write(f"**Avg Chunk Size:** {sum(chunk_sizes)/len(chunk_sizes):.0f} words")
            
            with col1:
                st.markdown("**📄 Page Content:**")
                content_text = page.get('content', 'No content available')
                st.text_area("Page content:", content_text, height=400, disabled=True)
                
                # Show individual chunks if available
                if st.session_state.rag_engine and hasattr(st.session_state.rag_engine, 'chunks'):
                    page_chunks = [i for i, meta in enumerate(st.session_state.rag_engine.chunk_metadata) 
                                 if meta['url'] == page.get('url')]
                    
                    if page_chunks and st.checkbox("Show content chunks"):
                        st.markdown("**🧩 Content Chunks:**")
                        for i, chunk_idx in enumerate(page_chunks):
                            with st.expander(f"Chunk {i+1} ({len(st.session_state.rag_engine.chunks[chunk_idx].split())} words)"):
                                st.text(st.session_state.rag_engine.chunks[chunk_idx])

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