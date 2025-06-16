"""
Web crawler module - handles website crawling and content extraction
"""
import json
import re
import time
from datetime import datetime
from typing import Dict, List, Any, Set, Optional, Callable
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
import trafilatura
from config.settings import CRAWLER_SETTINGS, CACHE_DIR


class WebCrawler:
    """Intelligent web crawler for extracting content from websites"""
    
    def __init__(self, max_pages: int = None, delay: float = None):
        self.max_pages = max_pages or CRAWLER_SETTINGS["max_pages"]
        self.delay = delay or CRAWLER_SETTINGS["delay"]
        self.timeout = CRAWLER_SETTINGS["timeout"]
        self.user_agent = CRAWLER_SETTINGS["user_agent"]
        self.scraped_content = []
        self.visited_urls = set()
        
        # Request session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })

    def _is_valid_url(self, url: str, base_domain: str) -> bool:
        """Check if URL is valid for crawling"""
        try:
            parsed = urlparse(url)
            
            # Must be same domain
            if parsed.netloc != base_domain:
                return False
            
            # Skip common non-content files
            skip_extensions = {'.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', 
                             '.zip', '.rar', '.tar', '.gz', '.exe', '.dmg', '.iso',
                             '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.ico',
                             '.mp3', '.mp4', '.wav', '.avi', '.mov', '.wmv', '.flv',
                             '.css', '.js', '.xml', '.json', '.rss', '.atom'}
            
            path_lower = parsed.path.lower()
            if any(path_lower.endswith(ext) for ext in skip_extensions):
                return False
            
            # Skip common non-content paths
            skip_patterns = ['/api/', '/admin/', '/login', '/register', '/cart', '/checkout',
                           '/search', '/tag/', '/category/', '/archive/', '/feed']
            
            if any(pattern in path_lower for pattern in skip_patterns):
                return False
            
            return True
            
        except Exception:
            return False

    def _extract_links(self, html: str, base_url: str) -> Set[str]:
        """Extract all valid links from HTML"""
        links = set()
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            base_domain = urlparse(base_url).netloc
            
            for link_tag in soup.find_all('a', href=True):
                href = link_tag.get('href')
                if href:
                    full_url = urljoin(base_url, href)
                    
                    # Clean URL (remove fragment)
                    full_url = full_url.split('#')[0]
                    
                    if self._is_valid_url(full_url, base_domain):
                        links.add(full_url)
        
        except Exception as e:
            print(f"Error extracting links: {e}")
        
        return links

    def _get_page_content(self, url: str) -> Optional[Dict[str, Any]]:
        """Extract content from a single page"""
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            # Extract clean text using trafilatura
            text_content = trafilatura.extract(response.text)
            if not text_content or len(text_content.strip()) < 100:
                return None
            
            # Parse HTML for metadata
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract metadata
            title = soup.find('title')
            title = title.get_text().strip() if title else urlparse(url).path
            
            description = soup.find('meta', attrs={'name': 'description'})
            description = description.get('content', '').strip() if description else ''
            
            # Extract headings
            headings = [h.get_text().strip() for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])]
            headings = [h for h in headings if h]  # Remove empty headings
            
            # Count words
            word_count = len(text_content.split())
            
            return {
                'url': url,
                'title': title,
                'content': text_content,
                'description': description,
                'headings': headings,
                'word_count': word_count,
                'scraped_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error processing {url}: {e}")
            return None

    def crawl_website(self, start_url: str, progress_callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """Crawl website starting from the given URL"""
        self.scraped_content = []
        self.visited_urls = set()
        
        domain = urlparse(start_url).netloc
        urls_to_visit = [start_url]
        
        while urls_to_visit and len(self.visited_urls) < self.max_pages:
            current_url = urls_to_visit.pop(0)
            
            if current_url in self.visited_urls:
                continue
            
            self.visited_urls.add(current_url)
            
            # Get page content
            page_content = self._get_page_content(current_url)
            
            if page_content:
                self.scraped_content.append(page_content)
                
                # Extract new links only if we haven't hit the limit
                if len(self.visited_urls) < self.max_pages:
                    try:
                        response = self.session.get(current_url, timeout=self.timeout)
                        new_links = self._extract_links(response.text, current_url)
                        
                        # Add new unvisited links
                        for link in new_links:
                            if link not in self.visited_urls and link not in urls_to_visit:
                                urls_to_visit.append(link)
                                
                    except Exception as e:
                        print(f"Error extracting links from {current_url}: {e}")
                
                # Progress callback
                if progress_callback:
                    progress_callback(
                        len(self.visited_urls), 
                        len(self.scraped_content), 
                        current_url,
                        page_content.get('title', 'Untitled')
                    )
            
            # Rate limiting
            time.sleep(self.delay)
        
        return self.scraped_content

    def save_cache(self, filename: str = None) -> str:
        """Save scraped content to cache file"""
        if not self.scraped_content:
            return ""
        
        # Generate filename if not provided
        if not filename:
            domain = urlparse(self.scraped_content[0]['url']).netloc.replace('www.', '')
            timestamp = datetime.now()
            date_str = timestamp.strftime("%b-%d-%Y")
            time_str = timestamp.strftime("%-I-%M%p").lower()
            page_count = len(self.scraped_content)
            filename = f"{domain}_{date_str}_{time_str}_{page_count}pages.json"
        
        # Ensure cache directory exists
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        filepath = CACHE_DIR / filename
        
        # Prepare cache data
        cache_data = {
            'domain': urlparse(self.scraped_content[0]['url']).netloc,
            'total_pages': len(self.scraped_content),
            'crawled_at': datetime.now().isoformat(),
            'content': self.scraped_content
        }
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        
        return str(filepath)

    def load_cache(self, filename: str) -> List[Dict[str, Any]]:
        """Load content from cache file"""
        try:
            filepath = CACHE_DIR / filename if not filename.startswith('/') else filename
            
            with open(filepath, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            content = cache_data.get('content', [])
            self.scraped_content = content
            return content
            
        except Exception as e:
            print(f"Error loading cache {filename}: {e}")
            return []

    def estimate_total_pages(self, start_url: str) -> Dict[str, Any]:
        """Estimate total number of pages using authoritative sources"""
        domain = urlparse(start_url).netloc
        
        # Try sitemap first
        sitemap_result = self._analyze_sitemaps(start_url, domain)
        if sitemap_result['total_pages'] is not None:
            return sitemap_result
        
        # Return unavailable if no reliable source found
        return {
            'total_pages': None,
            'source': 'unavailable',
            'details': 'Website does not provide reliable page count information',
            'confidence': 'low'
        }

    def _analyze_sitemaps(self, base_url: str, domain: str) -> Dict[str, Any]:
        """Comprehensively analyze sitemaps including nested ones"""
        try:
            sitemap_urls = [
                f"https://{domain}/sitemap.xml",
                f"https://{domain}/sitemap_index.xml",
                f"https://{domain}/sitemaps.xml",
                f"https://{domain}/sitemap/sitemap.xml"
            ]
            
            total_urls = 0
            found_sitemaps = []
            
            for sitemap_url in sitemap_urls:
                try:
                    response = self.session.get(sitemap_url, timeout=10)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'xml')
                        
                        # Check if it's a sitemap index
                        sitemaps = soup.find_all('sitemap')
                        if sitemaps:
                            # This is a sitemap index, process each sitemap
                            for sitemap in sitemaps:
                                loc = sitemap.find('loc')
                                if loc:
                                    nested_count = self._count_sitemap_urls(loc.text)
                                    total_urls += nested_count
                                    found_sitemaps.append(loc.text)
                        else:
                            # This is a regular sitemap
                            urls = soup.find_all('url')
                            total_urls += len(urls)
                            found_sitemaps.append(sitemap_url)
                        
                        break  # Found working sitemap
                        
                except Exception:
                    continue
            
            if total_urls > 0:
                return {
                    'total_pages': total_urls,
                    'source': 'sitemap',
                    'details': f'Found {len(found_sitemaps)} sitemap(s) with {total_urls} total URLs',
                    'confidence': 'high'
                }
            
        except Exception as e:
            print(f"Error analyzing sitemaps: {e}")
        
        return {
            'total_pages': None,
            'source': 'error',
            'details': 'Unable to access or parse website sitemaps',
            'confidence': 'low'
        }

    def _count_sitemap_urls(self, sitemap_url: str) -> int:
        """Count URLs in a specific sitemap"""
        try:
            response = self.session.get(sitemap_url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'xml')
                urls = soup.find_all('url')
                return len(urls)
        except Exception:
            pass
        return 0

    def get_crawl_stats(self) -> Dict[str, Any]:
        """Get statistics about the crawled content"""
        if not self.scraped_content:
            return {}
        
        total_words = sum(page.get('word_count', 0) for page in self.scraped_content)
        avg_words = total_words / len(self.scraped_content)
        
        return {
            'total_pages': len(self.scraped_content),
            'total_words': total_words,
            'average_words_per_page': avg_words,
            'domain': urlparse(self.scraped_content[0]['url']).netloc,
            'scraped_at': datetime.now().isoformat()
        }