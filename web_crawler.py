import requests
import trafilatura
import time
import json
import os
from urllib.parse import urljoin, urlparse, parse_qs
from typing import List, Dict, Set, Any
from bs4 import BeautifulSoup
import re
from datetime import datetime
import hashlib

class WebCrawler:
    def __init__(self, max_pages: int = 50, delay: float = 1.0):
        self.max_pages = max_pages
        self.delay = delay
        self.visited_urls = set()
        self.scraped_content = []
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
    def _is_valid_url(self, url: str, base_domain: str) -> bool:
        """Check if URL is valid for crawling"""
        try:
            parsed = urlparse(url)
            
            # Skip non-http(s) protocols
            if parsed.scheme not in ['http', 'https']:
                return False
                
            # Skip common file extensions
            excluded_extensions = {'.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', 
                                 '.zip', '.rar', '.tar', '.gz', '.mp3', '.mp4', '.avi', '.mov',
                                 '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.ico'}
            
            if any(parsed.path.lower().endswith(ext) for ext in excluded_extensions):
                return False
                
            # Stay within the same domain
            return parsed.netloc == base_domain
            
        except Exception:
            return False
    
    def _extract_links(self, html: str, base_url: str) -> Set[str]:
        """Extract all valid links from HTML"""
        links = set()
        try:
            soup = BeautifulSoup(html, 'html.parser')
            base_domain = urlparse(base_url).netloc
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(base_url, href)
                
                # Clean URL (remove fragments and some query parameters)
                parsed = urlparse(full_url)
                clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                
                if self._is_valid_url(clean_url, base_domain):
                    links.add(clean_url)
                    
        except Exception as e:
            print(f"Error extracting links: {e}")
            
        return links
    
    def _get_page_content(self, url: str) -> Dict[str, Any]:
        """Extract content from a single page"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # Extract clean text using trafilatura
            text_content = trafilatura.extract(response.text)
            if not text_content or len(text_content.strip()) < 100:
                return None
                
            # Extract metadata using BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            title = soup.find('title')
            title = title.get_text().strip() if title else url.split('/')[-1]
            
            description = soup.find('meta', attrs={'name': 'description'})
            description = description.get('content', '').strip() if description else ''
            
            keywords = soup.find('meta', attrs={'name': 'keywords'})
            keywords = keywords.get('content', '').strip() if keywords else ''
            
            # Extract headings
            headings = []
            for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                headings.append(h.get_text().strip())
            
            return {
                'url': url,
                'title': title,
                'description': description,
                'keywords': keywords,
                'content': text_content,
                'headings': headings,
                'word_count': len(text_content.split()),
                'scraped_at': datetime.now().isoformat(),
                'content_hash': hashlib.md5(text_content.encode()).hexdigest()
            }
            
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None
    
    def crawl_website(self, start_url: str, progress_callback=None) -> List[Dict[str, Any]]:
        """Crawl website starting from the given URL"""
        self.visited_urls.clear()
        self.scraped_content.clear()
        
        urls_to_visit = [start_url]
        base_domain = urlparse(start_url).netloc
        
        while urls_to_visit and len(self.visited_urls) < self.max_pages:
            current_url = urls_to_visit.pop(0)
            
            if current_url in self.visited_urls:
                continue
                
            print(f"Crawling: {current_url}")
            self.visited_urls.add(current_url)
            
            # Extract content
            content = self._get_page_content(current_url)
            if content:
                self.scraped_content.append(content)
                
                # Get the page HTML for link extraction
                try:
                    response = self.session.get(current_url, timeout=10)
                    new_links = self._extract_links(response.text, current_url)
                    
                    # Add new links to the queue
                    for link in new_links:
                        if link not in self.visited_urls and link not in urls_to_visit:
                            urls_to_visit.append(link)
                            
                except Exception as e:
                    print(f"Error getting links from {current_url}: {e}")
            
            # Progress callback
            if progress_callback:
                progress_callback(len(self.visited_urls), len(self.scraped_content))
            
            # Rate limiting
            time.sleep(self.delay)
        
        print(f"Crawling completed. Visited {len(self.visited_urls)} pages, extracted {len(self.scraped_content)} valid pages")
        return self.scraped_content
    
    def save_cache(self, filename: str = None) -> str:
        """Save scraped content to cache file"""
        if not filename:
            # Generate filename based on domain and timestamp
            if self.scraped_content:
                domain = urlparse(self.scraped_content[0]['url']).netloc
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"cache_{domain}_{timestamp}.json"
            else:
                filename = f"cache_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        cache_data = {
            'crawled_at': datetime.now().isoformat(),
            'total_pages': len(self.scraped_content),
            'pages_visited': len(self.visited_urls),
            'content': self.scraped_content
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        
        print(f"Cache saved to {filename}")
        return filename
    
    def load_cache(self, filename: str) -> List[Dict[str, Any]]:
        """Load content from cache file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            self.scraped_content = cache_data.get('content', [])
            print(f"Loaded {len(self.scraped_content)} pages from cache")
            return self.scraped_content
            
        except Exception as e:
            print(f"Error loading cache: {e}")
            return []
    
    def estimate_total_pages(self, start_url: str) -> int:
        """Estimate total number of pages on the website"""
        import re
        try:
            from urllib.parse import urlparse, urljoin
            
            # Get the base domain
            parsed_url = urlparse(start_url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            
            # Check for sitemap first
            sitemap_urls = ['/sitemap.xml', '/sitemap_index.xml', '/robots.txt']
            
            for sitemap_path in sitemap_urls:
                try:
                    sitemap_url = urljoin(base_url, sitemap_path)
                    response = self.session.get(sitemap_url, timeout=5)
                    
                    if response.status_code == 200:
                        content = response.text.lower()
                        
                        if 'sitemap' in sitemap_path and 'xml' in content:
                            # Count URLs in sitemap
                            url_count = len(re.findall(r'<(?:url|loc)>', content, re.IGNORECASE))
                            if url_count > 5:  # Valid sitemap with URLs
                                return min(url_count, 1000)  # Cap at reasonable limit
                        
                        elif 'robots.txt' in sitemap_path:
                            # Look for sitemap references in robots.txt
                            sitemap_refs = re.findall(r'sitemap:\s*(\S+)', content, re.IGNORECASE)
                            if sitemap_refs:
                                # Try to fetch the referenced sitemap
                                try:
                                    sitemap_response = self.session.get(sitemap_refs[0], timeout=5)
                                    if sitemap_response.status_code == 200:
                                        url_count = len(re.findall(r'<(?:url|loc)>', sitemap_response.text, re.IGNORECASE))
                                        if url_count > 5:
                                            return min(url_count, 1000)
                                except:
                                    pass
                except:
                    continue
            
            # Fallback: estimate based on discovered links during initial crawl
            # This will be updated during crawling
            return 50  # Conservative default estimate
            
        except Exception:
            return 50  # Default conservative estimate

    def get_crawl_stats(self) -> Dict[str, Any]:
        """Get statistics about the crawled content"""
        if not self.scraped_content:
            return {}
        
        total_words = sum(page.get('word_count', 0) for page in self.scraped_content)
        avg_words = total_words / len(self.scraped_content) if self.scraped_content else 0
        
        stats = {
            'total_pages': len(self.scraped_content),
            'total_words': total_words,
            'average_words_per_page': avg_words,
            'unique_titles': len(set(page.get('title', '') for page in self.scraped_content)),
            'pages_with_descriptions': len([p for p in self.scraped_content if p.get('description')])
        }
        
        # Add coverage percentage if available
        if hasattr(self, 'estimated_total_pages') and self.estimated_total_pages:
            coverage_percentage = (len(self.scraped_content) / self.estimated_total_pages) * 100
            stats['coverage_percentage'] = min(coverage_percentage, 100)
            stats['estimated_total_pages'] = self.estimated_total_pages
        
        return stats