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
        self.estimated_total_pages = None
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
        
        all_discovered_links = set([start_url])
        
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
                    
                    # Track all discovered links for better estimation
                    all_discovered_links.update(new_links)
                    
                    # Add new links to the queue
                    for link in new_links:
                        if link not in self.visited_urls and link not in urls_to_visit:
                            urls_to_visit.append(link)
                            
                except Exception as e:
                    print(f"Error getting links from {current_url}: {e}")
            
            # Enhanced progress callback with current page info
            if progress_callback:
                page_title = content.get('title', 'Untitled') if content else 'Failed to load'
                progress_callback(
                    len(self.visited_urls), 
                    len(self.scraped_content),
                    current_url,
                    page_title
                )
            
            # Rate limiting
            time.sleep(self.delay)
        
        print(f"Crawling completed. Visited {len(self.visited_urls)} pages, extracted {len(self.scraped_content)} valid pages")
        return self.scraped_content
    
    def save_cache(self, filename: str = None) -> str:
        """Save scraped content to cache file"""
        if not filename:
            # Generate human-readable filename based on domain and timestamp
            if self.scraped_content:
                domain = urlparse(self.scraped_content[0]['url']).netloc.replace('www.', '')
                timestamp = datetime.now()
                
                # Create readable date format: "Dec-15-2025_2-28pm"
                date_str = timestamp.strftime("%b-%d-%Y")
                time_str = timestamp.strftime("%I-%M%p").lower()
                
                filename = f"{domain}_{date_str}_{time_str}_{len(self.scraped_content)}pages.json"
            else:
                timestamp = datetime.now()
                date_str = timestamp.strftime("%b-%d-%Y_%I-%M%p").lower()
                filename = f"crawl_{date_str}_0pages.json"
        
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
    
    def estimate_total_pages(self, start_url: str) -> dict:
        """Estimate total number of pages using authoritative sources"""
        import re
        try:
            from urllib.parse import urlparse, urljoin
            
            # Get the base domain
            parsed_url = urlparse(start_url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            domain = parsed_url.netloc
            
            # First, try to get comprehensive sitemap data
            sitemap_data = self._analyze_sitemaps(base_url, domain)
            if sitemap_data['found']:
                return {
                    'total_pages': sitemap_data['count'],
                    'source': 'sitemap',
                    'reliability': 'high',
                    'details': sitemap_data['details']
                }
            
            # Try third-party service estimation (using built-in-browser for security)
            # Note: In a production environment, you could integrate with services like:
            # - Screaming Frog API
            # - Ahrefs API  
            # - SEMrush API
            # But for this implementation, we'll stick to website-provided data
            
            return {
                'total_pages': None,
                'source': 'unavailable',
                'reliability': 'none',
                'details': 'Website does not provide sitemap or reliable page count information'
            }
            
        except Exception as e:
            return {
                'total_pages': None,
                'source': 'error',
                'reliability': 'none',
                'details': f'Error analyzing website structure: {str(e)}'
            }
    
    def _analyze_sitemaps(self, base_url: str, domain: str) -> dict:
        """Comprehensively analyze sitemaps including nested ones"""
        import re
        from urllib.parse import urljoin
        
        total_urls = set()
        sitemap_files_found = []
        
        # Common sitemap locations
        sitemap_locations = [
            '/sitemap.xml',
            '/sitemap_index.xml', 
            '/sitemaps.xml',
            '/sitemap/sitemap.xml',
            '/wp-sitemap.xml',  # WordPress
            '/sitemap.php'
        ]
        
        # Check robots.txt first for sitemap declarations
        try:
            robots_url = urljoin(base_url, '/robots.txt')
            response = self.session.get(robots_url, timeout=10)
            if response.status_code == 200:
                sitemap_refs = re.findall(r'sitemap:\s*(\S+)', response.text, re.IGNORECASE)
                sitemap_locations.extend(sitemap_refs)
        except:
            pass
        
        # Process each sitemap location
        for sitemap_location in set(sitemap_locations):
            try:
                if sitemap_location.startswith('http'):
                    sitemap_url = sitemap_location
                else:
                    sitemap_url = urljoin(base_url, sitemap_location)
                
                response = self.session.get(sitemap_url, timeout=15)
                if response.status_code == 200:
                    content = response.text
                    
                    # Check if it's a sitemap index (contains references to other sitemaps)
                    if '<sitemapindex' in content.lower() or '<sitemap>' in content:
                        nested_sitemaps = re.findall(r'<loc[^>]*>(.*?)</loc>', content, re.IGNORECASE | re.DOTALL)
                        sitemap_files_found.append({
                            'url': sitemap_url,
                            'type': 'index',
                            'nested_count': len(nested_sitemaps)
                        })
                        
                        # Process nested sitemaps
                        for nested_url in nested_sitemaps:
                            nested_url = nested_url.strip()
                            if domain in nested_url:  # Only process same-domain sitemaps
                                try:
                                    nested_response = self.session.get(nested_url, timeout=10)
                                    if nested_response.status_code == 200:
                                        nested_urls = re.findall(r'<loc[^>]*>(.*?)</loc>', nested_response.text, re.IGNORECASE | re.DOTALL)
                                        domain_urls = [url.strip() for url in nested_urls if domain in url]
                                        total_urls.update(domain_urls)
                                        sitemap_files_found.append({
                                            'url': nested_url,
                                            'type': 'nested',
                                            'url_count': len(domain_urls)
                                        })
                                except:
                                    continue
                    
                    else:
                        # Regular sitemap with URLs
                        loc_urls = re.findall(r'<loc[^>]*>(.*?)</loc>', content, re.IGNORECASE | re.DOTALL)
                        domain_urls = [url.strip() for url in loc_urls if domain in url]
                        if domain_urls:
                            total_urls.update(domain_urls)
                            sitemap_files_found.append({
                                'url': sitemap_url,
                                'type': 'regular',
                                'url_count': len(domain_urls)
                            })
            except:
                continue
        
        if total_urls:
            return {
                'found': True,
                'count': len(total_urls),
                'details': f"Found {len(sitemap_files_found)} sitemap file(s) with {len(total_urls)} unique URLs",
                'files': sitemap_files_found
            }
        
        return {'found': False, 'count': 0, 'details': 'No valid sitemaps found', 'files': []}

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