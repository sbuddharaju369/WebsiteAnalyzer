import requests
import time
import trafilatura
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import random
from typing import List, Dict, Any

class VerizonScraper:
    def __init__(self):
        self.base_url = "https://www.verizon.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Rate limiting settings
        self.min_delay = 1.0  # Minimum delay between requests
        self.max_delay = 3.0  # Maximum delay between requests
        
        # URLs to scrape
        self.urls = {
            'mobile': [
                'https://www.verizon.com/plans/',
                'https://www.verizon.com/plans/unlimited/',
                'https://www.verizon.com/plans/shared-data/'
            ],
            'internet': [
                'https://www.verizon.com/home/fios/',
                'https://www.verizon.com/home/internet/',
                'https://www.verizon.com/home/fios-internet/'
            ],
            'prepaid': [
                'https://www.verizon.com/prepaid/',
                'https://www.verizon.com/prepaid/plans/'
            ],
            'bundles': [
                'https://www.verizon.com/bundles/',
                'https://www.verizon.com/home/bundles/'
            ]
        }
    
    def _make_request(self, url: str) -> str:
        """Make a rate-limited request to a URL"""
        try:
            # Rate limiting
            delay = random.uniform(self.min_delay, self.max_delay)
            time.sleep(delay)
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.text
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {str(e)}")
            return ""
    
    def _extract_clean_text(self, url: str) -> str:
        """Extract clean text content from a URL using trafilatura"""
        try:
            # Use trafilatura for better text extraction
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                text = trafilatura.extract(downloaded)
                return text if text else ""
            return ""
        except Exception as e:
            print(f"Error extracting text from {url}: {str(e)}")
            return ""
    
    def _parse_plan_info(self, html: str, url: str, category: str) -> List[Dict[str, Any]]:
        """Parse plan information from HTML"""
        plans = []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Look for common plan selectors
            plan_selectors = [
                '.plan-card',
                '.plan-tile',
                '.plan-option',
                '[data-testid*="plan"]',
                '.product-card',
                '.offer-card'
            ]
            
            plan_elements = []
            for selector in plan_selectors:
                elements = soup.select(selector)
                if elements:
                    plan_elements = elements
                    break
            
            # If no specific plan elements found, try to extract general content
            if not plan_elements:
                # Get clean text using trafilatura
                clean_text = self._extract_clean_text(url)
                if clean_text and len(clean_text.strip()) > 100:
                    plans.append({
                        'title': soup.title.string if soup.title else f"{category.title()} Plans",
                        'content': clean_text,
                        'price': self._extract_prices_from_text(clean_text),
                        'features': self._extract_features_from_text(clean_text),
                        'url': url,
                        'category': category
                    })
                return plans
            
            # Parse individual plan elements
            for element in plan_elements[:10]:  # Limit to first 10 plans to avoid overwhelming
                try:
                    title = ""
                    price = ""
                    features = []
                    content = ""
                    
                    # Extract title
                    title_selectors = ['h1', 'h2', 'h3', '.plan-name', '.title', '.name']
                    for selector in title_selectors:
                        title_elem = element.select_one(selector)
                        if title_elem:
                            title = title_elem.get_text().strip()
                            break
                    
                    # Extract price
                    price_selectors = ['.price', '.cost', '[data-testid*="price"]', '.plan-price']
                    for selector in price_selectors:
                        price_elem = element.select_one(selector)
                        if price_elem:
                            price = price_elem.get_text().strip()
                            break
                    
                    # Extract features
                    feature_selectors = ['.features li', '.benefits li', '.plan-features li', 'ul li']
                    for selector in feature_selectors:
                        feature_elems = element.select(selector)
                        if feature_elems:
                            features = [elem.get_text().strip() for elem in feature_elems[:5]]
                            break
                    
                    # Get all text content
                    content = element.get_text().strip()
                    
                    if title or content:
                        plans.append({
                            'title': title or f"{category.title()} Plan",
                            'content': content,
                            'price': price,
                            'features': features,
                            'url': url,
                            'category': category
                        })
                        
                except Exception as e:
                    print(f"Error parsing plan element: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"Error parsing HTML from {url}: {str(e)}")
        
        return plans
    
    def _extract_prices_from_text(self, text: str) -> str:
        """Extract price information from text"""
        import re
        price_patterns = [
            r'\$\d+(?:\.\d{2})?(?:/mo|/month|per month)?',
            r'\$\d+(?:\.\d{2})?\s*(?:monthly|per month)',
            r'\d+\s*dollars?\s*(?:per month|monthly)'
        ]
        
        prices = []
        for pattern in price_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            prices.extend(matches)
        
        return ', '.join(prices[:3]) if prices else ""
    
    def _extract_features_from_text(self, text: str) -> List[str]:
        """Extract feature keywords from text"""
        feature_keywords = [
            'unlimited', '5G', 'data', 'hotspot', 'streaming', 'international',
            'calling', 'texting', 'roaming', 'premium', 'HD', '4K', 'music',
            'video', 'cloud', 'security', 'support', 'installation', 'equipment'
        ]
        
        features = []
        text_lower = text.lower()
        for keyword in feature_keywords:
            if keyword in text_lower:
                features.append(keyword.title())
        
        return features[:10]  # Limit to 10 features
    
    def scrape_mobile_plans(self) -> List[Dict[str, Any]]:
        """Scrape mobile plan information"""
        print("Scraping mobile plans...")
        plans = []
        
        for url in self.urls['mobile']:
            try:
                html = self._make_request(url)
                if html:
                    url_plans = self._parse_plan_info(html, url, 'mobile')
                    plans.extend(url_plans)
                    print(f"Found {len(url_plans)} mobile plans from {url}")
                    
            except Exception as e:
                print(f"Error scraping mobile plans from {url}: {str(e)}")
                continue
        
        return plans
    
    def scrape_internet_plans(self) -> List[Dict[str, Any]]:
        """Scrape internet plan information"""
        print("Scraping internet plans...")
        plans = []
        
        for url in self.urls['internet']:
            try:
                html = self._make_request(url)
                if html:
                    url_plans = self._parse_plan_info(html, url, 'internet')
                    plans.extend(url_plans)
                    print(f"Found {len(url_plans)} internet plans from {url}")
                    
            except Exception as e:
                print(f"Error scraping internet plans from {url}: {str(e)}")
                continue
        
        return plans
    
    def scrape_prepaid_plans(self) -> List[Dict[str, Any]]:
        """Scrape prepaid plan information"""
        print("Scraping prepaid plans...")
        plans = []
        
        for url in self.urls['prepaid']:
            try:
                html = self._make_request(url)
                if html:
                    url_plans = self._parse_plan_info(html, url, 'prepaid')
                    plans.extend(url_plans)
                    print(f"Found {len(url_plans)} prepaid plans from {url}")
                    
            except Exception as e:
                print(f"Error scraping prepaid plans from {url}: {str(e)}")
                continue
        
        return plans
    
    def scrape_bundle_plans(self) -> List[Dict[str, Any]]:
        """Scrape bundle plan information"""
        print("Scraping bundle plans...")
        plans = []
        
        for url in self.urls['bundles']:
            try:
                html = self._make_request(url)
                if html:
                    url_plans = self._parse_plan_info(html, url, 'bundles')
                    plans.extend(url_plans)
                    print(f"Found {len(url_plans)} bundle plans from {url}")
                    
            except Exception as e:
                print(f"Error scraping bundle plans from {url}: {str(e)}")
                continue
        
        return plans
