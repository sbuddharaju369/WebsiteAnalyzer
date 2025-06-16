"""
Cache management utilities for handling cached content files
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from config.settings import CACHE_DIR


class CacheManager:
    """Manages cached content files"""
    
    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_files(self) -> List[Dict[str, Any]]:
        """Get list of available cache files with metadata"""
        cache_files = []
        
        for file_path in self.cache_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                cache_files.append({
                    'filename': file_path.name,
                    'path': str(file_path),
                    'domain': data.get('domain', 'Unknown'),
                    'total_pages': data.get('total_pages', 0),
                    'crawled_at': data.get('crawled_at', ''),
                    'file_size': file_path.stat().st_size
                })
            except Exception as e:
                print(f"Error reading cache file {file_path}: {e}")
        
        # Sort by creation time (newest first)
        cache_files.sort(key=lambda x: x['crawled_at'], reverse=True)
        return cache_files

    def load_cache(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load content from cache file"""
        try:
            file_path = self.cache_dir / filename
            
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            print(f"Error loading cache {filename}: {e}")
            return None

    def save_cache(self, content: List[Dict[str, Any]], domain: str, filename: str = None) -> str:
        """Save content to cache file"""
        if not content:
            return ""
        
        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now()
            date_str = timestamp.strftime("%b-%d-%Y")
            time_str = timestamp.strftime("%-I-%M%p").lower()
            page_count = len(content)
            filename = f"{domain}_{date_str}_{time_str}_{page_count}pages.json"
        
        file_path = self.cache_dir / filename
        
        # Prepare cache data
        cache_data = {
            'domain': domain,
            'total_pages': len(content),
            'crawled_at': datetime.now().isoformat(),
            'content': content
        }
        
        # Save to file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        
        return str(file_path)

    def format_cache_name(self, filename: str) -> str:
        """Format cache filename for display"""
        if not filename:
            return "Select cache file..."
        
        # Handle new readable format (domain_date_time_pages.json)
        if filename.endswith('pages.json'):
            parts = filename.replace('.json', '').split('_')
            if len(parts) >= 4:
                domain = parts[0]
                date_part = parts[1]  # e.g., "Dec-15-2025"
                time_part = parts[2]  # e.g., "2-28pm"
                pages_part = parts[3]  # e.g., "48pages"
                
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
                    return f"{domain} - {date_str}"
                except:
                    pass
        
        # Fallback to filename
        return filename

    def cleanup_old_caches(self, keep_count: int = 10):
        """Clean up old cache files, keeping only the most recent ones"""
        cache_files = self.get_cache_files()
        
        if len(cache_files) <= keep_count:
            return
        
        # Remove oldest files
        files_to_remove = cache_files[keep_count:]
        
        for file_info in files_to_remove:
            try:
                file_path = Path(file_info['path'])
                file_path.unlink()
                print(f"Removed old cache file: {file_info['filename']}")
            except Exception as e:
                print(f"Error removing cache file {file_info['filename']}: {e}")