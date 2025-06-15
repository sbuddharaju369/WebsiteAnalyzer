import json
import os
from typing import List, Dict, Any
from datetime import datetime

def save_scraped_data(data: List[Dict[str, Any]], filename: str = "verizon_data.json"):
    """Save scraped data to a JSON file"""
    try:
        # Add timestamp to data
        output_data = {
            'scraped_at': datetime.now().isoformat(),
            'total_items': len(data),
            'data': data
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(data)} items to {filename}")
        return True
        
    except Exception as e:
        print(f"Error saving data to {filename}: {str(e)}")
        return False

def load_scraped_data(filename: str = "verizon_data.json") -> List[Dict[str, Any]]:
    """Load scraped data from a JSON file"""
    try:
        if not os.path.exists(filename):
            return []
        
        with open(filename, 'r', encoding='utf-8') as f:
            file_data = json.load(f)
        
        # Handle both old format (list) and new format (dict with metadata)
        if isinstance(file_data, list):
            data = file_data
        elif isinstance(file_data, dict) and 'data' in file_data:
            data = file_data['data']
        else:
            return []
        
        print(f"Loaded {len(data)} items from {filename}")
        return data
        
    except Exception as e:
        print(f"Error loading data from {filename}: {str(e)}")
        return []

def clean_text(text: str) -> str:
    """Clean and normalize text content"""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Remove common HTML entities
    replacements = {
        '&nbsp;': ' ',
        '&amp;': '&',
        '&lt;': '<',
        '&gt;': '>',
        '&quot;': '"',
        '&#39;': "'",
        '&hellip;': '...',
        '\xa0': ' '  # Non-breaking space
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text.strip()

def extract_numbers(text: str) -> List[str]:
    """Extract numbers from text (useful for prices, data amounts, etc.)"""
    import re
    
    # Pattern to match numbers with optional decimal points and common suffixes
    pattern = r'\b\d+(?:\.\d+)?(?:\s*(?:GB|MB|TB|minutes?|mins?|texts?|dollars?|\$|USD))?\b'
    matches = re.findall(pattern, text, re.IGNORECASE)
    
    return matches

def format_plan_data(plan: Dict[str, Any]) -> str:
    """Format plan data for display"""
    output = []
    
    if plan.get('title'):
        output.append(f"**{plan['title']}**")
    
    if plan.get('category'):
        output.append(f"*Category: {plan['category'].title()}*")
    
    if plan.get('price'):
        output.append(f"💰 Price: {plan['price']}")
    
    if plan.get('features'):
        features = plan['features'] if isinstance(plan['features'], list) else [plan['features']]
        if features:
            output.append("✨ Features:")
            for feature in features[:5]:  # Limit to 5 features
                output.append(f"  • {feature}")
    
    if plan.get('content'):
        content = plan['content'][:300]  # Limit content length
        if len(plan['content']) > 300:
            content += "..."
        output.append(f"📝 Details: {content}")
    
    if plan.get('url'):
        output.append(f"🔗 Source: {plan['url']}")
    
    return '\n'.join(output)

def get_data_statistics(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Get statistics about the scraped data"""
    if not data:
        return {}
    
    stats = {
        'total_items': len(data),
        'categories': {},
        'has_price': 0,
        'has_features': 0,
        'avg_content_length': 0
    }
    
    content_lengths = []
    
    for item in data:
        # Count categories
        category = item.get('category', 'unknown')
        stats['categories'][category] = stats['categories'].get(category, 0) + 1
        
        # Count items with price
        if item.get('price'):
            stats['has_price'] += 1
        
        # Count items with features
        if item.get('features'):
            stats['has_features'] += 1
        
        # Track content length
        content = item.get('content', '')
        if content:
            content_lengths.append(len(content))
    
    if content_lengths:
        stats['avg_content_length'] = sum(content_lengths) / len(content_lengths)
    
    return stats
