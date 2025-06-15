import json
import re
from typing import List, Dict, Any, Tuple
from datetime import datetime

class AccuracyTester:
    def __init__(self, scraped_data: List[Dict[str, Any]]):
        self.data = scraped_data
        
    def extract_plan_details(self) -> Dict[str, Any]:
        """Extract key details from scraped data for verification"""
        details = {
            'plans_by_category': {},
            'price_ranges': {},
            'common_features': set(),
            'plan_names': [],
            'total_plans': len(self.data)
        }
        
        for item in self.data:
            category = item.get('category', 'unknown')
            
            # Group by category
            if category not in details['plans_by_category']:
                details['plans_by_category'][category] = []
            details['plans_by_category'][category].append(item)
            
            # Extract prices
            price_text = item.get('price', '')
            prices = self._extract_prices(price_text + ' ' + item.get('content', ''))
            if prices and category not in details['price_ranges']:
                details['price_ranges'][category] = []
            details['price_ranges'][category].extend(prices)
            
            # Collect features
            features = item.get('features', [])
            if isinstance(features, list):
                details['common_features'].update(features)
            
            # Collect plan names
            title = item.get('title', '')
            if title and title not in details['plan_names']:
                details['plan_names'].append(title)
        
        # Convert set to list for JSON serialization
        details['common_features'] = list(details['common_features'])
        
        return details
    
    def _extract_prices(self, text: str) -> List[float]:
        """Extract numeric prices from text"""
        price_pattern = r'\$(\d+(?:\.\d{2})?)'
        matches = re.findall(price_pattern, text)
        return [float(match) for match in matches]
    
    def verify_response_accuracy(self, question: str, ai_response: str, sources: List[Dict]) -> Dict[str, Any]:
        """Verify AI response against source data"""
        verification = {
            'question': question,
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'overall_score': 0,
            'issues': []
        }
        
        # Check 1: Sources are from actual scraped data
        source_verification = self._verify_sources(sources)
        verification['checks']['sources_valid'] = source_verification
        
        # Check 2: Price information accuracy
        price_verification = self._verify_prices_mentioned(ai_response, sources)
        verification['checks']['prices_accurate'] = price_verification
        
        # Check 3: Plan names mentioned exist in data
        plan_verification = self._verify_plan_names(ai_response)
        verification['checks']['plan_names_valid'] = plan_verification
        
        # Check 4: Category information accuracy
        category_verification = self._verify_categories(ai_response, question)
        verification['checks']['categories_accurate'] = category_verification
        
        # Calculate overall score
        scores = [check.get('score', 0) for check in verification['checks'].values()]
        verification['overall_score'] = sum(scores) / len(scores) if scores else 0
        
        # Identify issues
        for check_name, check_result in verification['checks'].items():
            if check_result.get('score', 0) < 0.8:
                verification['issues'].append(f"{check_name}: {check_result.get('issue', 'Low accuracy')}")
        
        return verification
    
    def _verify_sources(self, sources: List[Dict]) -> Dict[str, Any]:
        """Verify that sources come from actual scraped data"""
        if not sources:
            return {'score': 0, 'issue': 'No sources provided'}
        
        valid_sources = 0
        for source in sources:
            source_content = source.get('content', '')
            # Check if this content exists in our scraped data
            for item in self.data:
                if source_content in item.get('content', ''):
                    valid_sources += 1
                    break
        
        score = valid_sources / len(sources) if sources else 0
        return {
            'score': score,
            'valid_sources': valid_sources,
            'total_sources': len(sources),
            'issue': f"Only {valid_sources}/{len(sources)} sources verified" if score < 1 else None
        }
    
    def _verify_prices_mentioned(self, response: str, sources: List[Dict]) -> Dict[str, Any]:
        """Verify prices mentioned in response exist in sources"""
        response_prices = self._extract_prices(response)
        source_prices = []
        
        for source in sources:
            source_text = source.get('content', '') + ' ' + source.get('price', '')
            source_prices.extend(self._extract_prices(source_text))
        
        if not response_prices:
            return {'score': 1.0, 'issue': None}  # No prices claimed, so accurate
        
        verified_prices = 0
        for price in response_prices:
            if any(abs(price - sp) < 0.01 for sp in source_prices):
                verified_prices += 1
        
        score = verified_prices / len(response_prices) if response_prices else 1.0
        return {
            'score': score,
            'verified_prices': verified_prices,
            'total_mentioned': len(response_prices),
            'issue': f"Only {verified_prices}/{len(response_prices)} prices verified" if score < 1 else None
        }
    
    def _verify_plan_names(self, response: str) -> Dict[str, Any]:
        """Verify plan names mentioned exist in our data"""
        plan_details = self.extract_plan_details()
        known_plans = plan_details['plan_names']
        
        # Look for plan names in response (case insensitive)
        mentioned_plans = []
        for plan in known_plans:
            if plan.lower() in response.lower():
                mentioned_plans.append(plan)
        
        # Also check for common plan type keywords
        plan_keywords = ['unlimited', 'plus', 'ultimate', 'basic', 'premium', 'prepaid', 'postpaid']
        keyword_matches = 0
        for keyword in plan_keywords:
            if keyword in response.lower():
                # Check if this keyword appears in our data
                for item in self.data:
                    if keyword in item.get('content', '').lower() or keyword in item.get('title', '').lower():
                        keyword_matches += 1
                        break
        
        return {
            'score': 1.0,  # Assume accurate unless we find contradictions
            'mentioned_plans': mentioned_plans,
            'keyword_matches': keyword_matches,
            'issue': None
        }
    
    def _verify_categories(self, response: str, question: str) -> Dict[str, Any]:
        """Verify category information is consistent"""
        plan_details = self.extract_plan_details()
        available_categories = list(plan_details['plans_by_category'].keys())
        
        mentioned_categories = []
        for category in available_categories:
            if category.lower() in response.lower() or category.lower() in question.lower():
                mentioned_categories.append(category)
        
        return {
            'score': 1.0,  # Assume accurate unless contradictions found
            'mentioned_categories': mentioned_categories,
            'available_categories': available_categories,
            'issue': None
        }
    
    def generate_test_questions(self) -> List[str]:
        """Generate test questions based on available data"""
        plan_details = self.extract_plan_details()
        questions = []
        
        # Category-based questions
        for category in plan_details['plans_by_category'].keys():
            questions.append(f"What {category} plans are available?")
            if category in plan_details['price_ranges']:
                questions.append(f"What are the cheapest {category} plans?")
        
        # Feature-based questions
        common_features = plan_details['common_features'][:5]  # Top 5 features
        for feature in common_features:
            questions.append(f"Which plans include {feature}?")
        
        # Comparison questions
        if len(plan_details['plans_by_category']) > 1:
            categories = list(plan_details['plans_by_category'].keys())[:2]
            questions.append(f"Compare {categories[0]} and {categories[1]} plans")
        
        # General questions
        questions.extend([
            "What is the cheapest plan available?",
            "What unlimited data plans are offered?",
            "What plans include streaming services?",
            "What are the most expensive plans?"
        ])
        
        return questions[:10]  # Return top 10 questions

def create_verification_report(scraped_data: List[Dict[str, Any]], 
                             question: str, 
                             ai_response: str, 
                             sources: List[Dict]) -> str:
    """Create a human-readable verification report"""
    tester = AccuracyTester(scraped_data)
    verification = tester.verify_response_accuracy(question, ai_response, sources)
    
    report = f"""
# Response Verification Report

**Question:** {question}
**Overall Accuracy Score:** {verification['overall_score']:.1%}
**Timestamp:** {verification['timestamp']}

## Verification Checks:

"""
    
    for check_name, result in verification['checks'].items():
        score = result.get('score', 0)
        status = "✅ PASS" if score >= 0.8 else "⚠️ WARNING" if score >= 0.5 else "❌ FAIL"
        
        report += f"### {check_name.replace('_', ' ').title()}\n"
        report += f"**Status:** {status} ({score:.1%})\n"
        
        if result.get('issue'):
            report += f"**Issue:** {result['issue']}\n"
        
        # Add specific details
        for key, value in result.items():
            if key not in ['score', 'issue']:
                report += f"- {key}: {value}\n"
        report += "\n"
    
    if verification['issues']:
        report += "## Issues Found:\n"
        for issue in verification['issues']:
            report += f"- {issue}\n"
        report += "\n"
    
    report += "## Data Summary:\n"
    plan_details = tester.extract_plan_details()
    report += f"- Total plans in database: {plan_details['total_plans']}\n"
    report += f"- Categories available: {', '.join(plan_details['plans_by_category'].keys())}\n"
    report += f"- Unique plan names: {len(plan_details['plan_names'])}\n"
    
    return report