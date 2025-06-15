import os
import json
import psycopg2
from datetime import datetime
from typing import List, Dict, Any, Optional

class DatabaseManager:
    def __init__(self):
        self.connection = None
        self.connect()
        self.init_tables()
    
    def connect(self):
        """Connect to PostgreSQL database"""
        try:
            DATABASE_URL = os.getenv("DATABASE_URL")
            if not DATABASE_URL:
                raise ValueError("DATABASE_URL environment variable is required")
            
            self.connection = psycopg2.connect(DATABASE_URL)
            self.connection.autocommit = True
            return True
        except Exception as e:
            print(f"Database connection error: {e}")
            self.connection = None
            return False
    
    def init_tables(self):
        """Initialize database tables"""
        if not self.connection:
            return False
        
        try:
            cursor = self.connection.cursor()
            
            # Create plans table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS verizon_plans (
                    id SERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT,
                    price TEXT,
                    features JSONB,
                    url TEXT,
                    category TEXT NOT NULL,
                    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create query history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS query_history (
                    id SERIAL PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    sources_used JSONB,
                    accuracy_score FLOAT,
                    response_time_ms INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create scraping sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scraping_sessions (
                    id SERIAL PRIMARY KEY,
                    total_plans_scraped INTEGER NOT NULL,
                    categories_scraped JSONB,
                    scraping_duration_ms INTEGER,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.close()
            return True
            
        except Exception as e:
            print(f"Error creating tables: {e}")
            return False
    
    def save_scraped_plans(self, plans_data: List[Dict[str, Any]], session_notes: str = None) -> int:
        """Save scraped plans to database and return session ID"""
        if not self.connection:
            return -1
        
        try:
            cursor = self.connection.cursor()
            
            # Clear existing plans
            cursor.execute("DELETE FROM verizon_plans")
            
            # Insert new plans
            for plan in plans_data:
                features = plan.get('features', [])
                features_json = json.dumps(features) if isinstance(features, list) else json.dumps([])
                
                cursor.execute("""
                    INSERT INTO verizon_plans (title, content, price, features, url, category, scraped_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    plan.get('title', ''),
                    plan.get('content', ''),
                    plan.get('price', ''),
                    features_json,
                    plan.get('url', ''),
                    plan.get('category', 'unknown'),
                    datetime.utcnow()
                ))
            
            # Create scraping session record
            categories = list(set([plan.get('category', 'unknown') for plan in plans_data]))
            cursor.execute("""
                INSERT INTO scraping_sessions (total_plans_scraped, categories_scraped, notes, created_at)
                VALUES (%s, %s, %s, %s) RETURNING id
            """, (
                len(plans_data),
                json.dumps(categories),
                session_notes or '',
                datetime.utcnow()
            ))
            
            session_id = cursor.fetchone()[0]
            cursor.close()
            return session_id
            
        except Exception as e:
            print(f"Error saving plans: {e}")
            return -1
    
    def get_all_plans(self) -> List[Dict[str, Any]]:
        """Get all plans from database"""
        if not self.connection:
            return []
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT id, title, content, price, features, url, category, scraped_at
                FROM verizon_plans ORDER BY id
            """)
            
            plans = []
            for row in cursor.fetchall():
                plans.append({
                    'id': row[0],
                    'title': row[1],
                    'content': row[2],
                    'price': row[3],
                    'features': json.loads(row[4]) if row[4] else [],
                    'url': row[5],
                    'category': row[6],
                    'scraped_at': row[7].isoformat() if row[7] else None
                })
            
            cursor.close()
            return plans
            
        except Exception as e:
            print(f"Error getting plans: {e}")
            return []
    
    def get_plans_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get plans filtered by category"""
        if not self.connection:
            return []
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT id, title, content, price, features, url, category, scraped_at
                FROM verizon_plans WHERE category = %s ORDER BY id
            """, (category,))
            
            plans = []
            for row in cursor.fetchall():
                plans.append({
                    'id': row[0],
                    'title': row[1],
                    'content': row[2],
                    'price': row[3],
                    'features': json.loads(row[4]) if row[4] else [],
                    'url': row[5],
                    'category': row[6],
                    'scraped_at': row[7].isoformat() if row[7] else None
                })
            
            cursor.close()
            return plans
            
        except Exception as e:
            print(f"Error getting plans by category: {e}")
            return []
    
    def save_query(self, question: str, answer: str, sources: List[Dict], 
                   accuracy_score: float = None, response_time_ms: int = None) -> int:
        """Save query and response to database"""
        if not self.connection:
            return -1
        
        try:
            cursor = self.connection.cursor()
            
            # Prepare sources data
            sources_data = []
            for source in sources:
                sources_data.append({
                    'title': source.get('title', ''),
                    'category': source.get('category', ''),
                    'similarity_score': source.get('similarity_score', 0),
                    'url': source.get('url', '')
                })
            
            cursor.execute("""
                INSERT INTO query_history (question, answer, sources_used, accuracy_score, response_time_ms, created_at)
                VALUES (%s, %s, %s, %s, %s, %s) RETURNING id
            """, (
                question,
                answer,
                json.dumps(sources_data),
                accuracy_score,
                response_time_ms,
                datetime.utcnow()
            ))
            
            query_id = cursor.fetchone()[0]
            cursor.close()
            return query_id
            
        except Exception as e:
            print(f"Error saving query: {e}")
            return -1
    
    def get_query_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent query history"""
        if not self.connection:
            return []
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT id, question, answer, sources_used, accuracy_score, response_time_ms, created_at
                FROM query_history ORDER BY created_at DESC LIMIT %s
            """, (limit,))
            
            queries = []
            for row in cursor.fetchall():
                queries.append({
                    'id': row[0],
                    'question': row[1],
                    'answer': row[2],
                    'sources_used': json.loads(row[3]) if row[3] else [],
                    'accuracy_score': row[4],
                    'response_time_ms': row[5],
                    'created_at': row[6].isoformat() if row[6] else None
                })
            
            cursor.close()
            return queries
            
        except Exception as e:
            print(f"Error getting query history: {e}")
            return []
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        if not self.connection:
            return {}
        
        try:
            cursor = self.connection.cursor()
            
            # Get total counts
            cursor.execute("SELECT COUNT(*) FROM verizon_plans")
            total_plans = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM query_history")
            total_queries = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM scraping_sessions")
            total_sessions = cursor.fetchone()[0]
            
            # Get category distribution
            cursor.execute("SELECT category, COUNT(*) FROM verizon_plans GROUP BY category")
            category_counts = {}
            for row in cursor.fetchall():
                category_counts[row[0]] = row[1]
            
            # Get latest scraping session
            cursor.execute("""
                SELECT id, total_plans_scraped, created_at 
                FROM scraping_sessions ORDER BY created_at DESC LIMIT 1
            """)
            latest_session = cursor.fetchone()
            
            cursor.close()
            
            return {
                'total_plans': total_plans,
                'total_queries': total_queries,
                'total_scraping_sessions': total_sessions,
                'category_distribution': category_counts,
                'latest_scraping_session': {
                    'id': latest_session[0] if latest_session else None,
                    'date': latest_session[2].isoformat() if latest_session and latest_session[2] else None,
                    'plans_scraped': latest_session[1] if latest_session else 0
                }
            }
            
        except Exception as e:
            print(f"Error getting database stats: {e}")
            return {}
    
    def search_plans(self, search_term: str, category: str = "") -> List[Dict[str, Any]]:
        """Search plans by content or title"""
        if not self.connection:
            return []
        
        try:
            cursor = self.connection.cursor()
            
            search_filter = f"%{search_term.lower()}%"
            
            if category and category.strip():
                cursor.execute("""
                    SELECT id, title, content, price, features, url, category, scraped_at
                    FROM verizon_plans 
                    WHERE category = %s AND (LOWER(title) LIKE %s OR LOWER(content) LIKE %s)
                    ORDER BY id
                """, (category, search_filter, search_filter))
            else:
                cursor.execute("""
                    SELECT id, title, content, price, features, url, category, scraped_at
                    FROM verizon_plans 
                    WHERE LOWER(title) LIKE %s OR LOWER(content) LIKE %s
                    ORDER BY id
                """, (search_filter, search_filter))
            
            plans = []
            for row in cursor.fetchall():
                plans.append({
                    'id': row[0],
                    'title': row[1],
                    'content': row[2],
                    'price': row[3],
                    'features': json.loads(row[4]) if row[4] else [],
                    'url': row[5],
                    'category': row[6],
                    'scraped_at': row[7].isoformat() if row[7] else None
                })
            
            cursor.close()
            return plans
            
        except Exception as e:
            print(f"Error searching plans: {e}")
            return []
    
    def test_connection(self) -> bool:
        """Test database connection"""
        if not self.connection:
            return False
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()
            return result is not None
        except Exception as e:
            print(f"Database connection test failed: {e}")
            return False
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None