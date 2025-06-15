import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import JSON

Base = declarative_base()

class VerizonPlan(Base):
    __tablename__ = "verizon_plans"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=True)
    price = Column(String, nullable=True)
    features = Column(JSON, nullable=True)  # Store as JSON array
    url = Column(String, nullable=True)
    category = Column(String, nullable=False, index=True)
    scraped_at = Column(DateTime, default=datetime.utcnow, index=True)
    
class QueryHistory(Base):
    __tablename__ = "query_history"
    
    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    sources_used = Column(JSON, nullable=True)  # Store source IDs and similarity scores
    accuracy_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    response_time_ms = Column(Integer, nullable=True)

class ScrapingSession(Base):
    __tablename__ = "scraping_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    total_plans_scraped = Column(Integer, nullable=False)
    categories_scraped = Column(JSON, nullable=True)  # Store list of categories
    scraping_duration_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    notes = Column(Text, nullable=True)

def get_db_session():
    """Get database session"""
    if SessionLocal is None:
        raise ValueError("Database not initialized. Call initialize_database() first.")
    return SessionLocal()

class DatabaseManager:
    def __init__(self):
        self.engine = engine
        if not initialize_database():
            raise ValueError("Failed to initialize database")
    
    def save_scraped_plans(self, plans_data: List[Dict[str, Any]], session_notes: str = None) -> int:
        """Save scraped plans to database and return session ID"""
        db = get_db_session()
        try:
            # Clear existing data from this scraping session
            db.query(VerizonPlan).delete()
            
            # Save new plans
            plan_objects = []
            for plan in plans_data:
                plan_obj = VerizonPlan(
                    title=plan.get('title', ''),
                    content=plan.get('content', ''),
                    price=plan.get('price', ''),
                    features=plan.get('features', []),
                    url=plan.get('url', ''),
                    category=plan.get('category', 'unknown'),
                    scraped_at=datetime.utcnow()
                )
                plan_objects.append(plan_obj)
            
            db.add_all(plan_objects)
            
            # Create scraping session record
            categories = list(set([plan.get('category', 'unknown') for plan in plans_data]))
            session_obj = ScrapingSession(
                total_plans_scraped=len(plans_data),
                categories_scraped=categories,
                created_at=datetime.utcnow(),
                notes=session_notes
            )
            db.add(session_obj)
            
            db.commit()
            
            # Return session ID
            db.refresh(session_obj)
            return session_obj.id
            
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
    
    def get_all_plans(self) -> List[Dict[str, Any]]:
        """Get all plans from database"""
        db = get_db_session()
        try:
            plans = db.query(VerizonPlan).all()
            return [
                {
                    'id': plan.id,
                    'title': plan.title,
                    'content': plan.content,
                    'price': plan.price,
                    'features': plan.features,
                    'url': plan.url,
                    'category': plan.category,
                    'scraped_at': plan.scraped_at.isoformat() if plan.scraped_at else None
                }
                for plan in plans
            ]
        finally:
            db.close()
    
    def get_plans_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get plans filtered by category"""
        db = get_db_session()
        try:
            plans = db.query(VerizonPlan).filter(VerizonPlan.category == category).all()
            return [
                {
                    'id': plan.id,
                    'title': plan.title,
                    'content': plan.content,
                    'price': plan.price,
                    'features': plan.features,
                    'url': plan.url,
                    'category': plan.category,
                    'scraped_at': plan.scraped_at.isoformat() if plan.scraped_at else None
                }
                for plan in plans
            ]
        finally:
            db.close()
    
    def save_query(self, question: str, answer: str, sources: List[Dict], 
                   accuracy_score: float = None, response_time_ms: int = None) -> int:
        """Save query and response to database"""
        db = get_db_session()
        try:
            # Prepare sources data (store just the essential info)
            sources_data = []
            for source in sources:
                sources_data.append({
                    'title': source.get('title', ''),
                    'category': source.get('category', ''),
                    'similarity_score': source.get('similarity_score', 0),
                    'url': source.get('url', '')
                })
            
            query_obj = QueryHistory(
                question=question,
                answer=answer,
                sources_used=sources_data,
                accuracy_score=accuracy_score,
                response_time_ms=response_time_ms,
                created_at=datetime.utcnow()
            )
            
            db.add(query_obj)
            db.commit()
            
            db.refresh(query_obj)
            return query_obj.id
            
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
    
    def get_query_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent query history"""
        db = get_db_session()
        try:
            queries = db.query(QueryHistory).order_by(
                QueryHistory.created_at.desc()
            ).limit(limit).all()
            
            return [
                {
                    'id': query.id,
                    'question': query.question,
                    'answer': query.answer,
                    'sources_used': query.sources_used,
                    'accuracy_score': query.accuracy_score,
                    'response_time_ms': query.response_time_ms,
                    'created_at': query.created_at.isoformat() if query.created_at else None
                }
                for query in queries
            ]
        finally:
            db.close()
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        db = get_db_session()
        try:
            total_plans = db.query(VerizonPlan).count()
            total_queries = db.query(QueryHistory).count()
            total_sessions = db.query(ScrapingSession).count()
            
            # Get category distribution
            categories = db.query(VerizonPlan.category).distinct().all()
            category_counts = {}
            for category in categories:
                count = db.query(VerizonPlan).filter(
                    VerizonPlan.category == category[0]
                ).count()
                category_counts[category[0]] = count
            
            # Get latest scraping session
            latest_session = db.query(ScrapingSession).order_by(
                ScrapingSession.created_at.desc()
            ).first()
            
            return {
                'total_plans': total_plans,
                'total_queries': total_queries,
                'total_scraping_sessions': total_sessions,
                'category_distribution': category_counts,
                'latest_scraping_session': {
                    'id': latest_session.id if latest_session else None,
                    'date': latest_session.created_at.isoformat() if latest_session else None,
                    'plans_scraped': latest_session.total_plans_scraped if latest_session else 0
                }
            }
        finally:
            db.close()
    
    def search_plans(self, search_term: str, category: str = None) -> List[Dict[str, Any]]:
        """Search plans by content or title"""
        db = get_db_session()
        try:
            query = db.query(VerizonPlan)
            
            if category:
                query = query.filter(VerizonPlan.category == category)
            
            # Search in title and content
            search_filter = f"%{search_term.lower()}%"
            query = query.filter(
                (VerizonPlan.title.ilike(search_filter)) |
                (VerizonPlan.content.ilike(search_filter))
            )
            
            plans = query.all()
            return [
                {
                    'id': plan.id,
                    'title': plan.title,
                    'content': plan.content,
                    'price': plan.price,
                    'features': plan.features,
                    'url': plan.url,
                    'category': plan.category,
                    'scraped_at': plan.scraped_at.isoformat() if plan.scraped_at else None
                }
                for plan in plans
            ]
        finally:
            db.close()

# Test database connection
def test_connection():
    """Test database connection"""
    try:
        db = get_db_session()
        result = db.execute("SELECT 1").fetchone()
        db.close()
        return True
    except Exception as e:
        print(f"Database connection error: {e}")
        return False