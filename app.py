import streamlit as st
import time
import os
from datetime import datetime, timedelta
from scraper import VerizonScraper
from chroma_engine_simple import ChromaRAGEngine
from utils import save_scraped_data, load_scraped_data
from testing_utils import AccuracyTester, create_verification_report
from database_simple import DatabaseManager

# Page configuration
st.set_page_config(
    page_title="Verizon Plan Assistant",
    page_icon="📱",
    layout="wide"
)

def initialize_session_state():
    """Initialize session state variables"""
    if 'rag_engine' not in st.session_state:
        st.session_state.rag_engine = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'last_scrape_time' not in st.session_state:
        st.session_state.last_scrape_time = None
    if 'scraped_data' not in st.session_state:
        st.session_state.scraped_data = []
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = None
    if 'use_database' not in st.session_state:
        st.session_state.use_database = False

def check_openai_key():
    """Check if OpenAI API key is available"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("🔑 OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        st.stop()
    return api_key

def scrape_verizon_data():
    """Scrape Verizon plan data"""
    scraper = VerizonScraper()
    
    with st.spinner("🕷️ Scraping Verizon plan information..."):
        try:
            # Get plan data from multiple categories
            mobile_plans = scraper.scrape_mobile_plans()
            internet_plans = scraper.scrape_internet_plans()
            prepaid_plans = scraper.scrape_prepaid_plans()
            bundle_plans = scraper.scrape_bundle_plans()
            
            # Combine all data
            all_data = mobile_plans + internet_plans + prepaid_plans + bundle_plans
            
            if all_data:
                # Save to JSON file (for backward compatibility)
                save_scraped_data(all_data)
                
                # Save to database if enabled
                if st.session_state.use_database and st.session_state.db_manager:
                    try:
                        session_id = st.session_state.db_manager.save_scraped_plans(
                            all_data, 
                            f"Scraped {len(all_data)} plans from {len(set([p.get('category') for p in all_data]))} categories"
                        )
                        st.success(f"✅ Data saved to database (Session ID: {session_id})")
                    except Exception as e:
                        st.warning(f"⚠️ Database save failed: {str(e)}")
                
                st.session_state.scraped_data = all_data
                st.session_state.last_scrape_time = datetime.now()
                return all_data
            else:
                st.error("❌ No data was scraped. Please check the scraping configuration.")
                return []
                
        except Exception as e:
            st.error(f"❌ Error during scraping: {str(e)}")
            return []

def load_existing_data():
    """Load existing scraped data if available"""
    try:
        data = load_scraped_data()
        if data:
            st.session_state.scraped_data = data
            # Try to get last modification time of the data file
            if os.path.exists("verizon_data.json"):
                mod_time = os.path.getmtime("verizon_data.json")
                st.session_state.last_scrape_time = datetime.fromtimestamp(mod_time)
            return data
        return []
    except Exception as e:
        st.warning(f"⚠️ Could not load existing data: {str(e)}")
        return []

def initialize_rag_engine(data):
    """Initialize the ChromaDB RAG engine with scraped data"""
    if not data:
        return None
        
    with st.spinner("🧠 Initializing AI engine with ChromaDB..."):
        try:
            rag_engine = ChromaRAGEngine()
            rag_engine.process_documents(data)
            return rag_engine
        except Exception as e:
            st.error(f"❌ Error initializing RAG engine: {str(e)}")
            return None

def main():
    # Initialize session state
    initialize_session_state()
    
    # Check for OpenAI API key
    check_openai_key()
    
    # Header
    st.title("📱 Verizon Plan Assistant")
    st.markdown("Ask questions about Verizon's mobile, internet, and bundle plans using AI-powered search.")
    
    # Sidebar for data management
    with st.sidebar:
        st.header("📊 Data Management")
        
        # Database configuration
        st.markdown("### 🗄️ Database Settings")
        use_database = st.checkbox("Use PostgreSQL Database", value=st.session_state.use_database)
        
        if use_database != st.session_state.use_database:
            st.session_state.use_database = use_database
            if use_database:
                try:
                    st.session_state.db_manager = DatabaseManager()
                    if st.session_state.db_manager.test_connection():
                        st.success("✅ Database connected!")
                    else:
                        st.error("❌ Database connection failed")
                        st.session_state.use_database = False
                except Exception as e:
                    st.error(f"❌ Database error: {str(e)}")
                    st.session_state.use_database = False
        
        # Show database stats if connected
        if st.session_state.use_database and st.session_state.db_manager:
            try:
                db_stats = st.session_state.db_manager.get_database_stats()
                if db_stats:
                    st.markdown("### 📈 Database Statistics")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Plans in DB", db_stats.get('total_plans', 0))
                        st.metric("Queries", db_stats.get('total_queries', 0))
                    with col2:
                        st.metric("Sessions", db_stats.get('total_scraping_sessions', 0))
                        if db_stats.get('latest_scraping_session', {}).get('date'):
                            st.metric("Last Scrape", 
                                    db_stats['latest_scraping_session']['date'][:10])
            except Exception as e:
                st.warning(f"⚠️ Database stats error: {str(e)}")
        
        # Show last scrape time
        if st.session_state.last_scrape_time:
            st.info(f"🕐 Last updated: {st.session_state.last_scrape_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Data refresh buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Refresh Data", help="Scrape fresh data from Verizon"):
                scraped_data = scrape_verizon_data()
                if scraped_data:
                    st.session_state.rag_engine = initialize_rag_engine(scraped_data)
                    st.session_state.data_loaded = True
                    st.success("✅ Data refreshed successfully!")
                    st.rerun()
        
        with col2:
            if st.button("📁 Load Data", help="Load data from JSON or database"):
                if st.session_state.use_database and st.session_state.db_manager:
                    # Load from database
                    try:
                        db_data = st.session_state.db_manager.get_all_plans()
                        if db_data:
                            st.session_state.scraped_data = db_data
                            st.session_state.rag_engine = initialize_rag_engine(db_data)
                            st.session_state.data_loaded = True
                            st.success("✅ Data loaded from database!")
                        else:
                            st.warning("⚠️ No data found in database")
                    except Exception as e:
                        st.error(f"❌ Database load error: {str(e)}")
                else:
                    # Load from JSON file
                    existing_data = load_existing_data()
                    if existing_data:
                        st.session_state.rag_engine = initialize_rag_engine(existing_data)
                        st.session_state.data_loaded = True
                        st.success("✅ Data loaded from file!")
                st.rerun()
        
        # Vector database search functionality
        if st.session_state.rag_engine:
            st.markdown("### 🔍 Vector Search (ChromaDB)")
            search_term = st.text_input("Semantic search:", placeholder="e.g., unlimited data plans with streaming")
            category_filter = st.selectbox("Filter by category:", 
                                         ["all", "mobile", "internet", "prepaid", "bundles"])
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔍 Vector Search"):
                    try:
                        filters = {"category": category_filter} if category_filter != "all" else {}
                        search_results = st.session_state.rag_engine.search_similar(search_term, filters, k=10)
                        st.session_state.vector_search_results = search_results
                        st.success(f"Found {len(search_results)} similar plans")
                    except Exception as e:
                        st.error(f"Vector search error: {str(e)}")
            
            with col2:
                if st.button("📊 ChromaDB Stats"):
                    try:
                        stats = st.session_state.rag_engine.get_collection_stats()
                        st.session_state.chroma_stats = stats
                        st.success("ChromaDB statistics updated")
                    except Exception as e:
                        st.error(f"Stats error: {str(e)}")
        
        # Traditional database search functionality
        if st.session_state.use_database and st.session_state.db_manager:
            st.markdown("### 🗄️ Database Search")
            db_search_term = st.text_input("Database search:", placeholder="e.g., unlimited, 5G", key="db_search")
            db_category_filter = st.selectbox("DB Filter by category:", 
                                         ["All", "mobile", "internet", "prepaid", "bundles"], key="db_category")
            
            if st.button("🔍 Search PostgreSQL"):
                try:
                    category_value = db_category_filter if db_category_filter != "All" else ""
                    search_results = st.session_state.db_manager.search_plans(
                        db_search_term, 
                        category_value
                    )
                    st.session_state.db_search_results = search_results
                    st.success(f"Found {len(search_results)} matching plans in PostgreSQL")
                except Exception as e:
                    st.error(f"Database search error: {str(e)}")
        
        # Show data statistics
        if st.session_state.scraped_data:
            st.markdown("### 📈 Current Data Statistics")
            total_plans = len(st.session_state.scraped_data)
            st.metric("Total Plans", total_plans)
            
            # Count by category
            categories = {}
            for item in st.session_state.scraped_data:
                cat = item.get('category', 'Unknown')
                categories[cat] = categories.get(cat, 0) + 1
            
            for category, count in categories.items():
                st.metric(f"{category} Plans", count)
        
        # Query history
        if st.session_state.use_database and st.session_state.db_manager:
            if st.button("📜 View Query History"):
                try:
                    history = st.session_state.db_manager.get_query_history(10)
                    st.session_state.query_history = history
                except Exception as e:
                    st.error(f"History error: {str(e)}")
    
    # Main content area
    if not st.session_state.data_loaded:
        st.info("👆 Please refresh or load data from the sidebar to get started.")
        
        # Try to load existing data automatically
        existing_data = load_existing_data()
        if existing_data:
            st.session_state.rag_engine = initialize_rag_engine(existing_data)
            st.session_state.data_loaded = True
            st.rerun()
    else:
        # Query interface
        st.header("💬 Ask Questions")
        
        # Sample questions
        with st.expander("💡 Sample Questions"):
            sample_questions = [
                "What are the cheapest mobile plans available?",
                "Compare unlimited data plans",
                "What internet speeds are available in my area?",
                "What bundles include both mobile and internet?",
                "What are the differences between prepaid and postpaid plans?",
                "What 5G plans does Verizon offer?",
                "What are the international calling options?",
                "What business plans are available?"
            ]
            
            for question in sample_questions:
                if st.button(f"❓ {question}", key=f"sample_{hash(question)}"):
                    st.session_state.user_question = question
        
        # User input
        user_question = st.text_input(
            "Ask your question:",
            placeholder="e.g., What are the best unlimited data plans?",
            key="question_input"
        )
        
        # Use sample question if set
        if hasattr(st.session_state, 'user_question'):
            user_question = st.session_state.user_question
            delattr(st.session_state, 'user_question')
        
        if user_question and st.session_state.rag_engine:
            with st.spinner("🤔 Thinking..."):
                try:
                    # Get AI response
                    start_time = datetime.now()
                    response = st.session_state.rag_engine.query(user_question)
                    response_time = int((datetime.now() - start_time).total_seconds() * 1000)
                    
                    # Save query to database if enabled
                    if st.session_state.use_database and st.session_state.db_manager:
                        try:
                            st.session_state.db_manager.save_query(
                                user_question,
                                response['answer'],
                                response.get('sources', []),
                                response_time_ms=response_time
                            )
                        except Exception as e:
                            st.warning(f"Failed to save query to database: {str(e)}")
                    
                    # Display response
                    st.markdown("### 🤖 AI Assistant Response")
                    st.markdown(response['answer'])
                    
                    # Accuracy verification section
                    st.markdown("### 🔍 Response Verification")
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        if st.button("🧪 Test Accuracy", help="Verify response against source data"):
                            with st.spinner("Verifying response accuracy..."):
                                verification_report = create_verification_report(
                                    st.session_state.scraped_data,
                                    user_question,
                                    response['answer'],
                                    response.get('sources', [])
                                )
                                st.session_state.verification_report = verification_report
                    
                    with col2:
                        if st.button("📋 Generate Test Questions", help="Get suggested test questions"):
                            tester = AccuracyTester(st.session_state.scraped_data)
                            test_questions = tester.generate_test_questions()
                            st.session_state.test_questions = test_questions
                    
                    # Display verification report if available
                    if hasattr(st.session_state, 'verification_report'):
                        with st.expander("📊 Accuracy Report", expanded=True):
                            st.markdown(st.session_state.verification_report)
                    
                    # Display test questions if available
                    if hasattr(st.session_state, 'test_questions'):
                        with st.expander("🎯 Suggested Test Questions"):
                            st.markdown("**Try these questions to test the system:**")
                            for i, q in enumerate(st.session_state.test_questions, 1):
                                if st.button(f"{i}. {q}", key=f"test_q_{i}"):
                                    st.session_state.user_question = q
                                    st.rerun()
                    
                    # Show sources if available
                    if response.get('sources'):
                        with st.expander("📚 Sources"):
                            for i, source in enumerate(response['sources'], 1):
                                st.markdown(f"**Source {i}:**")
                                st.markdown(f"- **Category:** {source.get('category', 'N/A')}")
                                st.markdown(f"- **Title:** {source.get('title', 'N/A')}")
                                if source.get('url'):
                                    st.markdown(f"- **URL:** {source['url']}")
                                st.markdown(f"- **Similarity Score:** {source.get('similarity_score', 0):.3f}")
                                st.markdown(f"- **Content:** {source.get('content', 'N/A')[:200]}...")
                                st.markdown("---")
                    
                except Exception as e:
                    st.error(f"❌ Error processing question: {str(e)}")
        
        elif user_question and not st.session_state.rag_engine:
            st.error("❌ RAG engine not initialized. Please refresh or load data first.")
    
    # Additional sections for database features
    col1, col2 = st.columns(2)
    
    # Display ChromaDB statistics if available
    if hasattr(st.session_state, 'chroma_stats') and st.session_state.chroma_stats:
        with col1:
            st.markdown("### 📊 ChromaDB Statistics")
            stats = st.session_state.chroma_stats
            st.metric("Total Chunks", stats.get('total_chunks', 0))
            st.metric("Queries Processed", stats.get('query_history_count', 0))
            
            if stats.get('categories'):
                st.markdown("**Categories Distribution:**")
                for category, count in stats['categories'].items():
                    st.write(f"- {category}: {count} chunks")
    
    # Display vector search results if available
    if hasattr(st.session_state, 'vector_search_results') and st.session_state.vector_search_results:
        with col2:
            st.markdown("### 🔍 Vector Search Results")
            for i, result in enumerate(st.session_state.vector_search_results[:5], 1):
                similarity = result.get('similarity_score', 0)
                with st.expander(f"{i}. {result.get('title', 'Untitled')} (Score: {similarity:.3f})"):
                    st.write(f"**Category:** {result.get('category', 'Unknown')}")
                    if result.get('price'):
                        st.write(f"**Price:** {result['price']}")
                    if result.get('features'):
                        features_str = ', '.join(result['features'][:3]) if isinstance(result['features'], list) else str(result['features'])
                        st.write(f"**Features:** {features_str}")
                    if result.get('content'):
                        st.write(f"**Content:** {result['content'][:200]}...")
                    st.write(f"**Similarity Score:** {similarity:.3f}")
    
    # Display PostgreSQL search results if available
    if hasattr(st.session_state, 'db_search_results') and st.session_state.db_search_results:
        st.markdown("### 🗄️ PostgreSQL Search Results")
        for i, result in enumerate(st.session_state.db_search_results[:5], 1):
            with st.expander(f"{i}. {result.get('title', 'Untitled')} ({result.get('category', 'Unknown')})"):
                if result.get('price'):
                    st.write(f"**Price:** {result['price']}")
                if result.get('features'):
                    st.write(f"**Features:** {', '.join(result['features'][:3])}")
                if result.get('content'):
                    st.write(f"**Content:** {result['content'][:300]}...")
                if result.get('url'):
                    st.write(f"**Source:** {result['url']}")
    
    # Display PostgreSQL query history if available
    if hasattr(st.session_state, 'query_history') and st.session_state.query_history:
        st.markdown("### 📜 PostgreSQL Query History")
        for i, query in enumerate(st.session_state.query_history[:3], 1):
            with st.expander(f"{i}. {query.get('question', 'Unknown')[:50]}..."):
                st.write(f"**Question:** {query.get('question', 'N/A')}")
                st.write(f"**Answer:** {query.get('answer', 'N/A')[:200]}...")
                if query.get('created_at'):
                    st.write(f"**Date:** {query['created_at'][:19]}")
                if query.get('response_time_ms'):
                    st.write(f"**Response Time:** {query['response_time_ms']}ms")
    
    # Display ChromaDB query history if available
    if st.session_state.rag_engine:
        chroma_history = st.session_state.rag_engine.get_query_history(5)
        if chroma_history:
            st.markdown("### 🧠 ChromaDB Query History")
            for i, query in enumerate(chroma_history, 1):
                with st.expander(f"{i}. {query.get('question', 'Unknown')[:50]}..."):
                    st.write(f"**Question:** {query.get('question', 'N/A')}")
                    st.write(f"**Answer:** {query.get('answer', 'N/A')[:200]}...")
                    st.write(f"**Sources Used:** {query.get('sources_count', 0)}")
                    if query.get('timestamp'):
                        st.write(f"**Date:** {query['timestamp'][:19]}")

if __name__ == "__main__":
    main()
