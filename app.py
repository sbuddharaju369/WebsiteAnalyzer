import streamlit as st
import time
import os
from datetime import datetime, timedelta
from scraper import VerizonScraper
from rag_engine import RAGEngine
from utils import save_scraped_data, load_scraped_data
from testing_utils import AccuracyTester, create_verification_report
from database import DatabaseManager, test_connection

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
                # Save scraped data
                save_scraped_data(all_data)
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
    """Initialize the RAG engine with scraped data"""
    if not data:
        return None
        
    with st.spinner("🧠 Initializing AI engine..."):
        try:
            rag_engine = RAGEngine()
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
            if st.button("📁 Load Saved", help="Load previously scraped data"):
                existing_data = load_existing_data()
                if existing_data:
                    st.session_state.rag_engine = initialize_rag_engine(existing_data)
                    st.session_state.data_loaded = True
                    st.success("✅ Data loaded successfully!")
                    st.rerun()
        
        # Show data statistics
        if st.session_state.scraped_data:
            st.markdown("### 📈 Data Statistics")
            total_plans = len(st.session_state.scraped_data)
            st.metric("Total Plans", total_plans)
            
            # Count by category
            categories = {}
            for item in st.session_state.scraped_data:
                cat = item.get('category', 'Unknown')
                categories[cat] = categories.get(cat, 0) + 1
            
            for category, count in categories.items():
                st.metric(f"{category} Plans", count)
    
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
                    response = st.session_state.rag_engine.query(user_question)
                    
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

if __name__ == "__main__":
    main()
