"""
Visualization components for the web content analyzer
"""
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_agraph import agraph, Node, Edge, Config
import streamlit as st


def create_content_visualization(content: List[Dict[str, Any]]) -> Tuple[Optional[go.Figure], Optional[go.Figure]]:
    """Create visualizations for crawled content"""
    if not content:
        return None, None
    
    try:
        # Create DataFrame
        df = pd.DataFrame([
            {
                'title': page.get('title', 'Untitled')[:50] + "..." if len(page.get('title', '')) > 50 else page.get('title', 'Untitled'),
                'word_count': page.get('word_count', 0),
                'url_depth': len(page.get('url', '').split('/')) - 3,  # Subtract protocol and domain
                'headings_count': len(page.get('headings', []))
            }
            for page in content
        ])
        
        # Word count distribution
        fig_words = px.histogram(
            df, 
            x='word_count', 
            nbins=20,
            title='Word Count Distribution',
            labels={'word_count': 'Words per Page', 'count': 'Number of Pages'},
            color_discrete_sequence=['#1f77b4']
        )
        fig_words.update_layout(
            xaxis_title="Words per Page",
            yaxis_title="Number of Pages",
            showlegend=False,
            height=400
        )
        
        # URL depth analysis
        if df['url_depth'].nunique() > 1:
            fig_depth = px.scatter(
                df, 
                x='url_depth', 
                y='word_count',
                title='Content Depth vs Word Count',
                labels={'url_depth': 'URL Depth Level', 'word_count': 'Word Count'},
                hover_data=['title'],
                color_discrete_sequence=['#ff7f0e']
            )
            fig_depth.update_layout(
                xaxis_title="URL Depth Level",
                yaxis_title="Word Count",
                showlegend=False,
                height=400
            )
        else:
            fig_depth = None
        
        return fig_words, fig_depth
        
    except Exception as e:
        st.error(f"Error creating visualizations: {e}")
        return None, None


def create_improved_network_graph(content: List[Dict[str, Any]]) -> Optional[bool]:
    """Create an improved network graph of content relationships"""
    if len(content) < 2:
        return None
    
    try:
        # Limit to 15 pages for better visualization
        limited_content = content[:15]
        
        nodes = []
        edges = []
        
        # Create nodes
        for i, page in enumerate(limited_content):
            title = page.get('title', f'Page {i+1}')
            word_count = page.get('word_count', 0)
            
            # Truncate long titles
            if len(title) > 30:
                title = title[:27] + "..."
            
            # Node size based on word count (normalized)
            max_words = max(p.get('word_count', 0) for p in limited_content)
            min_words = min(p.get('word_count', 0) for p in limited_content)
            
            if max_words > min_words:
                normalized_size = 20 + (word_count - min_words) / (max_words - min_words) * 30
            else:
                normalized_size = 35
            
            # Color based on URL depth
            url_depth = len(page.get('url', '').split('/')) - 3
            colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#dda0dd']
            color = colors[min(url_depth, len(colors) - 1)]
            
            nodes.append(Node(
                id=str(i),
                label=title,
                size=normalized_size,
                color=color,
                title=f"Words: {word_count}\nURL: {page.get('url', 'N/A')}"
            ))
        
        # Create edges based on content similarity (simplified)
        for i in range(len(limited_content)):
            for j in range(i + 1, len(limited_content)):
                # Simple similarity based on common words in titles
                title_i = limited_content[i].get('title', '').lower().split()
                title_j = limited_content[j].get('title', '').lower().split()
                
                common_words = set(title_i) & set(title_j)
                if len(common_words) > 1:  # At least 2 common words
                    similarity = len(common_words) / max(len(title_i), len(title_j))
                    
                    if similarity > 0.3:  # Only show stronger connections
                        edges.append(Edge(
                            source=str(i),
                            target=str(j),
                            width=similarity * 3,
                            color='#cccccc'
                        ))
        
        # Configuration
        config = Config(
            width=700,
            height=500,
            directed=False,
            physics=True,
            hierarchical=False,
            nodeHighlightBehavior=True,
            highlightColor="#F7A7A6",
            collapsible=False,
            node={'labelProperty': 'label'},
            link={'labelProperty': 'label', 'renderLabel': False}
        )
        
        # Display the graph
        agraph(nodes=nodes, edges=edges, config=config)
        return True
        
    except Exception as e:
        st.error(f"Error creating network graph: {e}")
        return None


def create_analytics_dashboard(content: List[Dict[str, Any]], stats: Dict[str, Any]) -> None:
    """Create comprehensive analytics dashboard"""
    if not content:
        st.info("No content available for analytics")
        return
    
    try:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Pages", stats.get('total_pages', 0))
        
        with col2:
            st.metric("Total Words", f"{stats.get('total_words', 0):,}")
        
        with col3:
            st.metric("Avg Words/Page", f"{stats.get('average_words_per_page', 0):.0f}")
        
        with col4:
            if stats.get('coverage_percentage'):
                coverage = stats['coverage_percentage']
                delta_color = "normal" if coverage >= 50 else "inverse"
                st.metric("Coverage", f"{coverage:.1f}%", delta_color=delta_color)
            else:
                st.metric("Coverage", "Unknown")
        
        # Detailed analytics
        st.subheader("Content Distribution")
        
        # Create content table
        content_df = pd.DataFrame([
            {
                'Title': page.get('title', 'Untitled')[:40] + "..." if len(page.get('title', '')) > 40 else page.get('title', 'Untitled'),
                'Words': page.get('word_count', 0),
                'Headings': len(page.get('headings', [])),
                'URL Depth': len(page.get('url', '').split('/')) - 3
            }
            for page in content
        ])
        
        st.dataframe(content_df, use_container_width=True, height=300)
        
    except Exception as e:
        st.error(f"Error creating analytics dashboard: {e}")


def display_source_analysis(sources: List[Dict[str, Any]]) -> None:
    """Display detailed source analysis"""
    if not sources:
        st.info("No sources available")
        return
    
    st.caption("Each source shows how well that page content matches your question")
    
    for i, source in enumerate(sources[:5], 1):
        similarity = source.get('similarity_score', 0)
        
        # Color code relevance
        if similarity >= 0.7:
            relevance_icon = "🟢"
            relevance_text = "Highly Relevant"
        elif similarity >= 0.5:
            relevance_icon = "🟡"
            relevance_text = "Moderately Relevant"
        else:
            relevance_icon = "🟠"
            relevance_text = "Somewhat Relevant"
        
        st.markdown(f"""
        **{i}. {source.get('title', 'Untitled')}**  
        🔗 [{source.get('url', 'No URL')}]({source.get('url', '#')})  
        {relevance_icon} {relevance_text} ({similarity:.1%})
        """)


def display_reliability_guide() -> None:
    """Display the reliability improvement guide"""
    with st.expander("💡 How to Improve Answer Reliability"):
        st.markdown("""
        **To get the most reliable answers from the Web Content Analyzer:**
        
        **1. Crawl More Pages (15-100 recommended)**
        - More content = better context for AI analysis
        - Diverse pages provide comprehensive coverage
        - Current crawl limit can be adjusted in the sidebar
        - Check the "Website Coverage" percentage in the sidebar - aim for 50%+ when possible
        
        **2. Ask Specific Questions**
        - Instead of: "Tell me about this site"
        - Try: "What are the main product features?" or "What pricing options are available?"
        
        **3. Use Keywords in Search**
        - For semantic search, use specific terms from the website
        - Multiple related keywords work better than single words
        
        **4. Check Source Quality**
        - Look at the sources shown with each answer
        - Higher similarity scores (70%+) indicate better matches
        - Multiple relevant sources increase confidence
        
        **5. Compare Question vs Search Results**
        - Use questions for analysis and summaries
        - Use semantic search to verify specific details
        
        **6. Website Content Quality Matters**
        - Well-structured sites with clear content work better
        - Sites with lots of text content provide richer analysis
        - Technical documentation and product pages are ideal
        
        **Current Quality Indicators:**
        - 🟢 Very Reliable (80%+): High confidence, multiple good sources
        - 🟡 Mostly Reliable (60-80%): Good sources, decent coverage
        - 🟠 Somewhat Reliable (40-60%): Limited relevant information
        - 🔴 Limited Reliability (<40%): Insufficient matching content
        """)


def display_confidence_explanation(confidence: float) -> None:
    """Display confidence score explanation"""
    # Convert confidence to reliability indicator
    if confidence >= 0.8:
        reliability = "🟢 Very Reliable"
        explanation = "High confidence with strong source matches"
    elif confidence >= 0.6:
        reliability = "🟡 Mostly Reliable"
        explanation = "Good confidence with decent source quality"
    elif confidence >= 0.4:
        reliability = "🟠 Somewhat Reliable"
        explanation = "Moderate confidence, limited relevant sources"
    else:
        reliability = "🔴 Limited Reliability"
        explanation = "Low confidence, insufficient matching content"
    
    st.markdown(f"**Reliability:** {reliability}")
    
    with st.expander("What does this confidence score mean?"):
        st.markdown(f"""
        **Current Score: {confidence:.1%}**
        
        {explanation}
        
        **This score is based on:**
        - How well the content matches your question
        - Number of relevant sources found
        - Content overlap and consistency
        
        Higher scores mean the answer is more reliable and well-supported by the website content.
        """)