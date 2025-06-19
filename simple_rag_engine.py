import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from openai import OpenAI
import tiktoken
import re
import json

class SimpleRAGEngine:
    """Simple RAG engine without ChromaDB dependencies"""
    
    def __init__(self):
        # Initialize OpenAI client
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.client = OpenAI(api_key=self.api_key)
        self.encoding = tiktoken.encoding_for_model("gpt-4o")
        
        # Simple in-memory storage
        self.chunks = []
        self.chunk_metadata = []
        self.content = []
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def _smart_chunk_text(self, text: str, max_tokens: int = 1000, overlap_tokens: int = 100) -> List[str]:
        """Intelligently chunk text based on semantic boundaries"""
        if not text or not text.strip():
            return []
        
        # First, try to split by double newlines (paragraphs)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph_tokens = self._count_tokens(paragraph)
            current_tokens = self._count_tokens(current_chunk)
            
            if current_tokens + paragraph_tokens <= max_tokens:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    
                    # Add overlap from current chunk
                    if overlap_tokens > 0:
                        overlap_text = self._get_overlap_text(current_chunk, overlap_tokens)
                        current_chunk = overlap_text + "\n\n" + paragraph
                    else:
                        current_chunk = paragraph
                else:
                    # Single paragraph is too long, split by sentences
                    sentences = re.split(r'[.!?]+', paragraph)
                    temp_chunk = ""
                    
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if not sentence:
                            continue
                            
                        sentence_tokens = self._count_tokens(sentence)
                        temp_tokens = self._count_tokens(temp_chunk)
                        
                        if temp_tokens + sentence_tokens <= max_tokens:
                            temp_chunk += ". " + sentence if temp_chunk else sentence
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                                temp_chunk = sentence
                            else:
                                # Single sentence is too long, truncate
                                chunks.append(sentence[:max_tokens*4])  # Rough estimate
                                temp_chunk = ""
                    
                    if temp_chunk:
                        current_chunk = temp_chunk
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def _get_overlap_text(self, text: str, overlap_tokens: int) -> str:
        """Get the last part of text for overlap"""
        words = text.split()
        if len(words) <= overlap_tokens // 4:  # Rough estimate: 4 chars per token
            return text
        
        overlap_words = words[-(overlap_tokens // 4):]
        return " ".join(overlap_words)
    
    def clear_collection(self):
        """Clear all stored content"""
        self.chunks = []
        self.chunk_metadata = []
        self.content = []
    
    def process_web_content(self, web_content: List[Dict[str, Any]], domain: str = None):
        """Process web content and store with smart chunking"""
        if not web_content:
            return
        
        # Clear existing content
        self.clear_collection()
        self.content = web_content
        
        chunk_id = 0
        
        # Process each page
        for page_idx, page in enumerate(web_content):
            if 'content' in page and page['content']:
                # Smart chunk the content
                chunks = self._smart_chunk_text(page['content'])
                
                for chunk_idx, chunk in enumerate(chunks):
                    self.chunks.append(chunk)
                    self.chunk_metadata.append({
                        'title': page.get('title', 'Untitled'),
                        'url': page.get('url', ''),
                        'domain': domain or 'unknown',
                        'chunk_index': chunk_idx,
                        'page_index': page_idx,
                        'chunk_id': chunk_id
                    })
                    chunk_id += 1
        
        import streamlit as st
        st.success(f"Processed {len(web_content)} pages into {len(self.chunks)} intelligent chunks for analysis")
    
    def _semantic_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Perform semantic search across chunks"""
        if not self.chunks:
            return []
        
        # Simple keyword-based relevance scoring
        query_words = set(query.lower().split())
        scored_chunks = []
        
        for i, chunk in enumerate(self.chunks):
            chunk_words = set(chunk.lower().split())
            # Calculate relevance score based on word overlap
            overlap = len(query_words.intersection(chunk_words))
            total_words = len(query_words.union(chunk_words))
            relevance = overlap / total_words if total_words > 0 else 0
            
            # Boost score if query terms appear close together
            if overlap > 0:
                chunk_lower = chunk.lower()
                for word in query_words:
                    if word in chunk_lower:
                        relevance += 0.1
            
            if relevance > 0:
                scored_chunks.append({
                    'chunk': chunk,
                    'metadata': self.chunk_metadata[i],
                    'similarity': relevance
                })
        
        # Sort by relevance and return top k
        scored_chunks.sort(key=lambda x: x['similarity'], reverse=True)
        return scored_chunks[:k]

    def analyze_content(self, question: str, verbosity: str = 'concise'):
        """Analyze content with semantic search and provide answers"""
        if not self.chunks:
            return {"answer": "No content available for analysis"}
        
        # Find most relevant chunks using semantic search
        relevant_chunks = self._semantic_search(question, k=5)
        
        if not relevant_chunks:
            # Fallback to first few chunks if no relevant ones found
            context_parts = []
            for i, chunk in enumerate(self.chunks[:3]):
                metadata = self.chunk_metadata[i]
                context_parts.append(f"Source: {metadata['title']}\nContent: {chunk[:800]}...")
        else:
            context_parts = []
            sources = []
            for item in relevant_chunks:
                chunk = item['chunk']
                metadata = item['metadata']
                context_parts.append(f"Source: {metadata['title']}\nContent: {chunk}")
                sources.append({
                    'title': metadata['title'],
                    'url': metadata['url'],
                    'relevance': f"{item['similarity']:.2f}"
                })
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Create prompt based on verbosity
        system_prompt = "You are a helpful assistant that analyzes website content and provides accurate, well-sourced answers based solely on the provided information. Always cite which sources you're drawing from."
        
        if verbosity == 'concise':
            user_prompt = f"Based on this website content, provide a brief, focused answer to: {question}\n\nContent:\n{context}"
            max_tokens = 400
        elif verbosity == 'comprehensive':
            user_prompt = f"Based on this website content, provide a detailed, thorough analysis for: {question}. Include specific examples and details from the sources.\n\nContent:\n{context}"
            max_tokens = 1200
        else:  # balanced
            user_prompt = f"Based on this website content, provide a balanced, informative answer to: {question}\n\nContent:\n{context}"
            max_tokens = 800
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.3
            )
            
            # Calculate confidence based on relevance of sources
            if relevant_chunks:
                avg_relevance = sum(item['similarity'] for item in relevant_chunks) / len(relevant_chunks)
                confidence = min(avg_relevance * 2, 1.0)  # Scale to 0-1
            else:
                confidence = 0.5  # Medium confidence for fallback
            
            return {
                "answer": response.choices[0].message.content,
                "sources": sources if relevant_chunks else [{"title": meta["title"], "url": meta["url"]} for meta in self.chunk_metadata[:3]],
                "confidence": confidence,
                "chunks_used": len(relevant_chunks) if relevant_chunks else 3
            }
            
        except Exception as e:
            return {"answer": f"Error analyzing content: {str(e)}", "sources": [], "confidence": 0.0}
    
    def suggest_questions(self):
        """Suggest relevant questions based on the content"""
        if not self.chunks:
            return []
        
        # Generic suggestions based on common content analysis
        return [
            "What are the main topics covered on this website?",
            "What products or services are offered?",
            "What are the key features mentioned?",
            "How can I contact this company?",
            "What are the pricing details?",
            "What are the terms and conditions?"
        ]
    
    def get_content_summary(self):
        """Get content summary"""
        return {
            "total_chunks": len(self.chunks),
            "total_pages": len(self.content),
            "avg_chunk_size": sum(len(chunk.split()) for chunk in self.chunks) / len(self.chunks) if self.chunks else 0
        }