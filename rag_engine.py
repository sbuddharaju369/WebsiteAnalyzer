import os
import json
import numpy as np
import faiss
from typing import List, Dict, Any, Optional
from openai import OpenAI

class RAGEngine:
    def __init__(self):
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.client = OpenAI(api_key=api_key)
        
        # Initialize FAISS index
        self.dimension = 1536  # OpenAI text-embedding-ada-002 dimension
        self.index = None
        self.documents = []
        self.embeddings = []
        
    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        if not text or len(text.strip()) < chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to end at a sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                sentence_ends = ['.', '!', '?', '\n']
                for i in range(end, max(start + chunk_size - 100, start), -1):
                    if text[i] in sentence_ends and i < len(text) - 1:
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            
            # Prevent infinite loop
            if start >= len(text):
                break
        
        return chunks
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI"""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {str(e)}")
            return [0.0] * self.dimension
    
    def process_documents(self, documents: List[Dict[str, Any]]):
        """Process documents and create embeddings"""
        print("Processing documents for RAG...")
        
        self.documents = []
        embeddings_list = []
        
        for doc in documents:
            try:
                # Create text content for embedding
                content_parts = []
                
                if doc.get('title'):
                    content_parts.append(f"Title: {doc['title']}")
                
                if doc.get('category'):
                    content_parts.append(f"Category: {doc['category']}")
                
                if doc.get('price'):
                    content_parts.append(f"Price: {doc['price']}")
                
                if doc.get('features'):
                    features_text = ', '.join(doc['features']) if isinstance(doc['features'], list) else str(doc['features'])
                    content_parts.append(f"Features: {features_text}")
                
                if doc.get('content'):
                    content_parts.append(f"Details: {doc['content']}")
                
                full_text = '\n'.join(content_parts)
                
                # Chunk the text
                chunks = self._chunk_text(full_text)
                
                for chunk in chunks:
                    if len(chunk.strip()) > 20:  # Only process meaningful chunks
                        # Get embedding
                        embedding = self._get_embedding(chunk)
                        
                        # Store document chunk with metadata
                        doc_chunk = {
                            'content': chunk,
                            'title': doc.get('title', ''),
                            'category': doc.get('category', ''),
                            'price': doc.get('price', ''),
                            'features': doc.get('features', []),
                            'url': doc.get('url', ''),
                            'full_content': doc.get('content', '')
                        }
                        
                        self.documents.append(doc_chunk)
                        embeddings_list.append(embedding)
                        
            except Exception as e:
                print(f"Error processing document: {str(e)}")
                continue
        
        if embeddings_list:
            # Create FAISS index
            embeddings_array = np.array(embeddings_list).astype('float32')
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings_array)
            self.index.add(embeddings_array)
            
            print(f"Created FAISS index with {len(self.documents)} document chunks")
        else:
            print("No documents were processed successfully")
    
    def _search_similar(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        if not self.index or not self.documents:
            return []
        
        try:
            # Get query embedding
            query_embedding = self._get_embedding(query)
            query_vector = np.array([query_embedding]).astype('float32')
            
            # Normalize for cosine similarity
            faiss.normalize_L2(query_vector)
            
            # Search
            scores, indices = self.index.search(query_vector, min(k, len(self.documents)))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents):
                    doc = self.documents[idx].copy()
                    doc['similarity_score'] = float(score)
                    results.append(doc)
            
            return results
            
        except Exception as e:
            print(f"Error searching similar documents: {str(e)}")
            return []
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system"""
        try:
            # Search for relevant documents
            relevant_docs = self._search_similar(question, k=5)
            
            if not relevant_docs:
                return {
                    'answer': "I don't have enough information to answer your question. Please try refreshing the data or asking a different question.",
                    'sources': []
                }
            
            # Prepare context from relevant documents
            context_parts = []
            sources = []
            
            for i, doc in enumerate(relevant_docs):
                # Add to context
                context_parts.append(f"Source {i+1}:")
                context_parts.append(f"Category: {doc.get('category', 'N/A')}")
                context_parts.append(f"Title: {doc.get('title', 'N/A')}")
                if doc.get('price'):
                    context_parts.append(f"Price: {doc['price']}")
                if doc.get('features'):
                    features = doc['features'] if isinstance(doc['features'], list) else [doc['features']]
                    context_parts.append(f"Features: {', '.join(features)}")
                context_parts.append(f"Details: {doc['content']}")
                context_parts.append("")  # Empty line for separation
                
                # Add to sources
                sources.append({
                    'title': doc.get('title', 'N/A'),
                    'category': doc.get('category', 'N/A'),
                    'content': doc['content'],
                    'price': doc.get('price', ''),
                    'features': doc.get('features', []),
                    'url': doc.get('url', ''),
                    'similarity_score': doc.get('similarity_score', 0)
                })
            
            context = '\n'.join(context_parts)
            
            # Create prompt for OpenAI
            system_prompt = """You are a helpful assistant that answers questions about Verizon plans and services. 
            Use the provided context to answer questions accurately and comprehensively. 
            If the context doesn't contain enough information to fully answer the question, say so.
            Always mention specific plan names, prices, and features when available.
            Format your response clearly with bullet points or numbered lists when appropriate."""
            
            user_prompt = f"""Context information about Verizon plans:
{context}

Question: {question}

Please provide a comprehensive answer based on the context above. Include specific details like plan names, prices, and features when available."""
            
            # Get response from OpenAI
            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            # do not change this unless explicitly requested by the user
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            answer = response.choices[0].message.content
            
            return {
                'answer': answer,
                'sources': sources
            }
            
        except Exception as e:
            return {
                'answer': f"Sorry, I encountered an error while processing your question: {str(e)}",
                'sources': []
            }
