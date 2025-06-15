import os
import chromadb
from datetime import datetime
from typing import List, Dict, Any, Optional
from openai import OpenAI

class ChromaRAGEngine:
    def __init__(self, collection_name: str = "verizon_plans"):
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.client = OpenAI(api_key=api_key)
        
        # Initialize ChromaDB with in-memory client for stability
        self.chroma_client = chromadb.EphemeralClient()
        
        # Create collection
        self.collection_name = collection_name
        self.collection = self._get_or_create_collection()
        
        # Store for query history
        self.query_history = []
        
    def _get_or_create_collection(self):
        """Get or create ChromaDB collection"""
        try:
            return self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "Verizon plan data for RAG"}
            )
        except Exception as e:
            print(f"Error creating collection: {e}")
            try:
                return self.chroma_client.get_collection(self.collection_name)
            except:
                raise ValueError(f"Cannot create or access collection: {e}")
    
    def _chunk_text(self, text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        if not text or len(text.strip()) < chunk_size:
            return [text.strip()] if text.strip() else []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # Try to end at a sentence boundary
            if end < len(text):
                sentence_ends = ['.', '!', '?', '\n']
                for i in range(end, max(start + chunk_size - 100, start), -1):
                    if i < len(text) and text[i] in sentence_ends:
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk and len(chunk) > 20:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    def clear_collection(self):
        """Clear all data from the collection"""
        try:
            self.chroma_client.delete_collection(self.collection_name)
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "Verizon plan data for RAG"}
            )
            print("Collection cleared successfully")
        except Exception as e:
            print(f"Error clearing collection: {str(e)}")
    
    def process_documents(self, documents: List[Dict[str, Any]]):
        """Process documents and store in ChromaDB"""
        print("Processing documents for ChromaDB...")
        
        # Clear existing data
        self.clear_collection()
        
        batch_size = 10  # Process in smaller batches
        
        for batch_start in range(0, len(documents), batch_size):
            batch_end = min(batch_start + batch_size, len(documents))
            batch = documents[batch_start:batch_end]
            
            doc_ids = []
            doc_texts = []
            doc_metadatas = []
            
            for doc_idx, doc in enumerate(batch, start=batch_start):
                try:
                    # Create simple text content
                    content_parts = []
                    
                    title = str(doc.get('title', '')).strip()
                    category = str(doc.get('category', '')).strip()
                    price = str(doc.get('price', '')).strip()
                    content = str(doc.get('content', '')).strip()
                    
                    if title:
                        content_parts.append(f"Title: {title}")
                    
                    if category:
                        content_parts.append(f"Category: {category}")
                    
                    if price:
                        content_parts.append(f"Price: {price}")
                    
                    if content:
                        content_parts.append(f"Details: {content}")
                    
                    full_text = '\n'.join(content_parts)
                    
                    if not full_text.strip():
                        continue
                    
                    # Simple chunking - just use the full text
                    doc_id = f"doc_{doc_idx}"
                    
                    # Simple metadata
                    metadata = {
                        'title': title[:100] if title else '',
                        'category': category[:50] if category else '',
                        'price': price[:50] if price else '',
                        'doc_index': str(doc_idx)
                    }
                    
                    doc_ids.append(doc_id)
                    doc_texts.append(full_text[:2000])  # Limit text length
                    doc_metadatas.append(metadata)
                    
                except Exception as e:
                    print(f"Error processing document {doc_idx}: {str(e)}")
                    continue
            
            # Add batch to ChromaDB
            if doc_ids:
                try:
                    self.collection.add(
                        ids=doc_ids,
                        documents=doc_texts,
                        metadatas=doc_metadatas
                    )
                    print(f"Added batch {batch_start//batch_size + 1}: {len(doc_ids)} documents")
                except Exception as e:
                    print(f"Error adding batch to ChromaDB: {str(e)}")
        
        print(f"Completed processing all documents")
    
    def search_similar(self, query: str, filters: Dict[str, Any] = None, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents using semantic similarity"""
        try:
            # Prepare where clause for filtering
            where_clause = {}
            if filters:
                for key, value in filters.items():
                    if value and value != "all" and value != "":
                        where_clause[key] = value
            
            # Search in ChromaDB
            search_kwargs = {
                "query_texts": [query],
                "n_results": min(k, 20),
                "include": ['documents', 'metadatas', 'distances']
            }
            
            if where_clause:
                search_kwargs["where"] = where_clause
            
            results = self.collection.query(**search_kwargs)
            
            # Format results
            formatted_results = []
            if results and results.get('ids') and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    try:
                        result = {
                            'id': results['ids'][0][i],
                            'content': results['documents'][0][i],
                            'metadata': results['metadatas'][0][i],
                            'similarity_score': 1.0 - results['distances'][0][i] if results.get('distances') else 0.5
                        }
                        formatted_results.append(result)
                    except Exception as e:
                        print(f"Error formatting result {i}: {str(e)}")
                        continue
            
            return formatted_results
            
        except Exception as e:
            print(f"Error searching documents: {str(e)}")
            return []
    
    def query(self, question: str, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Query the RAG system"""
        try:
            # Search for relevant documents
            relevant_docs = self.search_similar(question, filters, k=5)
            
            if not relevant_docs:
                return {
                    'answer': "I couldn't find relevant information to answer your question. Please try rephrasing or check if data has been loaded.",
                    'sources': []
                }
            
            # Prepare context for OpenAI
            context_parts = []
            sources = []
            
            for doc in relevant_docs:
                context_parts.append(doc['content'])
                sources.append({
                    'title': doc['metadata'].get('title', 'Unknown'),
                    'category': doc['metadata'].get('category', 'Unknown'),
                    'price': doc['metadata'].get('price', 'N/A'),
                    'similarity_score': doc['similarity_score']
                })
            
            context = '\n---\n'.join(context_parts)
            
            # Generate response using OpenAI
            system_prompt = """You are a helpful assistant that answers questions about Verizon telecommunications plans. 
            Use only the provided context to answer questions. Be specific and include plan names, prices, and features when available.
            If the context doesn't contain enough information, say so clearly."""
            
            user_prompt = f"""Context from Verizon plan information:
{context}

Question: {question}

Please provide a comprehensive answer based on the context above. Include specific details like plan names, prices, and features when available."""
            
            # Get response from OpenAI
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
            
            # Store query in history
            self.query_history.append({
                'question': question,
                'answer': answer,
                'timestamp': datetime.now().isoformat(),
                'sources_count': len(sources)
            })
            
            return {
                'answer': answer,
                'sources': sources
            }
            
        except Exception as e:
            return {
                'answer': f"Sorry, I encountered an error while processing your question: {str(e)}",
                'sources': []
            }
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the ChromaDB collection"""
        try:
            count = self.collection.count()
            return {
                'total_documents': count,
                'collection_name': self.collection_name,
                'last_updated': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'total_documents': 0,
                'collection_name': self.collection_name,
                'error': str(e)
            }
    
    def get_query_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent query history"""
        return self.query_history[-limit:] if self.query_history else []