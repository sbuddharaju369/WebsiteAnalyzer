import os
import json
import chromadb
from chromadb.config import Settings
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
        
        # Initialize ChromaDB with simpler settings
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Create or get collection
        self.collection_name = collection_name
        self.collection = self._get_or_create_collection()
        
        # Store for query history
        self.query_history = []
        
    def _get_or_create_collection(self):
        """Get or create ChromaDB collection"""
        try:
            # Try to delete existing collection and create fresh
            try:
                self.chroma_client.delete_collection(self.collection_name)
            except:
                pass  # Collection might not exist
            
            return self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "Verizon plan data for RAG"}
            )
        except Exception as e:
            print(f"Error creating collection: {e}")
            # If creation fails, try to get existing
            try:
                return self.chroma_client.get_collection(self.collection_name)
            except:
                raise ValueError(f"Cannot create or access collection: {e}")
    
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
            return []
    
    def _chunk_text(self, text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        if not text or len(text.strip()) < chunk_size:
            return [text] if text.strip() else []
        
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
        
        doc_ids = []
        doc_texts = []
        doc_metadatas = []
        
        for doc_idx, doc in enumerate(documents):
            try:
                # Create text content for embedding
                content_parts = []
                
                title = doc.get('title', '').strip()
                category = doc.get('category', '').strip()
                price = doc.get('price', '').strip()
                features = doc.get('features', [])
                content = doc.get('content', '').strip()
                url = doc.get('url', '').strip()
                
                if title:
                    content_parts.append(f"Title: {title}")
                
                if category:
                    content_parts.append(f"Category: {category}")
                
                if price:
                    content_parts.append(f"Price: {price}")
                
                if features:
                    if isinstance(features, list):
                        features_text = ', '.join(str(f) for f in features)
                    else:
                        features_text = str(features)
                    content_parts.append(f"Features: {features_text}")
                
                if content:
                    content_parts.append(f"Details: {content}")
                
                full_text = '\n'.join(content_parts)
                
                if not full_text.strip():
                    continue
                
                # Chunk the text
                chunks = self._chunk_text(full_text)
                
                for chunk_idx, chunk in enumerate(chunks):
                    if len(chunk.strip()) > 20:
                        # Create unique ID
                        doc_id = f"doc_{doc_idx}_chunk_{chunk_idx}"
                        
                        # Prepare simple metadata (avoid complex JSON)
                        metadata = {
                            'title': title,
                            'category': category,
                            'price': price,
                            'url': url,
                            'chunk_index': str(chunk_idx),
                            'doc_index': str(doc_idx)
                        }
                        
                        # Add features as separate fields to avoid JSON issues
                        if isinstance(features, list) and features:
                            for i, feature in enumerate(features[:5]):  # Limit to 5 features
                                metadata[f'feature_{i}'] = str(feature)
                        
                        doc_ids.append(doc_id)
                        doc_texts.append(chunk)
                        doc_metadatas.append(metadata)
                        
            except Exception as e:
                print(f"Error processing document {doc_idx}: {str(e)}")
                continue
        
        if doc_ids:
            try:
                # Add to ChromaDB without custom embeddings (let ChromaDB handle it)
                self.collection.add(
                    ids=doc_ids,
                    documents=doc_texts,
                    metadatas=doc_metadatas
                )
                print(f"Successfully added {len(doc_ids)} document chunks to ChromaDB")
            except Exception as e:
                print(f"Error adding documents to ChromaDB: {str(e)}")
        else:
            print("No documents were processed successfully")
    
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
                "n_results": min(k, 50),
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
                        # Convert distance to similarity score
                        distance = results['distances'][0][i] if results.get('distances') else 1.0
                        similarity_score = max(0, 1.0 - distance)  # Simple conversion
                        
                        metadata = results['metadatas'][0][i] if results.get('metadatas') else {}
                        
                        # Reconstruct features from metadata
                        features = []
                        for j in range(5):
                            feature_key = f'feature_{j}'
                            if feature_key in metadata:
                                features.append(metadata[feature_key])
                        
                        result = {
                            'content': results['documents'][0][i] if results.get('documents') else '',
                            'title': metadata.get('title', ''),
                            'category': metadata.get('category', ''),
                            'price': metadata.get('price', ''),
                            'features': features,
                            'url': metadata.get('url', ''),
                            'similarity_score': similarity_score,
                            'distance': distance
                        }
                        formatted_results.append(result)
                    except Exception as e:
                        print(f"Error formatting result {i}: {e}")
                        continue
            
            return formatted_results
            
        except Exception as e:
            print(f"Error searching similar documents: {str(e)}")
            return []
    
    def query(self, question: str, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Query the RAG system"""
        try:
            # Search for relevant documents
            relevant_docs = self.search_similar(question, filters, k=5)
            
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
                    context_parts.append(f"Features: {', '.join(doc['features'])}")
                context_parts.append(f"Details: {doc['content']}")
                context_parts.append("")
                
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
            query_record = {
                'question': question,
                'answer': answer,
                'sources_count': len(sources),
                'timestamp': datetime.now().isoformat()
            }
            self.query_history.append(query_record)
            
            return {
                'answer': answer,
                'sources': sources
            }
            
        except Exception as e:
            print(f"Error in query: {e}")
            return {
                'answer': f"Sorry, I encountered an error while processing your question: {str(e)}",
                'sources': []
            }
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the ChromaDB collection"""
        try:
            count = self.collection.count()
            
            # Get sample of metadata to analyze categories
            sample_results = self.collection.get(limit=min(count, 100))
            categories = {}
            
            if sample_results and sample_results.get('metadatas'):
                for metadata in sample_results['metadatas']:
                    category = metadata.get('category', 'unknown')
                    categories[category] = categories.get(category, 0) + 1
            
            return {
                'total_chunks': count,
                'categories': categories,
                'query_history_count': len(self.query_history)
            }
            
        except Exception as e:
            print(f"Error getting collection stats: {str(e)}")
            return {}
    
    def get_query_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent query history"""
        return self.query_history[-limit:] if self.query_history else []