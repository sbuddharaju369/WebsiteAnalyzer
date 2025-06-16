"""
RAG (Retrieval-Augmented Generation) engine module
Handles content processing, embeddings, and AI-powered question answering
"""
import json
import os
import time
from typing import Dict, List, Any, Optional
import tiktoken
import chromadb
from chromadb.config import Settings
import openai
from config.settings import RAG_SETTINGS, CHROMA_DIR, OPENAI_API_KEY


class WebRAGEngine:
    """RAG engine for processing web content and answering questions"""
    
    def __init__(self, collection_name: str = "web_content"):
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Validate API key
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        
        openai.api_key = OPENAI_API_KEY
        self._get_or_create_collection()

    def _get_or_create_collection(self):
        """Get or create ChromaDB collection"""
        try:
            # Ensure chroma directory exists
            CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=str(CHROMA_DIR),
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
            except Exception:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ChromaDB: {e}")

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))

    def _smart_chunk_text(self, text: str, max_tokens: int = None, overlap_tokens: int = None) -> List[str]:
        """Intelligently chunk text based on semantic boundaries"""
        max_tokens = max_tokens or RAG_SETTINGS["chunk_size"]
        overlap_tokens = overlap_tokens or RAG_SETTINGS["overlap"]
        
        # Split by paragraphs first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph_tokens = self._count_tokens(paragraph)
            current_tokens = self._count_tokens(current_chunk)
            
            # If paragraph is too large, split it by sentences
            if paragraph_tokens > max_tokens:
                sentences = [s.strip() for s in paragraph.split('.') if s.strip()]
                
                for sentence in sentences:
                    sentence = sentence + '.'
                    sentence_tokens = self._count_tokens(sentence)
                    
                    if current_tokens + sentence_tokens <= max_tokens:
                        current_chunk += " " + sentence if current_chunk else sentence
                        current_tokens = self._count_tokens(current_chunk)
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence
                        current_tokens = sentence_tokens
            else:
                # Check if we can add this paragraph to current chunk
                if current_tokens + paragraph_tokens <= max_tokens:
                    current_chunk += "\n\n" + paragraph if current_chunk else paragraph
                else:
                    # Save current chunk and start new one
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = paragraph
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Add overlap between chunks
        if len(chunks) > 1 and overlap_tokens > 0:
            overlapped_chunks = []
            
            for i, chunk in enumerate(chunks):
                if i == 0:
                    overlapped_chunks.append(chunk)
                else:
                    # Get overlap from previous chunk
                    prev_chunk = chunks[i-1]
                    overlap_text = self._get_overlap_text(prev_chunk, overlap_tokens)
                    
                    if overlap_text:
                        overlapped_chunk = overlap_text + "\n\n" + chunk
                        overlapped_chunks.append(overlapped_chunk)
                    else:
                        overlapped_chunks.append(chunk)
            
            return overlapped_chunks
        
        return chunks

    def _get_overlap_text(self, text: str, overlap_tokens: int) -> str:
        """Get the last part of text for overlap"""
        words = text.split()
        overlap_text = ""
        
        # Build overlap from the end
        for word in reversed(words):
            test_text = word + " " + overlap_text if overlap_text else word
            
            if self._count_tokens(test_text) <= overlap_tokens:
                overlap_text = test_text
            else:
                break
        
        return overlap_text.strip()

    def clear_collection(self):
        """Clear all data from the collection"""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            print(f"Error clearing collection: {e}")

    def process_web_content(self, web_content: List[Dict[str, Any]], domain: str = None, use_cached_embeddings: bool = True):
        """Process web content and store in ChromaDB with embedding caching"""
        if not web_content:
            return
        
        if use_cached_embeddings:
            self._load_cached_embeddings(web_content)
        else:
            self._create_new_embeddings(web_content)

    def _load_cached_embeddings(self, web_content: List[Dict[str, Any]]):
        """Load pre-computed embeddings from cached content"""
        documents = []
        embeddings = []
        metadatas = []
        ids = []
        
        doc_id = 0
        
        for page in web_content:
            content = page.get('content', '')
            if not content.strip():
                continue
            
            # Check if page has cached embeddings
            cached_embeddings = page.get('embeddings', [])
            cached_chunks = page.get('chunks', [])
            
            if cached_embeddings and cached_chunks and len(cached_embeddings) == len(cached_chunks):
                # Use cached embeddings
                for chunk, embedding in zip(cached_chunks, cached_embeddings):
                    documents.append(chunk)
                    embeddings.append(embedding)
                    metadatas.append({
                        'url': page.get('url', ''),
                        'title': page.get('title', ''),
                        'domain': page.get('domain', ''),
                        'word_count': len(chunk.split()),
                        'chunk_index': len(documents) - 1
                    })
                    ids.append(f"doc_{doc_id}")
                    doc_id += 1
            else:
                # No cached embeddings, create new ones
                chunks = self._smart_chunk_text(content)
                chunk_embeddings = []
                
                for chunk in chunks:
                    embedding = self._get_embedding(chunk)
                    if embedding:
                        chunk_embeddings.append(embedding)
                        documents.append(chunk)
                        embeddings.append(embedding)
                        metadatas.append({
                            'url': page.get('url', ''),
                            'title': page.get('title', ''),
                            'domain': page.get('domain', ''),
                            'word_count': len(chunk.split()),
                            'chunk_index': len(documents) - 1
                        })
                        ids.append(f"doc_{doc_id}")
                        doc_id += 1
                
                # Cache the embeddings for future use
                page['chunks'] = chunks
                page['embeddings'] = chunk_embeddings
        
        # Add to ChromaDB if we have content
        if documents:
            try:
                self.collection.add(
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
            except Exception as e:
                print(f"Error adding to ChromaDB: {e}")

    def _create_new_embeddings(self, web_content: List[Dict[str, Any]]):
        """Create new embeddings and store them in content for caching"""
        documents = []
        embeddings = []
        metadatas = []
        ids = []
        
        # Collect all chunks first for batch processing
        all_chunks = []
        chunk_page_mapping = []
        
        doc_id = 0
        
        for page_idx, page in enumerate(web_content):
            content = page.get('content', '')
            if not content.strip():
                continue
            
            chunks = self._smart_chunk_text(content)
            page_chunks = []
            
            for chunk in chunks:
                all_chunks.append(chunk)
                chunk_page_mapping.append(page_idx)
                page_chunks.append(chunk)
                
                metadatas.append({
                    'url': page.get('url', ''),
                    'title': page.get('title', ''),
                    'domain': page.get('domain', ''),
                    'word_count': len(chunk.split()),
                    'chunk_index': len(all_chunks) - 1
                })
                ids.append(f"doc_{doc_id}")
                doc_id += 1
            
            # Store chunks in page data
            page['chunks'] = page_chunks
            page['embeddings'] = []  # Will be populated after batch processing
        
        # Batch process embeddings for better performance
        if all_chunks:
            documents = all_chunks
            
            # Process embeddings in smaller batches to avoid rate limits
            batch_size = 10
            all_embeddings = []
            
            for i in range(0, len(all_chunks), batch_size):
                batch_chunks = all_chunks[i:i + batch_size]
                batch_embeddings = []
                
                for chunk in batch_chunks:
                    embedding = self._get_embedding(chunk)
                    if embedding:
                        batch_embeddings.append(embedding)
                    else:
                        # Use a zero vector if embedding fails
                        batch_embeddings.append([0.0] * 1536)  # OpenAI embedding dimension
                
                all_embeddings.extend(batch_embeddings)
                
                # Small delay to respect rate limits
                import time
                time.sleep(0.1)
            
            embeddings = all_embeddings
            
            # Map embeddings back to pages for caching
            embedding_idx = 0
            for page_idx, page in enumerate(web_content):
                page_chunk_count = len(page.get('chunks', []))
                if page_chunk_count > 0:
                    page_embeddings = embeddings[embedding_idx:embedding_idx + page_chunk_count]
                    page['embeddings'] = page_embeddings
                    embedding_idx += page_chunk_count
        
        # Add to ChromaDB
        if documents:
            try:
                self.collection.add(
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
            except Exception as e:
                print(f"Error adding to ChromaDB: {e}")

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text using OpenAI"""
        try:
            response = openai.embeddings.create(
                model=RAG_SETTINGS["embedding_model"],
                input=text.replace("\n", " ")
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None

    def search_content(self, query: str, filters: Dict[str, Any] = None, k: int = 10) -> List[Dict[str, Any]]:
        """Search for relevant content using semantic similarity"""
        try:
            query_embedding = self._get_embedding(query)
            if not query_embedding:
                return []
            
            # Build where clause for filtering
            where_clause = None
            if filters:
                where_clause = {}
                for key, value in filters.items():
                    if value:
                        where_clause[key] = value
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, 100),
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results
            search_results = []
            if results and results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                
                for doc, metadata, distance in zip(documents, metadatas, distances):
                    # Convert distance to similarity (ChromaDB uses cosine distance 0-2)
                    similarity = 1 - (distance / 2)
                    
                    search_results.append({
                        'content': doc,
                        'url': metadata.get('url', ''),
                        'title': metadata.get('title', ''),
                        'domain': metadata.get('domain', ''),
                        'similarity_score': max(0, min(1, similarity)),
                        'word_count': metadata.get('word_count', 0)
                    })
            
            return search_results
            
        except Exception as e:
            print(f"Error searching content: {e}")
            return []

    def analyze_content(self, question: str, context_limit: int = 5, verbosity: str = 'concise') -> Dict[str, Any]:
        """Analyze content and provide intelligent insights"""
        try:
            # Search for relevant content
            relevant_content = self.search_content(question, k=context_limit * 2)
            
            if not relevant_content:
                return {
                    'answer': "I couldn't find relevant content to answer your question. Please try rephrasing or check if the website has been crawled.",
                    'sources': [],
                    'confidence': 0.0
                }
            
            # Filter by similarity threshold
            filtered_content = [
                content for content in relevant_content 
                if content['similarity_score'] >= RAG_SETTINGS["similarity_threshold"]
            ]
            
            if not filtered_content:
                return {
                    'answer': "I found some content but it doesn't seem relevant enough to your question. Please try rephrasing your question or check if the website contains information about this topic.",
                    'sources': [],
                    'confidence': 0.0
                }
            
            # Prepare context
            context_parts = []
            sources = []
            
            for content in filtered_content[:context_limit]:
                context_parts.append(content['content'])
                sources.append({
                    'url': content['url'],
                    'title': content['title'],
                    'similarity_score': content['similarity_score'],
                    'domain': content['domain']
                })
            
            context = '\n---\n'.join(context_parts)
            
            # Deduplicate sources by URL and keep the highest relevance
            unique_sources = {}
            for source in sources:
                url = source['url']
                if url not in unique_sources or source['similarity_score'] > unique_sources[url]['similarity_score']:
                    unique_sources[url] = source
            
            sources = list(unique_sources.values())[:5]  # Limit to 5 unique sources
            
            # Calculate confidence based on source quality and relevance
            if sources:
                similarities = [s['similarity_score'] for s in sources]
                avg_similarity = sum(similarities) / len(similarities)
                source_count_factor = min(len(sources) / 3, 1.0)  # More sources = higher confidence
                confidence = (avg_similarity * 0.7 + source_count_factor * 0.3)
            else:
                confidence = 0.0
            
            # Truncate context if too long
            max_context_tokens = RAG_SETTINGS["max_context_length"]
            if self._count_tokens(context) > max_context_tokens:
                # Truncate context to fit within token limit
                context_words = context.split()
                while self._count_tokens(' '.join(context_words)) > max_context_tokens and context_words:
                    context_words.pop()
                context = ' '.join(context_words)
            
            # Generate answer based on verbosity
            if verbosity == 'concise':
                system_prompt = "You are a helpful assistant that provides concise, direct answers based on the given context. Keep responses brief and to the point."
                max_tokens = 150
            elif verbosity == 'comprehensive':
                system_prompt = "You are a helpful assistant that provides detailed, comprehensive answers based on the given context. Include relevant details and explanations."
                max_tokens = 500
            else:  # balanced
                system_prompt = "You are a helpful assistant that provides clear, informative answers based on the given context. Balance detail with conciseness."
                max_tokens = 300
            
            # Create the prompt
            prompt = f"""Based on the following context from a website, please answer the question.

Context:
{context}

Question: {question}

Please provide a helpful answer based only on the information provided in the context above. If the context doesn't contain enough information to answer the question, say so clearly."""
            
            # Call OpenAI API
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.3,
                timeout=30
            )
            
            answer = response.choices[0].message.content.strip()
            
            return {
                'answer': answer,
                'sources': sources,
                'confidence': confidence,
                'context_used': len(filtered_content),
                'verbosity': verbosity
            }
            
        except Exception as e:
            print(f"Error analyzing content: {e}")
            return {
                'answer': f"I encountered an error while processing your question: {str(e)}",
                'sources': [],
                'confidence': 0.0
            }

    def get_content_summary(self) -> Dict[str, Any]:
        """Get summary of the stored content"""
        try:
            result = self.collection.count()
            return {
                'total_chunks': result,
                'collection_name': self.collection_name
            }
        except Exception:
            return {'total_chunks': 0, 'collection_name': self.collection_name}

    def suggest_questions(self) -> List[str]:
        """Suggest relevant questions based on the content"""
        # Basic question suggestions - could be enhanced with AI generation
        return [
            "What is this website about?",
            "What are the main topics covered?",
            "What products or services are offered?",
            "How can I contact them?",
            "What are the key features mentioned?",
            "What pricing information is available?"
        ]