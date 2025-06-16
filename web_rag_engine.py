import os
import chromadb
from datetime import datetime
from typing import List, Dict, Any, Optional
from openai import OpenAI
import tiktoken
import re
import json
from urllib.parse import urlparse

class WebRAGEngine:
    def __init__(self, collection_name: str = "web_content"):
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.client = OpenAI(api_key=api_key)
        self.encoding = tiktoken.encoding_for_model("gpt-4o")
        
        # Initialize ChromaDB with persistent storage
        self.chroma_client = chromadb.PersistentClient(path="./web_content_db")
        
        # Create collection with metadata
        self.collection_name = collection_name
        self.collection = self._get_or_create_collection()
        
        # Store crawl metadata
        self.crawl_metadata = {}
        
    def _get_or_create_collection(self):
        """Get or create ChromaDB collection"""
        try:
            return self.chroma_client.get_collection(self.collection_name)
        except Exception:
            return self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "Web content for RAG analysis"}
            )
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def _smart_chunk_text(self, text: str, max_tokens: int = 500, overlap_tokens: int = 50) -> List[str]:
        """Intelligently chunk text based on semantic boundaries"""
        if not text or self._count_tokens(text) <= max_tokens:
            return [text] if text.strip() else []
        
        # Split on paragraphs first
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            para_tokens = self._count_tokens(paragraph)
            
            # If paragraph itself is too long, split on sentences
            if para_tokens > max_tokens:
                sentences = re.split(r'[.!?]+', paragraph)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    sent_tokens = self._count_tokens(sentence)
                    if current_tokens + sent_tokens > max_tokens and current_chunk:
                        chunks.append(current_chunk.strip())
                        # Add overlap from previous chunk
                        overlap_text = self._get_overlap_text(current_chunk, overlap_tokens)
                        current_chunk = overlap_text + sentence
                        current_tokens = self._count_tokens(current_chunk)
                    else:
                        current_chunk += " " + sentence if current_chunk else sentence
                        current_tokens += sent_tokens
            else:
                # Check if adding this paragraph exceeds token limit
                if current_tokens + para_tokens > max_tokens and current_chunk:
                    chunks.append(current_chunk.strip())
                    # Add overlap from previous chunk
                    overlap_text = self._get_overlap_text(current_chunk, overlap_tokens)
                    current_chunk = overlap_text + paragraph
                    current_tokens = self._count_tokens(current_chunk)
                else:
                    current_chunk += "\n\n" + paragraph if current_chunk else paragraph
                    current_tokens += para_tokens
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _get_overlap_text(self, text: str, overlap_tokens: int) -> str:
        """Get the last part of text for overlap"""
        words = text.split()
        if len(words) <= 10:
            return ""
        
        # Get approximately the last overlap_tokens worth of text
        overlap_words = []
        current_tokens = 0
        
        for word in reversed(words):
            word_tokens = self._count_tokens(word)
            if current_tokens + word_tokens > overlap_tokens:
                break
            overlap_words.insert(0, word)
            current_tokens += word_tokens
        
        return " ".join(overlap_words) + " " if overlap_words else ""
    
    def clear_collection(self):
        """Clear all data from the collection"""
        try:
            self.chroma_client.delete_collection(self.collection_name)
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "Web content for RAG analysis"}
            )
            print("Collection cleared successfully")
        except Exception as e:
            print(f"Error clearing collection: {str(e)}")
    
    def process_web_content(self, web_content: List[Dict[str, Any]], domain: str = None, use_cached_embeddings: bool = True):
        """Process web content and store in ChromaDB with embedding caching"""
        print(f"Processing {len(web_content)} web pages for RAG...")
        
        # Clear existing data
        self.clear_collection()
        
        # Store crawl metadata
        self.crawl_metadata = {
            'domain': domain or (urlparse(web_content[0]['url']).netloc if web_content else 'unknown'),
            'total_pages': len(web_content),
            'processed_at': datetime.now().isoformat(),
            'total_words': sum(page.get('word_count', 0) for page in web_content)
        }
        
        # Check if content already has cached embeddings
        has_cached_embeddings = all(page.get('embeddings_chunks') for page in web_content)
        
        if use_cached_embeddings and has_cached_embeddings:
            print("Using cached embeddings...")
            self._load_cached_embeddings(web_content)
        else:
            print("Creating new embeddings...")
            self._create_new_embeddings(web_content)
    
    def _load_cached_embeddings(self, web_content: List[Dict[str, Any]]):
        """Load pre-computed embeddings from cached content"""
        doc_ids = []
        doc_texts = []
        doc_metadatas = []
        
        total_chunks = 0
        for page_idx, page in enumerate(web_content):
            cached_chunks = page.get('embeddings_chunks', [])
            total_chunks += len(cached_chunks)
            
            for chunk in cached_chunks:
                doc_ids.append(chunk['id'])
                doc_texts.append(chunk['text'])
                doc_metadatas.append(chunk['metadata'])
        
        # Add to ChromaDB in batches
        batch_size = 50
        for i in range(0, len(doc_ids), batch_size):
            batch_end = min(i + batch_size, len(doc_ids))
            batch_ids = doc_ids[i:batch_end]
            batch_texts = doc_texts[i:batch_end]
            batch_metadatas = doc_metadatas[i:batch_end]
            
            try:
                self.collection.add(
                    documents=batch_texts,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                print(f"Added cached batch {i//batch_size + 1}: {len(batch_ids)} chunks")
            except Exception as e:
                print(f"Error adding cached batch {i//batch_size + 1}: {str(e)}")
        
        print(f"Successfully loaded {total_chunks} cached chunks from {len(web_content)} pages")
    
    def _create_new_embeddings(self, web_content: List[Dict[str, Any]]):
        """Create new embeddings and store them in content for caching"""
        doc_ids = []
        doc_texts = []
        doc_metadatas = []
        
        for page_idx, page in enumerate(web_content):
            try:
                # Extract page information
                url = page.get('url', '')
                title = page.get('title', '').strip()
                content = page.get('content', '').strip()
                description = page.get('description', '').strip()
                headings = page.get('headings', [])
                keywords = page.get('keywords', '').strip()
                word_count = page.get('word_count', 0)
                
                if not content:
                    continue
                
                # Create enhanced content for better embeddings
                # Start with the most important content first
                content_parts = []
                
                # Lead with title and main content for better semantic matching
                if title:
                    content_parts.append(title)
                
                if description:
                    content_parts.append(description)
                
                # Add main content
                content_parts.append(content)
                
                # Add supplementary information
                if headings:
                    content_parts.append(f"Key sections: {' | '.join(headings[:3])}")
                
                if keywords:
                    content_parts.append(f"Related topics: {keywords}")
                
                full_text = '\n\n'.join(content_parts)
                
                # Smart chunking with better parameters for embedding quality
                chunks = self._smart_chunk_text(full_text, max_tokens=600, overlap_tokens=100)
                
                for chunk_idx, chunk in enumerate(chunks):
                    if len(chunk.strip()) < 50:  # Skip very short chunks
                        continue
                    
                    # Create unique ID
                    doc_id = f"page_{page_idx}_chunk_{chunk_idx}"
                    
                    # Enhanced metadata
                    metadata = {
                        'url': url,
                        'title': title[:200] if title else 'Untitled',
                        'page_index': str(page_idx),
                        'chunk_index': str(chunk_idx),
                        'word_count': str(word_count),
                        'has_description': str(bool(description)),
                        'headings_count': str(len(headings)),
                        'domain': urlparse(url).netloc if url else 'unknown',
                        'url_path': urlparse(url).path if url else '',
                        'processed_at': datetime.now().isoformat()
                    }
                    
                    # Add first few keywords as separate metadata
                    if keywords:
                        keyword_list = [k.strip() for k in keywords.split(',')][:3]
                        for i, keyword in enumerate(keyword_list):
                            metadata[f'keyword_{i}'] = keyword
                    
                    doc_ids.append(doc_id)
                    doc_texts.append(chunk)
                    doc_metadatas.append(metadata)
                
                # Store chunks in page for caching
                if 'embeddings_chunks' not in page:
                    page['embeddings_chunks'] = []
                
                # Add chunks to page cache
                for chunk_idx, chunk in enumerate(chunks):
                    if len(chunk.strip()) < 50:
                        continue
                    
                    doc_id = f"page_{page_idx}_chunk_{chunk_idx}"
                    metadata = {
                        'url': url,
                        'title': title[:200] if title else 'Untitled',
                        'page_index': str(page_idx),
                        'chunk_index': str(chunk_idx),
                        'word_count': str(word_count),
                        'has_description': str(bool(description)),
                        'headings_count': str(len(headings)),
                        'domain': urlparse(url).netloc if url else 'unknown',
                        'url_path': urlparse(url).path if url else '',
                        'processed_at': datetime.now().isoformat()
                    }
                    
                    if keywords:
                        keyword_list = [k.strip() for k in keywords.split(',')][:3]
                        for i, keyword in enumerate(keyword_list):
                            metadata[f'keyword_{i}'] = keyword
                    
                    page['embeddings_chunks'].append({
                        'id': doc_id,
                        'text': chunk,
                        'metadata': metadata
                    })
                    
            except Exception as e:
                print(f"Error processing page {page_idx}: {str(e)}")
                continue
        
        # Add to ChromaDB in batches
        batch_size = 50
        total_added = 0
        
        for i in range(0, len(doc_ids), batch_size):
            batch_ids = doc_ids[i:i+batch_size]
            batch_texts = doc_texts[i:i+batch_size]
            batch_metadatas = doc_metadatas[i:i+batch_size]
            
            try:
                self.collection.add(
                    ids=batch_ids,
                    documents=batch_texts,
                    metadatas=batch_metadatas
                )
                total_added += len(batch_ids)
                print(f"Added batch {i//batch_size + 1}: {len(batch_ids)} chunks")
            except Exception as e:
                print(f"Error adding batch to ChromaDB: {str(e)}")
        
        print(f"Successfully processed {total_added} content chunks from {len(web_content)} pages")
    
    def search_content(self, query: str, filters: Dict[str, Any] = None, k: int = 10) -> List[Dict[str, Any]]:
        """Search for relevant content using semantic similarity"""
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
                        metadata = results['metadatas'][0][i] if results.get('metadatas') else {}
                        distance = results['distances'][0][i] if results.get('distances') else 0.5
                        
                        # ChromaDB uses cosine distance (0 = identical, 2 = opposite)
                        # Convert to similarity score (1 = identical, 0 = opposite)
                        similarity = max(0.0, 1.0 - (distance / 2.0))
                        
                        result = {
                            'id': results['ids'][0][i],
                            'content': results['documents'][0][i],
                            'metadata': metadata,
                            'similarity_score': similarity,
                            'url': metadata.get('url', ''),
                            'title': metadata.get('title', 'Untitled'),
                            'domain': metadata.get('domain', ''),
                            'word_count': metadata.get('word_count', '0')
                        }
                        formatted_results.append(result)
                    except Exception as e:
                        print(f"Error formatting result {i}: {str(e)}")
                        continue
            
            return formatted_results
            
        except Exception as e:
            print(f"Error searching content: {str(e)}")
            return []
    
    def analyze_content(self, question: str, context_limit: int = 5, verbosity: str = 'concise') -> Dict[str, Any]:
        """Analyze content and provide intelligent insights"""
        try:
            # Search for relevant content with more results for better context
            relevant_content = self.search_content(question, k=min(context_limit * 2, 15))
            
            if not relevant_content:
                return {
                    'answer': "I couldn't find relevant content to answer your question. Please try rephrasing or check if the website has been crawled.",
                    'sources': [],
                    'confidence': 0.0
                }
            
            # Prepare context
            context_parts = []
            sources = []
            
            for content in relevant_content:
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
                # Weight by both similarity and number of quality sources
                similarities = [s['similarity_score'] for s in sources]
                avg_similarity = sum(similarities) / len(similarities)
                
                # Boost confidence for multiple good sources
                source_count_bonus = min(0.1, len(sources) * 0.02)  # Up to 10% bonus for multiple sources
                
                # Penalize if top similarity is very low
                top_similarity = max(similarities) if similarities else 0
                if top_similarity < 0.3:
                    confidence = max(0.1, avg_similarity * 0.7)  # Reduce confidence for poor matches
                else:
                    confidence = min(0.95, avg_similarity + source_count_bonus)
            else:
                confidence = 0.1
            
            # Configure response style based on verbosity
            verbosity_configs = {
                'concise': {
                    'system_prompt': f"""You are a precise content analyst. Provide brief, accurate answers using only the provided web content.

Domain: {self.crawl_metadata.get('domain', 'Unknown')}
Total Pages: {self.crawl_metadata.get('total_pages', 0)}

Guidelines:
- Keep answers concise and to the point
- Use only information from the provided context
- Be specific and factual
- If information is insufficient, state this briefly
- Focus on key facts, not elaboration""",
                    'user_instruction': "Provide a precise, concise answer based on the content above. Keep it brief and factual.",
                    'max_tokens': 400,
                    'temperature': 0.1
                },
                'balanced': {
                    'system_prompt': f"""You are an expert content analyst. Provide clear, well-structured answers using the provided web content.

Domain: {self.crawl_metadata.get('domain', 'Unknown')}
Total Pages: {self.crawl_metadata.get('total_pages', 0)}

Guidelines:
- Provide balanced detail - not too brief, not too verbose
- Use only information from the provided context
- Include relevant examples and specifics
- Organize information clearly
- Explain key concepts when helpful""",
                    'user_instruction': "Provide a clear, well-structured answer with appropriate detail. Include relevant examples and organize the information logically.",
                    'max_tokens': 800,
                    'temperature': 0.2
                },
                'comprehensive': {
                    'system_prompt': f"""You are an expert content analyst. Provide detailed, comprehensive answers using the provided web content.

Domain: {self.crawl_metadata.get('domain', 'Unknown')}
Total Pages: {self.crawl_metadata.get('total_pages', 0)}

Guidelines:
- Provide thorough, detailed analysis
- Use only information from the provided context
- Include all relevant details, examples, and insights
- Explain context and implications
- Structure information with clear sections
- Highlight patterns and connections""",
                    'user_instruction': "Provide a comprehensive, detailed analysis based on the content above. Include all relevant information, examples, and insights. Structure your response clearly.",
                    'max_tokens': 1500,
                    'temperature': 0.3
                }
            }
            
            config = verbosity_configs.get(verbosity, verbosity_configs['concise'])
            
            user_prompt = f"""Web Content Context:
{context}

Question: {question}

{config['user_instruction']}"""
            
            # Get response from OpenAI
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": config['system_prompt']},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=config['max_tokens'],
                temperature=config['temperature']
            )
            
            answer = response.choices[0].message.content
            
            return {
                'answer': answer,
                'sources': sources,
                'confidence': confidence,
                'total_pages_searched': len(relevant_content),
                'domain': self.crawl_metadata.get('domain', 'Unknown')
            }
            
        except Exception as e:
            return {
                'answer': f"I encountered an error while analyzing the content: {str(e)}",
                'sources': [],
                'confidence': 0.0
            }
    
    def get_content_summary(self) -> Dict[str, Any]:
        """Get summary of the stored content"""
        try:
            count = self.collection.count()
            
            # Get sample of documents to analyze
            sample_results = self.collection.query(
                query_texts=["content summary overview"],
                n_results=min(10, count),
                include=['documents', 'metadatas']
            )
            
            # Analyze domains and topics
            domains = set()
            titles = []
            
            if sample_results and sample_results.get('metadatas'):
                for metadata in sample_results['metadatas'][0]:
                    if metadata.get('domain'):
                        domains.add(metadata['domain'])
                    if metadata.get('title'):
                        titles.append(metadata['title'])
            
            return {
                'total_chunks': count,
                'domains': list(domains),
                'sample_titles': titles[:5],
                'crawl_metadata': self.crawl_metadata,
                'collection_name': self.collection_name
            }
            
        except Exception as e:
            return {
                'total_chunks': 0,
                'error': str(e),
                'collection_name': self.collection_name
            }
    
    def suggest_questions(self) -> List[str]:
        """Suggest relevant questions based on the content"""
        try:
            # Get sample content to analyze
            sample_results = self.collection.query(
                query_texts=["main topics overview"],
                n_results=5,
                include=['documents', 'metadatas']
            )
            
            if not sample_results or not sample_results.get('documents'):
                return [
                    "What is this website about?",
                    "What are the main topics covered?",
                    "Can you summarize the key information?"
                ]
            
            # Generate questions based on content
            sample_content = '\n'.join(sample_results['documents'][0][:3])
            domain = self.crawl_metadata.get('domain', 'this website')
            
            suggested_questions = [
                f"What is {domain} about?",
                f"What are the main features or services offered by {domain}?",
                f"Can you summarize the key information from {domain}?",
                "What are the most important topics covered?",
                "What products or services are mentioned?",
                "What contact information or details are available?",
                "What are the main benefits highlighted?",
                "Are there any pricing or cost details mentioned?"
            ]
            
            return suggested_questions
            
        except Exception as e:
            print(f"Error generating questions: {e}")
            return [
                "What is this website about?",
                "What are the main topics covered?",
                "Can you summarize the key information?"
            ]