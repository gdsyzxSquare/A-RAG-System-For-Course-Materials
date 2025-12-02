"""
Retriever Module for RAG System
Implements BM25 and Dense retrieval methods
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
from src.embeddings import EmbeddingGenerator, QueryEncoder
from src.vector_store import FAISSVectorStore


class DenseRetriever:
    """Dense retrieval using vector similarity search"""
    
    def __init__(
        self, 
        vector_store: FAISSVectorStore,
        query_encoder: QueryEncoder
    ):
        """
        Initialize DenseRetriever
        
        Args:
            vector_store: FAISS vector store
            query_encoder: Query encoder
        """
        self.vector_store = vector_store
        self.query_encoder = query_encoder
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 5
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Retrieve relevant chunks using dense retrieval
        
        Args:
            query: Query text
            top_k: Number of results to retrieve
            
        Returns:
            Tuple of (retrieved chunks, similarity scores)
        """
        # Encode query
        query_embedding = self.query_encoder.encode_query(query)
        
        # Search in vector store
        chunks, distances = self.vector_store.search(query_embedding, top_k)
        
        # Convert L2 distances to similarity scores (inverse)
        # Lower distance = higher similarity
        scores = [1.0 / (1.0 + d) for d in distances]
        
        return chunks, scores


class BM25Retriever:
    """Sparse retrieval using BM25 algorithm"""
    
    def __init__(
        self, 
        chunks: List[Dict[str, Any]],
        k1: float = 1.5,
        b: float = 0.75
    ):
        """
        Initialize BM25Retriever
        
        Args:
            chunks: List of chunk dictionaries with 'text' field
            k1: BM25 k1 parameter
            b: BM25 b parameter
        """
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        
        # Tokenize documents
        self.tokenized_docs = [self._tokenize(chunk['text']) for chunk in chunks]
        
        # Create BM25 index
        self.bm25 = BM25Okapi(self.tokenized_docs)
        print(f"✓ Created BM25 index with {len(chunks)} documents")
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization (can be improved)
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Simple word tokenization (lowercase, split by space)
        return text.lower().split()
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 5
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Retrieve relevant chunks using BM25
        
        Args:
            query: Query text
            top_k: Number of results to retrieve
            
        Returns:
            Tuple of (retrieved chunks, BM25 scores)
        """
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Get corresponding chunks and scores
        retrieved_chunks = [self.chunks[i] for i in top_indices]
        retrieved_scores = [float(scores[i]) for i in top_indices]
        
        return retrieved_chunks, retrieved_scores


class HybridRetriever:
    """Hybrid retrieval combining BM25 and Dense retrieval"""
    
    def __init__(
        self,
        dense_retriever: DenseRetriever,
        bm25_retriever: BM25Retriever,
        alpha: float = 0.5
    ):
        """
        Initialize HybridRetriever
        
        Args:
            dense_retriever: Dense retriever
            bm25_retriever: BM25 retriever
            alpha: Weight for dense retrieval (1-alpha for BM25)
        """
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.alpha = alpha
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 5
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Retrieve using hybrid approach
        
        Args:
            query: Query text
            top_k: Number of results to retrieve
            
        Returns:
            Tuple of (retrieved chunks, combined scores)
        """
        # Get results from both retrievers
        dense_chunks, dense_scores = self.dense_retriever.retrieve(query, top_k * 2)
        bm25_chunks, bm25_scores = self.bm25_retriever.retrieve(query, top_k * 2)
        
        # Normalize scores
        dense_scores = self._normalize_scores(dense_scores)
        bm25_scores = self._normalize_scores(bm25_scores)
        
        # Combine results
        chunk_scores = {}
        
        for chunk, score in zip(dense_chunks, dense_scores):
            chunk_id = chunk.get('chunk_id', id(chunk))
            chunk_scores[chunk_id] = {
                'chunk': chunk,
                'score': self.alpha * score
            }
        
        for chunk, score in zip(bm25_chunks, bm25_scores):
            chunk_id = chunk.get('chunk_id', id(chunk))
            if chunk_id in chunk_scores:
                chunk_scores[chunk_id]['score'] += (1 - self.alpha) * score
            else:
                chunk_scores[chunk_id] = {
                    'chunk': chunk,
                    'score': (1 - self.alpha) * score
                }
        
        # Sort by combined score
        sorted_results = sorted(
            chunk_scores.values(), 
            key=lambda x: x['score'], 
            reverse=True
        )[:top_k]
        
        retrieved_chunks = [r['chunk'] for r in sorted_results]
        retrieved_scores = [r['score'] for r in sorted_results]
        
        return retrieved_chunks, retrieved_scores
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Normalize scores to [0, 1] range
        
        Args:
            scores: List of scores
            
        Returns:
            Normalized scores
        """
        if not scores or max(scores) == min(scores):
            return [0.0] * len(scores)
        
        min_score = min(scores)
        max_score = max(scores)
        return [(s - min_score) / (max_score - min_score) for s in scores]


class RetrieverFactory:
    """Factory to create different retrievers"""
    
    @staticmethod
    def create_retriever(
        method: str,
        vector_store: FAISSVectorStore = None,
        query_encoder: QueryEncoder = None,
        chunks: List[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Create a retriever based on method
        
        Args:
            method: Retrieval method ('dense', 'bm25', 'hybrid')
            vector_store: FAISS vector store (for dense/hybrid)
            query_encoder: Query encoder (for dense/hybrid)
            chunks: List of chunks (for bm25/hybrid)
            **kwargs: Additional parameters
            
        Returns:
            Retriever instance
        """
        if method == "dense":
            if not vector_store or not query_encoder:
                raise ValueError("Dense retriever requires vector_store and query_encoder")
            return DenseRetriever(vector_store, query_encoder)
        
        elif method == "bm25":
            if not chunks:
                raise ValueError("BM25 retriever requires chunks")
            return BM25Retriever(chunks, **kwargs)
        
        elif method == "hybrid":
            if not all([vector_store, query_encoder, chunks]):
                raise ValueError("Hybrid retriever requires all components")
            dense = DenseRetriever(vector_store, query_encoder)
            bm25 = BM25Retriever(chunks)
            return HybridRetriever(dense, bm25, **kwargs)
        
        else:
            raise ValueError(f"Unknown retrieval method: {method}")


if __name__ == "__main__":
    # Example usage
    from src.embeddings import EmbeddingGenerator
    
    # Create dummy data
    chunks = [
        {"text": "Python is a programming language", "chunk_id": 0},
        {"text": "Machine learning is a subset of AI", "chunk_id": 1},
        {"text": "Deep learning uses neural networks", "chunk_id": 2}
    ]
    
    # Test BM25
    bm25_retriever = BM25Retriever(chunks)
    results, scores = bm25_retriever.retrieve("What is Python?", top_k=2)
    
    print("BM25 Results:")
    for chunk, score in zip(results, scores):
        print(f"  {chunk['text']}: {score:.4f}")
