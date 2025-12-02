"""
Vector Store Module for RAG System
Manages FAISS vector index for efficient similarity search
"""

import os
import json
import numpy as np
import faiss
from typing import List, Tuple, Dict, Any


class FAISSVectorStore:
    """FAISS-based vector store for efficient similarity search"""
    
    def __init__(self, embedding_dim: int):
        """
        Initialize FAISS vector store
        
        Args:
            embedding_dim: Dimension of the embeddings
        """
        self.embedding_dim = embedding_dim
        self.index = None
        self.chunks = []
        self.is_trained = False
    
    def create_index(self, index_type: str = "flat"):
        """
        Create FAISS index
        
        Args:
            index_type: Type of index ('flat', 'ivf', 'hnsw')
        """
        if index_type == "flat":
            # Exact search using L2 distance
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        elif index_type == "ivf":
            # Approximate search with IVF
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(
                quantizer, 
                self.embedding_dim, 
                100  # number of clusters
            )
        elif index_type == "hnsw":
            # HNSW (Hierarchical Navigable Small World)
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        print(f"✓ Created {index_type.upper()} index with dimension {self.embedding_dim}")
    
    def add_embeddings(
        self, 
        embeddings: np.ndarray, 
        chunks: List[Dict[str, Any]]
    ):
        """
        Add embeddings and corresponding chunks to the index
        
        Args:
            embeddings: Numpy array of embeddings
            chunks: List of chunk dictionaries
        """
        if self.index is None:
            raise ValueError("Index not created. Call create_index() first.")
        
        # Ensure embeddings are float32 (FAISS requirement)
        embeddings = embeddings.astype('float32')
        
        # Train index if needed (for IVF)
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            print("Training index...")
            self.index.train(embeddings)
            self.is_trained = True
        
        # Add embeddings to index
        self.index.add(embeddings)
        self.chunks.extend(chunks)
        
        print(f"✓ Added {len(embeddings)} embeddings to index")
        print(f"  Total vectors in index: {self.index.ntotal}")
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 5
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Search for similar chunks
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            Tuple of (chunks, distances)
        """
        if self.index is None or self.index.ntotal == 0:
            raise ValueError("Index is empty. Add embeddings first.")
        
        # Ensure query is 2D array and float32
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Retrieve corresponding chunks
        retrieved_chunks = []
        retrieved_distances = []
        
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.chunks):
                retrieved_chunks.append(self.chunks[idx])
                retrieved_distances.append(float(dist))
        
        return retrieved_chunks, retrieved_distances
    
    def save(self, index_path: str, chunks_path: str):
        """
        Save index and chunks to disk
        
        Args:
            index_path: Path to save FAISS index
            chunks_path: Path to save chunks JSON
        """
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        os.makedirs(os.path.dirname(chunks_path), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save chunks
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Saved index to {index_path}")
        print(f"✓ Saved chunks to {chunks_path}")
    
    def load(self, index_path: str, chunks_path: str):
        """
        Load index and chunks from disk
        
        Args:
            index_path: Path to FAISS index
            chunks_path: Path to chunks JSON
        """
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load chunks
        with open(chunks_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        
        print(f"✓ Loaded index from {index_path}")
        print(f"  Total vectors: {self.index.ntotal}")
        print(f"✓ Loaded {len(self.chunks)} chunks from {chunks_path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store
        
        Returns:
            Dictionary of statistics
        """
        return {
            "total_vectors": self.index.ntotal if self.index else 0,
            "total_chunks": len(self.chunks),
            "embedding_dim": self.embedding_dim,
            "is_trained": self.is_trained
        }


class HybridVectorStore:
    """Combines multiple vector stores or retrieval methods"""
    
    def __init__(self, faiss_store: FAISSVectorStore, alpha: float = 0.5):
        """
        Initialize hybrid store
        
        Args:
            faiss_store: FAISS vector store
            alpha: Weight for dense retrieval (1-alpha for sparse)
        """
        self.faiss_store = faiss_store
        self.alpha = alpha
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 5
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Hybrid search combining dense and sparse retrieval
        
        Args:
            query_embedding: Query embedding
            top_k: Number of results
            
        Returns:
            Retrieved chunks and scores
        """
        # For now, just use FAISS (can extend to hybrid later)
        return self.faiss_store.search(query_embedding, top_k)


if __name__ == "__main__":
    # Example usage
    embedding_dim = 384
    
    # Create and test vector store
    store = FAISSVectorStore(embedding_dim)
    store.create_index("flat")
    
    # Create dummy embeddings and chunks
    dummy_embeddings = np.random.rand(10, embedding_dim).astype('float32')
    dummy_chunks = [
        {"text": f"Chunk {i}", "chunk_id": i} 
        for i in range(10)
    ]
    
    # Add to store
    store.add_embeddings(dummy_embeddings, dummy_chunks)
    
    # Test search
    query_emb = np.random.rand(embedding_dim).astype('float32')
    results, distances = store.search(query_emb, top_k=3)
    
    print("\nSearch results:")
    for chunk, dist in zip(results, distances):
        print(f"  {chunk['text']}: distance={dist:.4f}")
