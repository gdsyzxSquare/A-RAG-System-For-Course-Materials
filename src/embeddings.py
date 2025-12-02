"""
Embeddings Module for RAG System
Generates embeddings for text chunks using sentence transformers
"""

import os
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class EmbeddingGenerator:
    """Generate embeddings for text using sentence transformers"""
    
    def __init__(
        self, 
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        device: str = None
    ):
        """
        Initialize EmbeddingGenerator
        
        Args:
            model_name: Name of the sentence transformer model
            batch_size: Batch size for encoding
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.batch_size = batch_size
        
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"✓ Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def encode(
        self, 
        texts: Union[str, List[str]], 
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for text(s)
        
        Args:
            texts: Single text or list of texts
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization for cosine similarity
        )
        
        return embeddings
    
    def encode_chunks(self, chunks: List[dict]) -> np.ndarray:
        """
        Generate embeddings for a list of chunks
        
        Args:
            chunks: List of chunk dictionaries with 'text' field
            
        Returns:
            Numpy array of embeddings
        """
        texts = [chunk['text'] for chunk in chunks]
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.encode(texts)
        print(f"✓ Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def save_embeddings(self, embeddings: np.ndarray, filepath: str):
        """
        Save embeddings to file
        
        Args:
            embeddings: Numpy array of embeddings
            filepath: Path to save the embeddings
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.save(filepath, embeddings)
        print(f"✓ Saved embeddings to {filepath}")
    
    def load_embeddings(self, filepath: str) -> np.ndarray:
        """
        Load embeddings from file
        
        Args:
            filepath: Path to load embeddings from
            
        Returns:
            Numpy array of embeddings
        """
        embeddings = np.load(filepath)
        print(f"✓ Loaded embeddings from {filepath}, shape: {embeddings.shape}")
        return embeddings


class QueryEncoder:
    """Specialized encoder for queries"""
    
    def __init__(self, embedding_generator: EmbeddingGenerator):
        """
        Initialize QueryEncoder
        
        Args:
            embedding_generator: EmbeddingGenerator instance
        """
        self.generator = embedding_generator
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query
        
        Args:
            query: Query text
            
        Returns:
            Query embedding as numpy array
        """
        embedding = self.generator.encode(query, show_progress=False)
        return embedding[0] if len(embedding.shape) > 1 else embedding
    
    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """
        Encode multiple queries
        
        Args:
            queries: List of query texts
            
        Returns:
            Numpy array of query embeddings
        """
        return self.generator.encode(queries, show_progress=True)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Cosine similarity score
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


if __name__ == "__main__":
    # Example usage
    generator = EmbeddingGenerator()
    
    # Test encoding
    texts = [
        "This is a test sentence.",
        "Another example text for embedding."
    ]
    
    embeddings = generator.encode(texts)
    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"First embedding (first 5 dims): {embeddings[0][:5]}")
    
    # Test similarity
    similarity = cosine_similarity(embeddings[0], embeddings[1])
    print(f"Similarity between texts: {similarity:.4f}")
