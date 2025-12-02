"""
Chunking Module for RAG System
Implements multiple chunking strategies for document splitting
"""

import re
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class Chunk:
    """Represents a text chunk with metadata"""
    text: str
    chunk_id: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "text": self.text,
            "chunk_id": self.chunk_id,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "metadata": self.metadata or {}
        }


class FixedSizeChunker:
    """Split text into fixed-size chunks with optional overlap"""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        """
        Initialize FixedSizeChunker
        
        Args:
            chunk_size: Target size of each chunk (in characters)
            overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, text: str) -> List[Chunk]:
        """
        Split text into fixed-size chunks
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of Chunk objects
        """
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # Try to break at sentence boundary if possible
            if end < len(text):
                # Look for sentence endings near the boundary
                last_period = chunk_text.rfind('.')
                last_newline = chunk_text.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > self.chunk_size * 0.7:  # At least 70% of chunk_size
                    end = start + break_point + 1
                    chunk_text = text[start:end]
            
            chunks.append(Chunk(
                text=chunk_text.strip(),
                chunk_id=chunk_id,
                start_char=start,
                end_char=end,
                metadata={"chunking_method": "fixed_size"}
            ))
            
            chunk_id += 1
            start = end - self.overlap
        
        print(f"✓ Created {len(chunks)} fixed-size chunks")
        return chunks


class SemanticChunker:
    """Split text based on semantic boundaries (paragraphs, sections)"""
    
    def __init__(self, separators: List[str] = None, max_chunk_size: int = 1000):
        """
        Initialize SemanticChunker
        
        Args:
            separators: List of separators in priority order
            max_chunk_size: Maximum size of a chunk
        """
        self.separators = separators or ["\n\n", "\n", ". ", " "]
        self.max_chunk_size = max_chunk_size
    
    def chunk(self, text: str) -> List[Chunk]:
        """
        Split text using semantic separators
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of Chunk objects
        """
        chunks = []
        current_pos = 0
        
        # First try splitting by paragraphs (double newline)
        paragraphs = text.split('\n\n')
        
        chunk_id = 0
        current_chunk = ""
        chunk_start = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If adding this paragraph exceeds max_chunk_size, save current chunk
            if len(current_chunk) + len(para) > self.max_chunk_size and current_chunk:
                chunks.append(Chunk(
                    text=current_chunk.strip(),
                    chunk_id=chunk_id,
                    start_char=chunk_start,
                    end_char=chunk_start + len(current_chunk),
                    metadata={"chunking_method": "semantic"}
                ))
                chunk_id += 1
                chunk_start += len(current_chunk)
                current_chunk = ""
            
            current_chunk += para + "\n\n"
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(Chunk(
                text=current_chunk.strip(),
                chunk_id=chunk_id,
                start_char=chunk_start,
                end_char=chunk_start + len(current_chunk),
                metadata={"chunking_method": "semantic"}
            ))
        
        print(f"✓ Created {len(chunks)} semantic chunks")
        return chunks


class SlidingWindowChunker:
    """Split text using a sliding window approach"""
    
    def __init__(self, window_size: int = 512, step_size: int = 256):
        """
        Initialize SlidingWindowChunker
        
        Args:
            window_size: Size of the sliding window
            step_size: Step size for the window (smaller = more overlap)
        """
        self.window_size = window_size
        self.step_size = step_size
    
    def chunk(self, text: str) -> List[Chunk]:
        """
        Split text using sliding window
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of Chunk objects
        """
        chunks = []
        chunk_id = 0
        
        for start in range(0, len(text), self.step_size):
            end = start + self.window_size
            if start >= len(text):
                break
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    start_char=start,
                    end_char=min(end, len(text)),
                    metadata={"chunking_method": "sliding_window"}
                ))
                chunk_id += 1
        
        print(f"✓ Created {len(chunks)} sliding window chunks")
        return chunks


class ChunkerFactory:
    """Factory to create different chunkers"""
    
    @staticmethod
    def create_chunker(strategy: str, **kwargs):
        """
        Create a chunker based on strategy
        
        Args:
            strategy: Chunking strategy ('fixed_size', 'semantic', 'sliding_window')
            **kwargs: Additional arguments for the chunker
            
        Returns:
            Chunker instance
        """
        if strategy == "fixed_size":
            return FixedSizeChunker(**kwargs)
        elif strategy == "semantic":
            # Map chunk_size to max_chunk_size for semantic chunker
            if 'chunk_size' in kwargs and 'max_chunk_size' not in kwargs:
                kwargs['max_chunk_size'] = kwargs.pop('chunk_size')
            # Remove overlap parameter as semantic chunker doesn't use it
            kwargs.pop('overlap', None)
            return SemanticChunker(**kwargs)
        elif strategy == "sliding_window":
            # Map chunk_size to window_size, overlap to step_size
            if 'chunk_size' in kwargs and 'window_size' not in kwargs:
                kwargs['window_size'] = kwargs.pop('chunk_size')
            if 'overlap' in kwargs and 'step_size' not in kwargs:
                overlap = kwargs.pop('overlap')
                # step_size = window_size - overlap
                kwargs['step_size'] = kwargs.get('window_size', 512) - overlap
            return SlidingWindowChunker(**kwargs)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")


if __name__ == "__main__":
    # Example usage
    sample_text = """This is the first paragraph. It contains some information.
    
This is the second paragraph. It has different content.

This is the third paragraph with more details."""
    
    # Test fixed-size chunker
    chunker = FixedSizeChunker(chunk_size=50, overlap=10)
    chunks = chunker.chunk(sample_text)
    
    print("\nChunks:")
    for chunk in chunks:
        print(f"Chunk {chunk.chunk_id}: {chunk.text[:50]}...")
