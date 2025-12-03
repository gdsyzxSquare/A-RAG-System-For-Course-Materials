"""
Chunking Module for RAG System
Implements multiple chunking strategies for document splitting
Includes metadata extraction and dedicated metadata chunks
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Chunk:
    """Represents a text chunk with metadata"""
    text: str
    chunk_id: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any] = None
    chunk_type: str = "content"  # 'content' or 'metadata'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "text": self.text,
            "chunk_id": self.chunk_id,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "metadata": self.metadata or {},
            "chunk_type": self.chunk_type
        }


class MetadataExtractor:
    """Extract structured metadata from course documents"""
    
    # Metadata extraction patterns
    PATTERNS = {
        'instructor': [
            r'taught by (?:Prof\.|Professor)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+[A-Z]+)?)',
            r'Instructor:\s*(?:Prof\.|Professor)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'# ([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z]+)?)\s+AI Thrust',
        ],
        'location': [
            r'Location:\s*([^.\n]+)',
            r'Room\s+(\d+[A-Z]?,\s*[A-Z]\d+)',
            r'Rm\s+(\d+,?\s*[A-Z]\d+)',
        ],
        'time': [
            r'Time:\s*([^.\n]+(?:AM|PM)[^.\n]*)',
            r'((?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+\d{1,2}:\d{2}(?:AM|PM)\s*[-–]\s*\d{1,2}:\d{2}(?:AM|PM))',
        ],
        'email': [
            r'\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b',
        ],
        'section': [
            r'Section:\s*([A-Z]\d+)',
            r'Section\s+([A-Z]\d+\s*\(\d+\))',
        ],
        'grading': [
            r'((?:Homework|Assignment|Exam|Project|Final|Midterm|Quiz|Attendance)[^.]*\d+%)',
            r'(\d+%[^.]*(?:homework|assignment|exam|project|final|midterm|quiz|attendance))',
        ]
    }
    
    def extract(self, text: str) -> Dict[str, List[str]]:
        """
        Extract all metadata from text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of metadata fields and their values
        """
        metadata = {}
        
        for field, patterns in self.PATTERNS.items():
            values = []
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                values.extend([m.strip() for m in matches if m.strip()])
            
            if values:
                # Remove duplicates while preserving order
                unique_values = list(dict.fromkeys(values))
                metadata[field] = unique_values
        
        return metadata
    
    def create_metadata_chunks(
        self, 
        text: str, 
        start_chunk_id: int = 0
    ) -> List[Chunk]:
        """
        Create dedicated chunks for metadata
        
        Args:
            text: Source text
            start_chunk_id: Starting ID for metadata chunks
            
        Returns:
            List of metadata chunks
        """
        metadata = self.extract(text)
        chunks = []
        chunk_id = start_chunk_id
        
        for field, values in metadata.items():
            for value in values:
                # Create a natural language text for the metadata
                if field == 'instructor':
                    chunk_text = f"The instructor is {value}. This course is taught by {value}."
                elif field == 'location':
                    chunk_text = f"The class is located at {value}. The classroom is {value}."
                elif field == 'time':
                    chunk_text = f"The class time is {value}. Classes are held at {value}."
                elif field == 'email':
                    chunk_text = f"Contact email: {value}. You can reach us at {value}."
                elif field == 'section':
                    chunk_text = f"This is section {value} of the course."
                elif field == 'grading':
                    chunk_text = f"Grading policy: {value}"
                else:
                    chunk_text = f"{field}: {value}"
                
                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    start_char=-1,  # Metadata chunks don't have position
                    end_char=-1,
                    metadata={
                        "chunking_method": "metadata_extraction",
                        "metadata_field": field,
                        "metadata_value": value
                    },
                    chunk_type="metadata"
                ))
                chunk_id += 1
        
        return chunks


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
    def create_chunker(strategy: str, extract_metadata: bool = True, **kwargs):
        """
        Create a chunker based on strategy
        
        Args:
            strategy: Chunking strategy ('fixed_size', 'semantic', 'sliding_window')
            extract_metadata: Whether to extract and create metadata chunks
            **kwargs: Additional arguments for the chunker
            
        Returns:
            Chunker instance with optional metadata extraction wrapper
        """
        # Create base chunker
        if strategy == "fixed_size":
            base_chunker = FixedSizeChunker(**kwargs)
        elif strategy == "semantic":
            # Map chunk_size to max_chunk_size for semantic chunker
            if 'chunk_size' in kwargs and 'max_chunk_size' not in kwargs:
                kwargs['max_chunk_size'] = kwargs.pop('chunk_size')
            # Remove overlap parameter as semantic chunker doesn't use it
            kwargs.pop('overlap', None)
            base_chunker = SemanticChunker(**kwargs)
        elif strategy == "sliding_window":
            # Map chunk_size to window_size, overlap to step_size
            if 'chunk_size' in kwargs and 'window_size' not in kwargs:
                kwargs['window_size'] = kwargs.pop('chunk_size')
            if 'overlap' in kwargs and 'step_size' not in kwargs:
                overlap = kwargs.pop('overlap')
                # step_size = window_size - overlap
                kwargs['step_size'] = kwargs.get('window_size', 512) - overlap
            base_chunker = SlidingWindowChunker(**kwargs)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
        
        # Wrap with metadata extraction if enabled
        if extract_metadata:
            return MetadataEnhancedChunker(base_chunker)
        else:
            return base_chunker


class MetadataEnhancedChunker:
    """Wrapper that adds metadata chunks to any base chunker"""
    
    def __init__(self, base_chunker):
        """
        Initialize with a base chunker
        
        Args:
            base_chunker: The underlying chunking strategy
        """
        self.base_chunker = base_chunker
        self.metadata_extractor = MetadataExtractor()
    
    def chunk(self, text: str) -> List[Chunk]:
        """
        Chunk text and add dedicated metadata chunks
        
        Args:
            text: Input text
            
        Returns:
            List of chunks (content + metadata)
        """
        # Get content chunks from base chunker
        content_chunks = self.base_chunker.chunk(text)
        
        # Extract and create metadata chunks
        metadata_chunks = self.metadata_extractor.create_metadata_chunks(
            text, 
            start_chunk_id=len(content_chunks)
        )
        
        # Combine: metadata chunks first for priority
        all_chunks = metadata_chunks + content_chunks
        
        # Re-number chunk IDs
        for i, chunk in enumerate(all_chunks):
            chunk.chunk_id = i
        
        if metadata_chunks:
            print(f"✓ Added {len(metadata_chunks)} metadata chunks")
            print(f"  Total chunks: {len(all_chunks)} ({len(metadata_chunks)} metadata + {len(content_chunks)} content)")
        
        return all_chunks


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
