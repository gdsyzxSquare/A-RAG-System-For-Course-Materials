"""
Data Loader Module for RAG System
Handles loading and preprocessing of course documents
"""

import os
import json
from typing import List, Dict, Any
from pathlib import Path


class DataLoader:
    """Load and preprocess course documents"""
    
    def __init__(self, raw_data_path: str):
        """
        Initialize DataLoader
        
        Args:
            raw_data_path: Path to raw text data file
        """
        self.raw_data_path = raw_data_path
        
    def load_text(self) -> str:
        """
        Load raw text from file
        
        Returns:
            Raw text content
        """
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"Data file not found: {self.raw_data_path}")
        
        with open(self.raw_data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        print(f"✓ Loaded {len(text)} characters from {self.raw_data_path}")
        return text
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove special characters if needed (customize based on your data)
        # text = re.sub(r'[^\w\s\u4e00-\u9fff.,;:!?()，。；：！？（）]', '', text)
        
        # Normalize line breaks
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove multiple consecutive newlines
        while '\n\n\n' in text:
            text = text.replace('\n\n\n', '\n\n')
        
        print(f"✓ Cleaned text to {len(text)} characters")
        return text
    
    def load_and_preprocess(self) -> str:
        """
        Load and preprocess data in one step
        
        Returns:
            Preprocessed text
        """
        text = self.load_text()
        text = self.clean_text(text)
        return text


class DocumentStructure:
    """Structure to represent processed documents"""
    
    def __init__(self, text: str, metadata: Dict[str, Any] = None):
        """
        Initialize Document
        
        Args:
            text: Document text content
            metadata: Optional metadata (source, page, etc.)
        """
        self.text = text
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "text": self.text,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentStructure':
        """Create from dictionary"""
        return cls(text=data["text"], metadata=data.get("metadata", {}))


def save_documents(documents: List[DocumentStructure], output_path: str):
    """
    Save processed documents to JSON
    
    Args:
        documents: List of Document objects
        output_path: Path to save JSON file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    data = [doc.to_dict() for doc in documents]
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Saved {len(documents)} documents to {output_path}")


def load_documents(input_path: str) -> List[DocumentStructure]:
    """
    Load documents from JSON
    
    Args:
        input_path: Path to JSON file
        
    Returns:
        List of Document objects
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    documents = [DocumentStructure.from_dict(d) for d in data]
    print(f"✓ Loaded {len(documents)} documents from {input_path}")
    return documents


if __name__ == "__main__":
    # Example usage
    loader = DataLoader("data/raw/course_documents.txt")
    text = loader.load_and_preprocess()
    print(f"\nFirst 500 characters:\n{text[:500]}")
