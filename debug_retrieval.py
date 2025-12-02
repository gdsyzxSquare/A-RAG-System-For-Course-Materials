"""
Debug script for RAG retrieval issues
Helps diagnose why certain queries fail to retrieve relevant information
"""

import yaml
import os
from src.data_loader import DataLoader
from src.chunking import ChunkerFactory
from src.embeddings import EmbeddingGenerator, QueryEncoder
from src.vector_store import FAISSVectorStore
from src.retriever import RetrieverFactory


def debug_retrieval(query: str, config_path: str = "configs/config.yaml"):
    """
    Debug retrieval for a specific query
    
    Args:
        query: The query that's failing
        config_path: Path to config file
    """
    print("\n" + "="*70)
    print(f"DEBUG: '{query}'")
    print("="*70)
    
    # Load config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Load the raw text to see what's actually in there
    print("\n[1] Checking raw text content...")
    loader = DataLoader(config['data']['raw_data_path'])
    raw_text = loader.load_and_preprocess()
    
    # Search for keywords in raw text
    keywords = ["instructor", "teacher", "professor", "Chengwei", "Qin"]
    print(f"\nSearching for keywords in raw text ({len(raw_text)} chars):")
    for keyword in keywords:
        count = raw_text.lower().count(keyword.lower())
        print(f"  '{keyword}': {count} occurrences")
        if count > 0:
            # Show first occurrence
            idx = raw_text.lower().find(keyword.lower())
            context_start = max(0, idx - 50)
            context_end = min(len(raw_text), idx + len(keyword) + 50)
            print(f"    Sample: ...{raw_text[context_start:context_end]}...")
    
    # Load chunks
    print("\n[2] Loading saved chunks...")
    try:
        vector_store = FAISSVectorStore(embedding_dim=384)
        vector_store.load(
            config['data']['embeddings_path'] + '.index',
            config['data']['embeddings_path'] + '.json'
        )
        chunks = vector_store.chunks
        print(f"✓ Loaded {len(chunks)} chunks")
        
        # Check if keywords appear in any chunks
        print(f"\nSearching for keywords in {len(chunks)} chunks:")
        for keyword in keywords:
            found_in = []
            for i, chunk in enumerate(chunks):
                if keyword.lower() in chunk['text'].lower():
                    found_in.append(i)
            print(f"  '{keyword}': found in {len(found_in)} chunks - {found_in[:5]}")
            if found_in:
                # Show first chunk with the keyword
                sample_chunk = chunks[found_in[0]]
                print(f"    Chunk {found_in[0]} sample: {sample_chunk['text'][:150]}...")
    
    except Exception as e:
        print(f"❌ Error loading chunks: {e}")
        print("\nHint: You may need to run: python main.py --mode build")
        return
    
    # Test retrieval with different methods
    print("\n[3] Testing retrieval methods...")
    
    embedding_gen = EmbeddingGenerator(
        model_name=config['embedding']['model_name'],
        batch_size=config['embedding']['batch_size']
    )
    query_encoder = QueryEncoder(embedding_gen)
    
    # Test Dense retrieval
    print(f"\n--- Dense Retrieval (Top {config['retrieval']['top_k']}) ---")
    try:
        dense_retriever = RetrieverFactory.create_retriever(
            method="dense",
            vector_store=vector_store,
            query_encoder=query_encoder,
            chunks=chunks
        )
        retrieved_chunks, scores = dense_retriever.retrieve(query, top_k=config['retrieval']['top_k'])
        
        for i, (chunk, score) in enumerate(zip(retrieved_chunks, scores), 1):
            print(f"\n[{i}] Score: {score:.4f}")
            print(f"    Chunk ID: {chunk.get('chunk_id', 'N/A')}")
            print(f"    Text: {chunk['text'][:200]}...")
    except Exception as e:
        print(f"❌ Dense retrieval error: {e}")
    
    # Test BM25 retrieval
    print(f"\n--- BM25 Retrieval (Top {config['retrieval']['top_k']}) ---")
    try:
        bm25_retriever = RetrieverFactory.create_retriever(
            method="bm25",
            chunks=chunks
        )
        retrieved_chunks, scores = bm25_retriever.retrieve(query, top_k=config['retrieval']['top_k'])
        
        for i, (chunk, score) in enumerate(zip(retrieved_chunks, scores), 1):
            print(f"\n[{i}] Score: {score:.4f}")
            print(f"    Chunk ID: {chunk.get('chunk_id', 'N/A')}")
            print(f"    Text: {chunk['text'][:200]}...")
    except Exception as e:
        print(f"❌ BM25 retrieval error: {e}")
    
    # Recommendations
    print("\n" + "="*70)
    print("DIAGNOSIS & RECOMMENDATIONS")
    print("="*70)
    
    if len(chunks) == 1:
        print("\n⚠ WARNING: Only 1 chunk generated!")
        print("  This is likely the problem. Your entire document is in one chunk.")
        print("  Solutions:")
        print("  1. Use fixed_size chunking with smaller chunk_size (e.g., 512)")
        print("  2. Or reduce max_chunk_size for semantic chunking (e.g., 512)")
        print("\n  Update config.yaml:")
        print("    chunking:")
        print("      strategy: 'fixed_size'")
        print("      chunk_size: 512")
        print("      chunk_overlap: 50")
        print("\n  Then rebuild: python main.py --mode build")
    
    elif any(raw_text.lower().count(kw.lower()) > 0 for kw in keywords):
        print("\n✓ Keywords found in raw text")
        if not any(any(kw.lower() in c['text'].lower() for kw in keywords) for c in chunks):
            print("❌ But NOT found in any chunks!")
            print("  This suggests chunking issue. Try:")
            print("  - Smaller chunk_size to preserve more context")
            print("  - Different chunking strategy")
        else:
            print("✓ Keywords found in chunks")
            print("  Issue may be with:")
            print("  - Query embedding (try different phrasing)")
            print("  - Retrieval method (try 'hybrid' instead of 'dense')")
            print("  - Top_k too small (increase to 10)")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        # Default problematic query
        query = "who is instructor"
    
    debug_retrieval(query)
    
    print("\n" + "="*70)
    print("To test another query, run:")
    print("  python debug_retrieval.py your query here")
    print("="*70)
