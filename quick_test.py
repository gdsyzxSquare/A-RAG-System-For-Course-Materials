#!/usr/bin/env python3
"""Quick test for RAG pipeline"""

import yaml
import os
from src.vector_store import FAISSVectorStore
from src.embeddings import EmbeddingGenerator, QueryEncoder
from src.retriever import RetrieverFactory
from src.rag_pipeline import RAGPipeline

# Load config
with open('configs/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

print("Loading RAG pipeline...")

# Load vector store
vector_store = FAISSVectorStore(embedding_dim=384)
vector_store.load(
    config['data']['embeddings_path'] + '.index',
    config['data']['embeddings_path'] + '.json'
)

# Create embedding generator and query encoder
embedding_gen = EmbeddingGenerator(
    model_name=config['embedding']['model_name'],
    batch_size=config['embedding']['batch_size']
)
query_encoder = QueryEncoder(embedding_gen)

# Create retriever
retriever = RetrieverFactory.create_retriever(
    method=config['retrieval']['method'],
    vector_store=vector_store,
    query_encoder=query_encoder,
    chunks=vector_store.chunks
)

# Create RAG pipeline
rag_pipeline = RAGPipeline.from_config('configs/config.yaml', retriever)

# Test query
query = "who is the instructor"
print(f"\nQuery: {query}\n")
result = rag_pipeline.query(query, return_context=True)

print(f"Answer: {result['answer']}\n")
print("Retrieved context chunks:")
for i, (chunk, score) in enumerate(zip(result['context'][:3], result['scores'][:3]), 1):
    print(f"\n[{i}] Score: {score:.4f}")
    print(f"    {chunk['text'][:250]}...")

