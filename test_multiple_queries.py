#!/usr/bin/env python3
"""测试多个查询的效果"""

import yaml
from src.vector_store import FAISSVectorStore
from src.embeddings import EmbeddingGenerator, QueryEncoder
from src.retriever import RetrieverFactory
from src.rag_pipeline import RAGPipeline

# 加载配置
with open('configs/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 加载pipeline
vector_store = FAISSVectorStore(embedding_dim=384)
vector_store.load(
    config['data']['embeddings_path'] + '.index',
    config['data']['embeddings_path'] + '.json'
)

embedding_gen = EmbeddingGenerator(
    model_name=config['embedding']['model_name'],
    batch_size=config['embedding']['batch_size']
)
query_encoder = QueryEncoder(embedding_gen)

retriever = RetrieverFactory.create_retriever(
    method=config['retrieval']['method'],
    vector_store=vector_store,
    query_encoder=query_encoder,
    chunks=vector_store.chunks
)

pipeline = RAGPipeline.from_config('configs/config.yaml', retriever)

# 测试查询
test_queries = [
    "who is the instructor",
    "what is python",
    "when is the final project",
    "how many assignments are there"
]

print("="*70)
print("RAG系统测试 - Query改写已启用")
print("="*70)

for q in test_queries:
    print(f"\n{'='*70}")
    print(f"Query: {q}")
    print('='*70)
    
    result = pipeline.query(q, return_context=True)
    
    print(f"\nAnswer: {result['answer']}\n")
    print("Top 3 retrieved chunks:")
    for i, (chunk, score) in enumerate(zip(result['context'][:3], result['scores'][:3]), 1):
        print(f"  [{i}] Score={score:.4f} | {chunk['text'][:80]}...")
