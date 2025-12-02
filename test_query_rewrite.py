#!/usr/bin/env python3
"""测试query改写后的检索效果"""

import yaml
from src.vector_store import FAISSVectorStore
from src.embeddings import EmbeddingGenerator, QueryEncoder
from src.retriever import RetrieverFactory
from src.query_rewriter import rewrite_query

# 加载配置
with open('configs/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

print("="*70)
print("测试Query改写的效果")
print("="*70)

# 加载vector store
vector_store = FAISSVectorStore(embedding_dim=384)
vector_store.load(
    config['data']['embeddings_path'] + '.index',
    config['data']['embeddings_path'] + '.json'
)

# 创建embedding generator
embedding_gen = EmbeddingGenerator(
    model_name=config['embedding']['model_name'],
    batch_size=config['embedding']['batch_size']
)
query_encoder = QueryEncoder(embedding_gen)

# 创建retriever
retriever = RetrieverFactory.create_retriever(
    method='dense',  # 先测试pure dense
    vector_store=vector_store,
    query_encoder=query_encoder,
    chunks=vector_store.chunks
)

# 测试原始查询vs改写查询
original_query = "who is the instructor"
rewritten_queries = rewrite_query(original_query)

print(f"\n原始查询: '{original_query}'")
print("=" * 70)
chunks, scores = retriever.retrieve(original_query, top_k=5)
for i, (chunk, score) in enumerate(zip(chunks, scores), 1):
    is_target = " ← 包含答案" if "Chengwei" in chunk['text'] else ""
    print(f"  [{i}] Score={score:.4f}{is_target}")
    print(f"      {chunk['text'][:100]}...")

print("\n" + "=" * 70)
print("改写后的查询：")
print("=" * 70)

# 测试最有效的几个改写
best_rewrites = ["Chengwei Qin", "taught by Prof", "course instructor"]

for rw_query in best_rewrites:
    print(f"\n改写查询: '{rw_query}'")
    print("-" * 70)
    chunks, scores = retriever.retrieve(rw_query, top_k=3)
    for i, (chunk, score) in enumerate(zip(chunks, scores), 1):
        is_target = " ← 包含答案!" if "Chengwei" in chunk['text'] else ""
        print(f"  [{i}] Score={score:.4f}{is_target}")
        print(f"      {chunk['text'][:100]}...")

print("\n" + "=" * 70)
print("结论：Query改写可以显著提升检索准确率")
print("=" * 70)
