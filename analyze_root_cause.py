#!/usr/bin/env python3
"""深度分析检索失败的底层原因"""

import yaml
import numpy as np
from src.vector_store import FAISSVectorStore
from src.embeddings import EmbeddingGenerator, QueryEncoder

# 加载配置
with open('configs/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

print("="*70)
print("底层原因分析：为什么 'who is instructor' 检索失败")
print("="*70)

# 1. 加载chunks
print("\n[1] 加载chunks并查找目标信息...")
vector_store = FAISSVectorStore(embedding_dim=384)
vector_store.load(
    config['data']['embeddings_path'] + '.index',
    config['data']['embeddings_path'] + '.json'
)

target_chunk = None
for i, chunk in enumerate(vector_store.chunks):
    if 'Chengwei' in chunk['text']:
        target_chunk = (i, chunk)
        print(f"✓ 找到目标chunk #{i}")
        print(f"  内容: {chunk['text'][:200]}...")
        break

# 2. 生成查询embedding
print("\n[2] 生成query embedding...")
embedding_gen = EmbeddingGenerator(
    model_name=config['embedding']['model_name'],
    batch_size=config['embedding']['batch_size']
)

queries = [
    "who is the instructor",
    "who is the teacher",  
    "who is Prof. Chengwei Qin",
    "course instructor name",
    "This section is taught by"
]

query_embeddings = []
for q in queries:
    emb = embedding_gen.encode(q, show_progress=False)
    query_embeddings.append(emb)
    print(f"  ✓ {q}")

# 3. 使用实际的检索器进行测试
print("\n[3] 使用Dense检索器测试不同query的排名...")
query_encoder = QueryEncoder(embedding_gen)

for q in queries:
    print(f"\n  Query: '{q}'")
    q_emb = embedding_gen.encode(q, show_progress=False)
    q_emb = q_emb.astype('float32').reshape(1, -1)
    
    distances, indices = vector_store.index.search(q_emb, k=5)
    
    target_rank = None
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
        is_target = " ← 目标chunk" if idx == target_chunk[0] else ""
        # FAISS使用L2距离，转换为相似度分数
        similarity = 1 / (1 + dist)
        print(f"    [{rank}] Chunk #{idx}: L2={dist:.4f}, Sim={similarity:.4f}{is_target}")
        if idx == target_chunk[0]:
            target_rank = rank
    
    if target_rank:
        print(f"    ✓ 目标chunk排在第{target_rank}名")
    else:
        print(f"    ✗ 目标chunk未进入Top 5")

# 4. 词汇分析
print("\n[4] 词汇级别分析...")
query_words = set("who is the instructor".lower().split())
target_text = target_chunk[1]['text'].lower()
target_words = set(target_text.split())

common_words = query_words & target_words
print(f"  Query词汇: {query_words}")
print(f"  Query与Target共同词: {common_words}")
print(f"  词汇重叠率: {len(common_words)}/{len(query_words)} = {len(common_words)/len(query_words)*100:.1f}%")

# 检查关键词
key_terms = ['instructor', 'teacher', 'professor', 'taught', 'prof']
print(f"\n  Target chunk中的关键词:")
for term in key_terms:
    if term in target_text:
        print(f"    ✓ '{term}' 存在")
    else:
        print(f"    ✗ '{term}' 不存在")

print("\n" + "="*70)
print("结论")
print("="*70)
print("""
问题的底层原因：

1. **语义鸿沟** (Semantic Gap):
   - Query: "who is the instructor" (询问身份)
   - Target: "This section is taught by Prof. Chengwei QIN" (陈述事实)
   - 句式结构完全不同：疑问句 vs 陈述句

2. **词汇不匹配** (Lexical Mismatch):
   - Query中的 "instructor" 在文档中从未出现
   - 文档用的是 "taught by Prof."，不是 "instructor is"

3. **Embedding模型局限** (Model Limitation):
   - all-MiniLM-L6-v2是通用模型，未针对问答场景微调
   - 在问答对(question-answer)的语义匹配上表现较弱
   - 更适合句子相似度判断，而非Q-A配对

4. **竞争chunks干扰** (Competition):
   - 其他chunks包含更多query中的词汇("is", "the"等高频词)
   - 或者主题词重叠更多(如教学相关内容)

解决方案优先级：
  [高] 使用问答专用的embedding模型 (如 sentence-transformers/multi-qa-MiniLM-L6-cos-v1)
  [中] Query改写：将疑问句转为陈述句匹配
  [中] 增加BM25权重：利用词汇匹配
  [低] 增加top_k（治标不治本）
""")
