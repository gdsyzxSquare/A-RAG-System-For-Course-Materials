# Course RAG System - Example Notebook

This notebook demonstrates the complete RAG pipeline from data loading to evaluation.

## Setup

```python
import sys
import os

# Add src to path
sys.path.append(os.path.abspath('..'))

from src.data_loader import DataLoader, save_documents, load_documents
from src.chunking import ChunkerFactory
from src.embeddings import EmbeddingGenerator, QueryEncoder
from src.vector_store import FAISSVectorStore
from src.retriever import RetrieverFactory
from src.rag_pipeline import RAGPipeline
from evaluation.evaluation import EvaluationDataset, RAGEvaluator
import yaml
```

## 1. Load and Preprocess Data

```python
# Load configuration
with open('../configs/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Load raw data
loader = DataLoader(config['data']['raw_data_path'])
text = loader.load_and_preprocess()

print(f"Loaded {len(text)} characters")
print(f"\nFirst 500 characters:\n{text[:500]}")
```

## 2. Chunk the Text

```python
# Create chunker
chunker = ChunkerFactory.create_chunker(
    strategy=config['chunking']['strategy'],
    chunk_size=config['chunking']['chunk_size'],
    overlap=config['chunking']['chunk_overlap']
)

# Chunk the text
chunks = chunker.chunk(text)

print(f"Created {len(chunks)} chunks")
print(f"\nExample chunk:")
print(chunks[0].text[:200])
```

## 3. Generate Embeddings

```python
# Initialize embedding generator
embedding_gen = EmbeddingGenerator(
    model_name=config['embedding']['model_name'],
    batch_size=config['embedding']['batch_size']
)

# Convert chunks to dictionaries
chunk_dicts = [chunk.to_dict() for chunk in chunks]

# Generate embeddings
embeddings = embedding_gen.encode_chunks(chunk_dicts)

print(f"Generated embeddings with shape: {embeddings.shape}")
```

## 4. Create Vector Store

```python
# Create FAISS index
vector_store = FAISSVectorStore(embedding_dim=embeddings.shape[1])
vector_store.create_index("flat")

# Add embeddings
vector_store.add_embeddings(embeddings, chunk_dicts)

# Save for later use
vector_store.save(
    config['data']['embeddings_path'] + '.index',
    config['data']['embeddings_path'] + '.json'
)

print("Vector store created and saved")
```

## 5. Initialize Retriever

```python
# Create query encoder
query_encoder = QueryEncoder(embedding_gen)

# Create retriever
retriever = RetrieverFactory.create_retriever(
    method=config['retrieval']['method'],
    vector_store=vector_store,
    query_encoder=query_encoder,
    chunks=chunk_dicts
)

print(f"Initialized {config['retrieval']['method']} retriever")
```

## 6. Test Retrieval

```python
# Test query
test_query = "What topics are covered in this course?"

chunks, scores = retriever.retrieve(test_query, top_k=3)

print(f"Query: {test_query}\n")
print("Top 3 retrieved chunks:")
for i, (chunk, score) in enumerate(zip(chunks, scores), 1):
    print(f"\n[{i}] Score: {score:.4f}")
    print(f"Text: {chunk['text'][:200]}...")
```

## 7. Create RAG Pipeline

```python
# Initialize RAG pipeline from config
rag_pipeline = RAGPipeline.from_config(
    '../configs/config.yaml',
    retriever
)

print("RAG Pipeline initialized")
```

## 8. Query the RAG System

```python
# Ask a question
question = "What is the main topic of this course?"

result = rag_pipeline.query(question, return_context=True)

print(f"Question: {question}\n")
print(f"Answer: {result['answer']}\n")
print(f"\nRetrieved {len(result['context'])} context chunks")
```

## 9. Compare RAG vs Closed-Book

```python
# RAG answer
rag_answer = rag_pipeline.query(question)['answer']

# Closed-book answer
cb_answer = rag_pipeline.closed_book_query(question)

print(f"Question: {question}\n")
print(f"RAG Answer:\n{rag_answer}\n")
print(f"\nClosed-Book Answer:\n{cb_answer}")
```

## 10. Create Evaluation Dataset

```python
# Create evaluation dataset
eval_dataset = EvaluationDataset()

# Add sample questions (you should create at least 50)
eval_dataset.add_sample(
    question="What is the first topic covered?",
    answer="Introduction to the course concepts",
    relevant_chunks=[0, 1, 2]
)

eval_dataset.add_sample(
    question="What are the learning objectives?",
    answer="Students will learn key concepts and practical applications",
    relevant_chunks=[3, 4]
)

# Save dataset
eval_dataset.save(config['data']['eval_dataset_path'])

print(f"Created evaluation dataset with {len(eval_dataset)} samples")
```

## 11. Run Evaluation

```python
# Load evaluation dataset
eval_dataset = EvaluationDataset(config['data']['eval_dataset_path'])

# Create evaluator
evaluator = RAGEvaluator(rag_pipeline, eval_dataset)

# Run full evaluation
results = evaluator.full_evaluation(k_values=[1, 3, 5])

# Save results
evaluator.save_results(results, '../results/baseline_results.json')
```

## 12. Visualize Results

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Plot comparison
metrics = ['exact_match', 'f1', 'rouge1', 'rougeL']
rag_scores = [results['rag'][m] for m in metrics]
cb_scores = [results['closed_book'][m] for m in metrics]

x = range(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar([i - width/2 for i in x], rag_scores, width, label='RAG', alpha=0.8)
ax.bar([i + width/2 for i in x], cb_scores, width, label='Closed-Book', alpha=0.8)

ax.set_xlabel('Metrics')
ax.set_ylabel('Scores')
ax.set_title('RAG vs Closed-Book Performance')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('../results/rag_vs_closedbook.png', dpi=300, bbox_inches='tight')
plt.show()

print("Visualization saved to results/rag_vs_closedbook.png")
```

## Next Steps

1. **Create more evaluation questions** - Aim for at least 50 questions covering different topics and difficulty levels
2. **Experiment with chunking strategies** - Try semantic chunking vs fixed-size
3. **Compare retrievers** - Test BM25 vs Dense vs Hybrid
4. **Optimize prompts** - Experiment with different prompt templates
5. **Implement advanced features** - Query rewriting, re-ranking, etc.

## Save Your Work

```python
# Make sure to save important artifacts
print("Important files created:")
print(f"  - Vector store: {config['data']['embeddings_path']}")
print(f"  - Evaluation dataset: {config['data']['eval_dataset_path']}")
print(f"  - Results: ../results/baseline_results.json")
```
