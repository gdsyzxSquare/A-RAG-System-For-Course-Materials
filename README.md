# Course RAG System

A production-ready Retrieval-Augmented Generation (RAG) system for answering questions based on course documents, featuring metadata extraction, re-ranking, and LLM-as-judge evaluation.

## ✨ Key Features

- 🔍 **Hybrid Retrieval**: Dense embeddings + BM25 fusion
- 🎯 **Re-ranking**: Cross-encoder for improved relevance
- 📊 **Metadata Extraction**: Dedicated chunks for structured info (instructor, location, time, etc.)
- 🔄 **Query Rewriting**: Pattern-based query optimization
- 🤖 **LLM-as-Judge**: Multi-criteria answer quality evaluation
- 📈 **Comprehensive Metrics**: F1, ROUGE, Recall@k, MRR, and more

## 📁 Project Structure

```
project/
├── configs/              # Configuration files
│   └── config.yaml      # Main configuration (supports all features)
├── data/                # Data directory
│   ├── raw/            # Raw course documents
│   └── embeddings/     # Vector embeddings (FAISS index)
├── src/                # Source code
│   ├── data_loader.py  # Data loading and preprocessing
│   ├── chunking.py     # Multiple strategies + metadata extraction
│   ├── embeddings.py   # Embedding generation
│   ├── vector_store.py # FAISS vector store
│   ├── retriever.py    # BM25, Dense, and Hybrid retrievers
│   ├── rag_pipeline.py # Complete RAG pipeline with re-ranker
│   └── query_rewriter.py # Query rewriting strategies
├── evaluation/         # Evaluation modules
|   ├── eval_dataset.json  # Evaluation dataset
│   ├── metrics.py      # Evaluation metrics
│   ├── evaluation.py   # Enhanced evaluation framework
│   └── llm_judge.py    # ✨ LLM-as-Judge implementation
├── results/            # Evaluation results
├── main.py            # Main entry point (supports all modes)
├── requirements.txt   # Python dependencies
├── clean_data         # Clean raw data
└── README.md         # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup

```bash
# Create .env file with your API keys
# Example:
# DASHSCOPE_API_KEY=your_dashscope_key_here
# MOONSHOT_API_KEY=your_moonshot_key_here
```

### 3. Basic Usage

```bash
# Complete pipeline: preprocess → build → query
python clean_data.py # Clean data placed in data/raw/course_documents.txt
python main.py --mode build       # Build vector store with embeddings
python main.py --mode query       # Interactive Q&A mode

# Evaluation with full features (LLM Judge + all metrics)
python main.py --mode evaluate    # Uses config.yaml settings
```

**⚡ Evaluation Tips**:
- Configure evaluation features in `configs/config.yaml`
- Set `run_closed_book: false` to skip closed-book eval (saves ~50% time/cost)
- Configure `llm_judge.criteria` to customize LLM evaluation

### 4. Prepare Data

Place your course documents (plain text extracted from PPT) in:
```
data/raw/course_documents.txt
```
run
```
python clean_data.py
```
to clean data
### 5. Build Index & Query

```bash
# Build vector store from documents
python main.py --mode build

# Query interactively
python main.py --mode query
```

The system will:
- Extract metadata
- Chunk documents
- Generate embeddings and create FAISS index
- Enable hybrid retrieval with re-ranking (optional)

### 6. Evaluation

Create an evaluation dataset at `evaluation/eval_dataset.json`:

```json
[
  {
    "question": "What is the main topic?",
    "answer": "The main topic is...",
    "relevant_chunks": [0, 1, 2],
    "metadata": {"difficulty": "easy"}
  }
]
```

Then run:

```bash
python main.py --mode evaluate
```

This generates a comprehensive evaluation report with:
- 📊 **Standard Metrics**: Recall@k, MRR, F1, ROUGE-L
- 🤖 **LLM-as-Judge** (optional): Multi-criteria quality evaluation  
- 🔄 **Closed-book Comparison** (optional): RAG vs no retrieval

**Tip**: Configure evaluation features in `configs/config.yaml` - see [Configuration](#-configuration) section below.

## 📝 Configuration

The system is configured via `configs/config.yaml`:


## 🤖 LLM-as-Judge Evaluation

The system includes LLM-based answer quality evaluation across multiple criteria.

## 📈 Metadata Extraction

The system automatically extracts structured metadata from course documents:

**Extracted Fields**:
- 📍 Location: Course venue
- 🕐 Time: Class schedule
- 👨‍🏫 Instructor: Name & contact
- 📊 Course Info: Code, title, credits
- 📚 Prerequisites: Required background
- 🎯 Learning Objectives
- 📖 Textbooks

## 📊 Project Requirements Checklist

### Core Objectives ✅
- [x] Data cleaning and preprocessing
- [x] Chunking strategies (fixed-size, semantic, sliding window)
- [x] Metadata extraction (13 metadata chunks)
- [x] Embedding generation (sentence-transformers)
- [x] Vector index (FAISS)
- [x] Retriever (BM25 + Dense + Hybrid)
- [x] Re-ranking (Cross-encoder: 10 → 5 chunks)
- [x] Generator (LLM integration)
- [x] Evaluation framework
  - [x] Retrieval metrics (Recall@k, MRR)
  - [x] Answer metrics (Exact Match, F1, ROUGE)
  - [x] **LLM-as-Judge** (5 criteria, configurable)
  - [x] **Configurable evaluation** (closed-book toggle, criteria selection)

### Comparison Experiments 🔄
- [x] Closed-book vs RAG (**configurable via config.yaml**)
- [x] Compare retrievers (BM25 vs Dense vs Hybrid)
- [x] Compare prompts (modify config.yaml)

### Advanced Features 🎯
- [x] Query rewriting (pattern-based optimization)
- [x] Re-ranking (Cross-encoder: top 10 → top 5)
- [x] Metadata extraction (13 specialized chunks)
- [x] LLM-as-Judge evaluation (multi-criteria)

