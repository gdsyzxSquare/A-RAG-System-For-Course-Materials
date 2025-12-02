# Course RAG System

A Retrieval-Augmented Generation (RAG) system for answering questions based on course documents (PPT lectures).

## 📁 Project Structure

```
project/
├── configs/              # Configuration files
│   └── config.yaml      # Main configuration
├── data/                # Data directory
│   ├── raw/            # Raw course documents
│   ├── processed/      # Processed chunks
│   └── embeddings/     # Vector embeddings
├── src/                # Source code
│   ├── data_loader.py  # Data loading and preprocessing
│   ├── chunking.py     # Document chunking strategies
│   ├── embeddings.py   # Embedding generation
│   ├── vector_store.py # FAISS vector store
│   ├── retriever.py    # BM25 and dense retrievers
│   └── rag_pipeline.py # Complete RAG pipeline
├── evaluation/         # Evaluation modules
│   ├── metrics.py      # Evaluation metrics
│   └── evaluation.py   # Evaluation framework
├── notebooks/          # Jupyter notebooks
│   └── 01_rag_pipeline_demo.md
├── results/            # Evaluation results
├── main.py            # Main entry point
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup

```bash
# Copy environment template
copy .env.example .env

# Edit .env and add your DeepSeek API key (recommended)
# DEEPSEEK_API_KEY=sk-your-key-here
# 
# Or use OpenAI API key:
# OPENAI_API_KEY=your-key-here
```

**推荐使用 DeepSeek API**:
- 💰 价格实惠（远低于OpenAI）
- 🇨🇳 中文支持好
- ⚡ 响应速度快
- 📖 查看 `DEEPSEEK_SETUP.md` 获取详细配置指南

### 2.5 Test API Connection

```bash
# Test DeepSeek API
python test_deepseek.py
```

这会验证你的API配置是否正确。

### 3. Prepare Data

Place your course documents (plain text extracted from PPT) in:
```
data/raw/course_documents.txt
```

### 4. Build Index

```bash
python main.py --mode build
```

This will:
- Load and preprocess your documents
- Chunk them into smaller pieces
- Generate embeddings
- Create a FAISS vector index

### 5. Query the System

```bash
python main.py --mode query
```

Interactive mode where you can ask questions about your course materials.

### 6. Evaluate

First, create an evaluation dataset at `evaluation/eval_dataset.json`:

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

Then run evaluation:

```bash
python main.py --mode evaluate
```

## 📊 Project Requirements Checklist

### Core Objectives ✅
- [x] Data cleaning and preprocessing
- [x] Chunking strategies (fixed-size, semantic, sliding window)
- [x] Embedding generation (sentence-transformers)
- [x] Vector index (FAISS)
- [x] Retriever (BM25 + Dense)
- [x] Generator (LLM integration)
- [x] Evaluation framework
  - [ ] Create evaluation dataset (≥50 samples) - **YOUR TASK**
  - [x] Retrieval metrics (Recall@k, MRR)
  - [x] Answer metrics (Exact Match, F1, ROUGE)
  - [ ] LLM-as-Judge (≥30 samples) - **TODO**

### Comparison Experiments 🔄
- [ ] Closed-book vs RAG - **Ready to run after eval dataset**
- [ ] Compare retrievers (BM25 vs Dense vs Hybrid) - **Implemented**
- [ ] Compare prompts - **Modify config.yaml**

### Advanced Features (Choose ≥2) 🎯
- [ ] Query rewriting (HyDE)
- [ ] Re-ranking (Cross-encoder)
- [ ] Latency & memory profiling
- [ ] Your own variant

## 📝 Next Steps

1. **Create evaluation dataset** (Priority 1)
   - Design at least 50 questions from your course materials
   - Include ground truth answers
   - Specify relevant chunk IDs
   - Save to `evaluation/eval_dataset.json`

2. **Run baseline experiments**
   - Build index with different chunking strategies
   - Compare retrievers (modify config.yaml)
   - Test different prompts

3. **Implement advanced features**
   - Choose at least 2 from the list
   - Document your implementation

4. **Analysis and reporting**
   - Generate comparison charts
   - Write findings
   - Document limitations

## 🛠️ Configuration

Edit `configs/config.yaml` to customize:
- Chunking strategy and size
- Embedding model
- Retrieval method
- LLM provider and model
- Prompts

## 📚 Key Files to Understand

- `src/chunking.py` - Different chunking strategies
- `src/retriever.py` - BM25, Dense, and Hybrid retrieval
- `src/rag_pipeline.py` - End-to-end RAG system
- `evaluation/metrics.py` - Evaluation metrics implementation

## ⚠️ Important Notes

1. **Import errors** shown by VS Code are normal - packages will be installed via requirements.txt
2. **API keys** required for LLM generation (OpenAI or Anthropic)
3. **Evaluation dataset** must be created manually based on your course content
4. **At least 50 evaluation samples** required per project requirements

## 📧 Support

For questions about the project requirements, refer to `requirment.txt`.

---

**Remember**: This is a framework. You need to:
1. Add your course documents
2. Create evaluation dataset
3. Run experiments
4. Implement advanced features
5. Analyze and report results

Good luck! 🎓
