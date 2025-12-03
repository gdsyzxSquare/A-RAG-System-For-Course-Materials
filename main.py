"""
Main script to run the complete RAG pipeline
Usage: python main.py --mode [build|query|evaluate]
"""

import argparse
import yaml
import os
from pathlib import Path

from src.data_loader import DataLoader
from src.chunking import ChunkerFactory
from src.embeddings import EmbeddingGenerator, QueryEncoder
from src.vector_store import FAISSVectorStore
from src.retriever import RetrieverFactory
from src.rag_pipeline import RAGPipeline
from evaluation.evaluation import EvaluationDataset, RAGEvaluator


def load_config(config_path: str = "configs/config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_index(config):
    """Build vector index from raw data"""
    print("\n" + "="*60)
    print("BUILDING INDEX")
    print("="*60)
    
    # 1. Load data
    print("\n[1/5] Loading data...")
    loader = DataLoader(config['data']['raw_data_path'])
    text = loader.load_and_preprocess()
    
    # 2. Chunk text
    print("\n[2/5] Chunking text...")
    chunker = ChunkerFactory.create_chunker(
        strategy=config['chunking']['strategy'],
        chunk_size=config['chunking']['chunk_size'],
        overlap=config['chunking']['chunk_overlap'],
        extract_metadata=config['chunking'].get('extract_metadata', True)
    )
    chunks = chunker.chunk(text)
    chunk_dicts = [chunk.to_dict() for chunk in chunks]
    
    # 3. Generate embeddings
    print("\n[3/5] Generating embeddings...")
    embedding_gen = EmbeddingGenerator(
        model_name=config['embedding']['model_name'],
        batch_size=config['embedding']['batch_size']
    )
    embeddings = embedding_gen.encode_chunks(chunk_dicts)
    
    # 4. Create vector store
    print("\n[4/5] Creating vector store...")
    vector_store = FAISSVectorStore(embedding_dim=embeddings.shape[1])
    vector_store.create_index("flat")
    vector_store.add_embeddings(embeddings, chunk_dicts)
    
    # 5. Save
    print("\n[5/5] Saving index...")
    os.makedirs(os.path.dirname(config['data']['embeddings_path']), exist_ok=True)
    vector_store.save(
        config['data']['embeddings_path'] + '.index',
        config['data']['embeddings_path'] + '.json'
    )
    
    print("\n✓ Index built successfully!")
    print(f"  - Total chunks: {len(chunks)}")
    print(f"  - Embedding dimension: {embeddings.shape[1]}")


def load_rag_pipeline(config):
    """Load RAG pipeline with saved index"""
    # Load vector store
    vector_store = FAISSVectorStore(embedding_dim=384)  # Default dimension
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
    rag_pipeline = RAGPipeline.from_config(config['config_path'], retriever)
    
    return rag_pipeline


def query_mode(config):
    """Interactive query mode"""
    print("\n" + "="*60)
    print("RAG QUERY MODE")
    print("="*60)
    print("Type 'quit' to exit\n")
    
    # Load pipeline
    rag_pipeline = load_rag_pipeline(config)
    
    while True:
        question = input("\nQuestion: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not question:
            continue
        
        # Get answer
        result = rag_pipeline.query(question, return_context=True)
        
        print(f"\nAnswer: {result['answer']}")
        print(f"\nRetrieved {len(result['context'])} context chunks")
        print("\nTop 3 chunks:")
        for i, (chunk, score) in enumerate(zip(result['context'][:3], result['scores'][:3]), 1):
            print(f"  [{i}] Score: {score:.4f} | {chunk['text'][:100]}...")


def evaluate_mode(config):
    """Run evaluation"""
    print("\n" + "="*60)
    print("EVALUATION MODE")
    print("="*60)
    
    # Load pipeline
    rag_pipeline = load_rag_pipeline(config)
    
    # Load evaluation dataset
    eval_dataset = EvaluationDataset(config['data']['eval_dataset_path'])
    
    if len(eval_dataset) == 0:
        print("\n⚠ Warning: Evaluation dataset is empty!")
        print(f"Please create evaluation dataset at: {config['data']['eval_dataset_path']}")
        return
    
    # Get evaluation configuration
    eval_config = config.get('evaluation', {})
    use_llm_judge = eval_config.get('llm_judge', {}).get('enabled', False)
    llm_judge_criteria = eval_config.get('llm_judge', {}).get('criteria', ['relevance', 'correctness', 'overall'])
    run_closed_book = eval_config.get('run_closed_book', True)
    
    # Print configuration
    print(f"\nEvaluation Configuration:")
    print(f"  - Dataset samples: {len(eval_dataset)}")
    print(f"  - LLM Judge: {'✓ Enabled' if use_llm_judge else '✗ Disabled'}")
    if use_llm_judge:
        print(f"    Criteria: {', '.join(llm_judge_criteria)}")
    print(f"  - Closed-book: {'✓ Enabled' if run_closed_book else '✗ Disabled'}")
    
    # Run evaluation
    evaluator = RAGEvaluator(
        rag_pipeline, 
        eval_dataset,
        use_llm_judge=use_llm_judge,
        llm_judge_criteria=llm_judge_criteria,
        run_closed_book=run_closed_book
    )
    results = evaluator.full_evaluation(k_values=eval_config.get('recall_at', [1, 3, 5]))
    
    # Save results
    os.makedirs(config['experiment']['results_dir'], exist_ok=True)
    results_path = os.path.join(
        config['experiment']['results_dir'], 
        f"{config['experiment']['name']}_results.json"
    )
    evaluator.save_results(results, results_path)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"\n📊 Results Summary:")
    print(f"  RAG F1 Score: {results['rag'].get('f1', 0):.3f}")
    print(f"  RAG Exact Match: {results['rag'].get('exact_match', 0):.3f}")
    
    if run_closed_book and results.get('closed_book'):
        cb_f1 = results['closed_book'].get('f1', 0)
        print(f"  Closed-Book F1: {cb_f1:.3f}")
        if cb_f1 > 0:
            improvement = ((results['rag']['f1'] - cb_f1) / cb_f1 * 100)
            print(f"  Improvement: {improvement:.1f}%")
    
    if use_llm_judge and 'llm_judge' in results:
        print(f"\n🤖 LLM Judge Scores:")
        for criterion, score in results['llm_judge']['aggregate_scores'].items():
            print(f"  {criterion.capitalize()}: {score:.2f}/5.0")
    
    print(f"\n📁 Results saved to: {results_path}")
    if use_llm_judge and 'llm_judge' in results:
        detailed_path = results_path.replace('.json', '_llm_judge_detailed.json')
        print(f"📁 Detailed LLM Judge results: {detailed_path}")


def main():
    parser = argparse.ArgumentParser(description="Course RAG System")
    parser.add_argument(
        '--mode', 
        type=str, 
        required=True,
        choices=['build', 'query', 'evaluate'],
        help='Operation mode: build (create index), query (interactive), or evaluate'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    config['config_path'] = args.config  # Store config path for later use
    
    # Execute based on mode
    if args.mode == 'build':
        build_index(config)
    elif args.mode == 'query':
        query_mode(config)
    elif args.mode == 'evaluate':
        evaluate_mode(config)


if __name__ == "__main__":
    main()
