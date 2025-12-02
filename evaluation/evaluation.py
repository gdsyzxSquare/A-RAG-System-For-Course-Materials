"""
Evaluation Module for RAG System
Handles comprehensive evaluation of RAG pipeline
"""

import json
import os
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from evaluation.metrics import (
    RetrievalMetrics, 
    AnswerQualityMetrics,
    exact_match,
    f1_score
)


class EvaluationDataset:
    """Manages evaluation dataset"""
    
    def __init__(self, dataset_path: Optional[str] = None):
        """
        Initialize evaluation dataset
        
        Args:
            dataset_path: Path to JSON file with evaluation data
        """
        self.dataset_path = dataset_path
        self.data = []
        
        if dataset_path and os.path.exists(dataset_path):
            self.load(dataset_path)
    
    def load(self, filepath: str):
        """
        Load dataset from JSON file
        
        Args:
            filepath: Path to JSON file
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        print(f"✓ Loaded {len(self.data)} evaluation samples from {filepath}")
    
    def save(self, filepath: str):
        """
        Save dataset to JSON file
        
        Args:
            filepath: Path to save JSON file
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        print(f"✓ Saved {len(self.data)} evaluation samples to {filepath}")
    
    def add_sample(
        self, 
        question: str, 
        answer: str, 
        relevant_chunks: List[Any],
        metadata: Dict[str, Any] = None
    ):
        """
        Add a sample to the dataset
        
        Args:
            question: Question text
            answer: Ground truth answer
            relevant_chunks: List of relevant chunk IDs
            metadata: Optional metadata
        """
        sample = {
            "question": question,
            "answer": answer,
            "relevant_chunks": relevant_chunks,
            "metadata": metadata or {}
        }
        self.data.append(sample)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class RAGEvaluator:
    """Evaluate RAG pipeline performance"""
    
    def __init__(
        self, 
        rag_pipeline,
        eval_dataset: EvaluationDataset
    ):
        """
        Initialize evaluator
        
        Args:
            rag_pipeline: RAG pipeline to evaluate
            eval_dataset: Evaluation dataset
        """
        self.rag_pipeline = rag_pipeline
        self.eval_dataset = eval_dataset
    
    def evaluate_retrieval(self, k_values: List[int] = [1, 3, 5]) -> Dict[str, float]:
        """
        Evaluate retrieval performance
        
        Args:
            k_values: List of k values for Recall@k
            
        Returns:
            Dictionary of retrieval metrics
        """
        print("\n" + "="*60)
        print("EVALUATING RETRIEVAL")
        print("="*60)
        
        # Check if dataset has relevant_chunks annotations
        has_relevant_chunks = all('relevant_chunks' in sample for sample in self.eval_dataset.data)
        
        if not has_relevant_chunks:
            print("⚠ Warning: Dataset does not contain 'relevant_chunks' annotations")
            print("  Skipping retrieval evaluation. Only answer quality will be evaluated.")
            return {}
        
        retrieved_ids_list = []
        relevant_ids_list = []
        
        for sample in tqdm(self.eval_dataset.data, desc="Retrieving"):
            question = sample['question']
            relevant_chunks = sample['relevant_chunks']
            
            # Retrieve chunks
            chunks, _ = self.rag_pipeline.retriever.retrieve(
                question, 
                top_k=max(k_values)
            )
            
            retrieved_ids = [chunk.get('chunk_id', i) for i, chunk in enumerate(chunks)]
            
            retrieved_ids_list.append(retrieved_ids)
            relevant_ids_list.append(relevant_chunks)
        
        # Calculate metrics
        metrics = RetrievalMetrics.evaluate(
            retrieved_ids_list, 
            relevant_ids_list, 
            k_values
        )
        
        print("\nRetrieval Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def evaluate_generation(self) -> Dict[str, float]:
        """
        Evaluate answer generation quality
        
        Returns:
            Dictionary of generation metrics
        """
        print("\n" + "="*60)
        print("EVALUATING GENERATION (RAG)")
        print("="*60)
        
        predictions = []
        ground_truths = []
        
        for sample in tqdm(self.eval_dataset.data, desc="Generating"):
            question = sample['question']
            ground_truth = sample['answer']
            
            # Generate answer with RAG
            result = self.rag_pipeline.query(question)
            prediction = result['answer']
            
            predictions.append(prediction)
            ground_truths.append(ground_truth)
        
        # Calculate metrics
        metrics = AnswerQualityMetrics.evaluate(predictions, ground_truths)
        
        print("\nAnswer Quality Metrics (RAG):")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return metrics, predictions
    
    def evaluate_closed_book(self) -> Dict[str, float]:
        """
        Evaluate closed-book (no RAG) generation
        
        Returns:
            Dictionary of generation metrics
        """
        print("\n" + "="*60)
        print("EVALUATING CLOSED-BOOK GENERATION")
        print("="*60)
        
        predictions = []
        ground_truths = []
        
        for sample in tqdm(self.eval_dataset.data, desc="Generating (closed-book)"):
            question = sample['question']
            ground_truth = sample['answer']
            
            # Generate answer without RAG
            prediction = self.rag_pipeline.closed_book_query(question)
            
            predictions.append(prediction)
            ground_truths.append(ground_truth)
        
        # Calculate metrics
        metrics = AnswerQualityMetrics.evaluate(predictions, ground_truths)
        
        print("\nAnswer Quality Metrics (Closed-book):")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return metrics, predictions
    
    def full_evaluation(self, k_values: List[int] = [1, 3, 5]) -> Dict[str, Any]:
        """
        Run complete evaluation: retrieval + RAG + closed-book
        
        Args:
            k_values: List of k values for Recall@k
            
        Returns:
            Dictionary with all results
        """
        results = {}
        
        # Retrieval evaluation
        results['retrieval'] = self.evaluate_retrieval(k_values)
        
        # RAG generation evaluation
        rag_metrics, rag_predictions = self.evaluate_generation()
        results['rag'] = rag_metrics
        results['rag_predictions'] = rag_predictions
        
        # Closed-book evaluation
        cb_metrics, cb_predictions = self.evaluate_closed_book()
        results['closed_book'] = cb_metrics
        results['closed_book_predictions'] = cb_predictions
        
        # Summary comparison
        print("\n" + "="*60)
        print("SUMMARY: RAG vs Closed-Book")
        print("="*60)
        print(f"{'Metric':<20} {'RAG':>15} {'Closed-Book':>15} {'Improvement':>15}")
        print("-"*60)
        
        for metric in ['exact_match', 'f1', 'rouge1', 'rougeL']:
            rag_val = rag_metrics[metric]
            cb_val = cb_metrics[metric]
            improvement = ((rag_val - cb_val) / cb_val * 100) if cb_val > 0 else 0
            print(f"{metric:<20} {rag_val:>15.4f} {cb_val:>15.4f} {improvement:>14.2f}%")
        
        return results
    
    def save_results(self, results: Dict[str, Any], filepath: str):
        """
        Save evaluation results to JSON
        
        Args:
            results: Evaluation results dictionary
            filepath: Path to save results
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Remove predictions from saved results (too large)
        save_data = {
            'retrieval': results['retrieval'],
            'rag': results['rag'],
            'closed_book': results['closed_book']
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\n✓ Saved evaluation results to {filepath}")


if __name__ == "__main__":
    print("RAG Evaluation Module")
    print("\nExample evaluation dataset format:")
    
    example_dataset = [
        {
            "question": "What is Python?",
            "answer": "Python is a high-level programming language.",
            "relevant_chunks": [0, 5, 12],
            "metadata": {"difficulty": "easy", "topic": "programming"}
        }
    ]
    
    print(json.dumps(example_dataset, indent=2, ensure_ascii=False))
