"""
Evaluation Metrics Module
Implements various metrics for RAG system evaluation
"""

import re
import string
from typing import List, Dict, Any, Tuple
from collections import Counter
import numpy as np


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


def exact_match(prediction: str, ground_truth: str) -> float:
    """
    Calculate exact match score
    
    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer
        
    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    pred_norm = normalize_text(prediction)
    gt_norm = normalize_text(ground_truth)
    
    return 1.0 if pred_norm == gt_norm else 0.0


def f1_score(prediction: str, ground_truth: str) -> float:
    """
    Calculate token-level F1 score
    
    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer
        
    Returns:
        F1 score
    """
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()
    
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0
    
    # Count token frequencies
    pred_counter = Counter(pred_tokens)
    gt_counter = Counter(gt_tokens)
    
    # Calculate overlap
    common = pred_counter & gt_counter
    num_common = sum(common.values())
    
    if num_common == 0:
        return 0.0
    
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)
    
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1


def recall_at_k(
    retrieved_ids: List[Any], 
    relevant_ids: List[Any], 
    k: int
) -> float:
    """
    Calculate Recall@k
    
    Args:
        retrieved_ids: List of retrieved document IDs
        relevant_ids: List of relevant document IDs
        k: Cutoff position
        
    Returns:
        Recall@k score
    """
    if not relevant_ids:
        return 0.0
    
    retrieved_at_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    
    hits = len(retrieved_at_k & relevant_set)
    recall = hits / len(relevant_set)
    
    return recall


def mean_reciprocal_rank(
    retrieved_ids_list: List[List[Any]], 
    relevant_ids_list: List[List[Any]]
) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR)
    
    Args:
        retrieved_ids_list: List of retrieved document ID lists
        relevant_ids_list: List of relevant document ID lists
        
    Returns:
        MRR score
    """
    reciprocal_ranks = []
    
    for retrieved_ids, relevant_ids in zip(retrieved_ids_list, relevant_ids_list):
        relevant_set = set(relevant_ids)
        
        for i, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_set:
                reciprocal_ranks.append(1.0 / i)
                break
        else:
            reciprocal_ranks.append(0.0)
    
    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0


def calculate_rouge_scores(prediction: str, ground_truth: str) -> Dict[str, float]:
    """
    Calculate ROUGE scores
    
    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer
        
    Returns:
        Dictionary of ROUGE scores
    """
    try:
        from rouge_score import rouge_scorer
        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(ground_truth, prediction)
        
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    except ImportError:
        print("Warning: rouge-score not installed. Install with: pip install rouge-score")
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}


class RetrievalMetrics:
    """Calculate retrieval metrics"""
    
    @staticmethod
    def evaluate(
        retrieved_ids_list: List[List[Any]],
        relevant_ids_list: List[List[Any]],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, float]:
        """
        Evaluate retrieval performance
        
        Args:
            retrieved_ids_list: List of retrieved document ID lists
            relevant_ids_list: List of relevant document ID lists
            k_values: List of k values for Recall@k
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Calculate Recall@k for each k
        for k in k_values:
            recalls = []
            for retrieved_ids, relevant_ids in zip(retrieved_ids_list, relevant_ids_list):
                recalls.append(recall_at_k(retrieved_ids, relevant_ids, k))
            metrics[f'recall@{k}'] = np.mean(recalls)
        
        # Calculate MRR
        metrics['mrr'] = mean_reciprocal_rank(retrieved_ids_list, relevant_ids_list)
        
        return metrics


class AnswerQualityMetrics:
    """Calculate answer quality metrics"""
    
    @staticmethod
    def evaluate(
        predictions: List[str],
        ground_truths: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate answer quality
        
        Args:
            predictions: List of predicted answers
            ground_truths: List of ground truth answers
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'exact_match': [],
            'f1': [],
            'rouge1': [],
            'rouge2': [],
            'rougeL': []
        }
        
        for pred, gt in zip(predictions, ground_truths):
            # Exact Match
            metrics['exact_match'].append(exact_match(pred, gt))
            
            # F1 Score
            metrics['f1'].append(f1_score(pred, gt))
            
            # ROUGE Scores
            rouge_scores = calculate_rouge_scores(pred, gt)
            metrics['rouge1'].append(rouge_scores['rouge1'])
            metrics['rouge2'].append(rouge_scores['rouge2'])
            metrics['rougeL'].append(rouge_scores['rougeL'])
        
        # Average all metrics
        return {k: np.mean(v) for k, v in metrics.items()}


if __name__ == "__main__":
    # Example usage
    
    # Test answer quality metrics
    pred = "The capital of France is Paris."
    gt = "Paris is the capital of France."
    
    em = exact_match(pred, gt)
    f1 = f1_score(pred, gt)
    rouge = calculate_rouge_scores(pred, gt)
    
    print("Answer Quality Metrics:")
    print(f"  Exact Match: {em:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  ROUGE-1: {rouge['rouge1']:.4f}")
    print(f"  ROUGE-L: {rouge['rougeL']:.4f}")
    
    # Test retrieval metrics
    retrieved = [['doc1', 'doc2', 'doc3', 'doc4', 'doc5']]
    relevant = [['doc2', 'doc5']]
    
    ret_metrics = RetrievalMetrics.evaluate(retrieved, relevant, k_values=[1, 3, 5])
    print("\nRetrieval Metrics:")
    for metric, value in ret_metrics.items():
        print(f"  {metric}: {value:.4f}")
