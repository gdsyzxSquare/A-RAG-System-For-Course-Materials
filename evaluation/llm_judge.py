"""
LLM-as-Judge Module
Use LLM to evaluate answer quality with detailed scoring criteria
"""

from typing import Dict, Any, List, Optional
from tqdm import tqdm
import json


class LLMJudge:
    """Use LLM to judge answer quality"""
    
    # Scoring prompts for different criteria
    FAITHFULNESS_PROMPT = """You are an expert evaluator. Assess if the answer is faithful to the provided context.

Context:
{context}

Question: {question}

Answer: {answer}

Evaluate FAITHFULNESS (does the answer only use information from the context?):
- Score 5: Completely faithful, all information is from context
- Score 4: Mostly faithful, minor unsupported details
- Score 3: Partially faithful, some unsupported claims
- Score 2: Mostly unfaithful, significant unsupported information
- Score 1: Completely unfaithful, contradicts or ignores context

Respond with ONLY a JSON object:
{{"score": <1-5>, "reasoning": "<brief explanation>"}}"""

    RELEVANCE_PROMPT = """You are an expert evaluator. Assess if the answer is relevant to the question.

Question: {question}

Answer: {answer}

Evaluate RELEVANCE (does the answer address the question?):
- Score 5: Perfectly relevant, directly answers the question
- Score 4: Mostly relevant, minor off-topic details
- Score 3: Partially relevant, misses key aspects
- Score 2: Barely relevant, mostly off-topic
- Score 1: Completely irrelevant

Respond with ONLY a JSON object:
{{"score": <1-5>, "reasoning": "<brief explanation>"}}"""

    COMPLETENESS_PROMPT = """You are an expert evaluator. Assess if the answer is complete.

Question: {question}

Ground Truth Answer: {ground_truth}

Generated Answer: {answer}

Evaluate COMPLETENESS (does the answer cover all important aspects?):
- Score 5: Complete, covers all key points
- Score 4: Mostly complete, minor omissions
- Score 3: Partially complete, missing important details
- Score 2: Incomplete, missing major information
- Score 1: Severely incomplete

Respond with ONLY a JSON object:
{{"score": <1-5>, "reasoning": "<brief explanation>"}}"""

    CORRECTNESS_PROMPT = """You are an expert evaluator. Assess if the answer is factually correct.

Question: {question}

Ground Truth Answer: {ground_truth}

Generated Answer: {answer}

Evaluate CORRECTNESS (is the answer factually accurate?):
- Score 5: Completely correct
- Score 4: Mostly correct, minor errors
- Score 3: Partially correct, some errors
- Score 2: Mostly incorrect
- Score 1: Completely incorrect

Respond with ONLY a JSON object:
{{"score": <1-5>, "reasoning": "<brief explanation>"}}"""

    OVERALL_PROMPT = """You are an expert evaluator. Provide an overall quality assessment.

Question: {question}

Ground Truth Answer: {ground_truth}

Generated Answer: {answer}

Provide an OVERALL QUALITY score:
- Score 5: Excellent answer, accurate and complete
- Score 4: Good answer, minor issues
- Score 3: Acceptable answer, noticeable gaps
- Score 2: Poor answer, significant problems
- Score 1: Very poor answer, fails to address question

Respond with ONLY a JSON object:
{{"score": <1-5>, "reasoning": "<brief explanation>"}}"""

    def __init__(self, llm_generator):
        """
        Initialize LLM Judge
        
        Args:
            llm_generator: LLMGenerator instance for evaluation
        """
        self.llm = llm_generator
        print("✓ LLM Judge initialized")
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response to extract score and reasoning
        
        Args:
            response: LLM response text
            
        Returns:
            Dictionary with score and reasoning
        """
        try:
            # Try to parse as JSON
            response = response.strip()
            
            # Find JSON object in response
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start >= 0 and end > start:
                json_str = response[start:end]
                result = json.loads(json_str)
                return {
                    'score': result.get('score', 0),
                    'reasoning': result.get('reasoning', '')
                }
        except Exception as e:
            print(f"Warning: Failed to parse LLM response: {e}")
            print(f"Response: {response[:200]}")
        
        # Fallback: try to extract score from text
        import re
        score_match = re.search(r'score["\s:]+(\d)', response.lower())
        if score_match:
            return {
                'score': int(score_match.group(1)),
                'reasoning': 'Failed to parse full response'
            }
        
        return {'score': 0, 'reasoning': 'Failed to parse response'}
    
    def evaluate_faithfulness(
        self, 
        question: str, 
        answer: str, 
        context: str
    ) -> Dict[str, Any]:
        """
        Evaluate faithfulness to context
        
        Args:
            question: Question text
            answer: Generated answer
            context: Retrieved context
            
        Returns:
            Score and reasoning
        """
        prompt = self.FAITHFULNESS_PROMPT.format(
            question=question,
            answer=answer,
            context=context
        )
        response = self.llm.generate(prompt)
        return self._parse_llm_response(response)
    
    def evaluate_relevance(
        self, 
        question: str, 
        answer: str
    ) -> Dict[str, Any]:
        """
        Evaluate relevance to question
        
        Args:
            question: Question text
            answer: Generated answer
            
        Returns:
            Score and reasoning
        """
        prompt = self.RELEVANCE_PROMPT.format(
            question=question,
            answer=answer
        )
        response = self.llm.generate(prompt)
        return self._parse_llm_response(response)
    
    def evaluate_completeness(
        self, 
        question: str, 
        answer: str, 
        ground_truth: str
    ) -> Dict[str, Any]:
        """
        Evaluate completeness compared to ground truth
        
        Args:
            question: Question text
            answer: Generated answer
            ground_truth: Ground truth answer
            
        Returns:
            Score and reasoning
        """
        prompt = self.COMPLETENESS_PROMPT.format(
            question=question,
            answer=answer,
            ground_truth=ground_truth
        )
        response = self.llm.generate(prompt)
        return self._parse_llm_response(response)
    
    def evaluate_correctness(
        self, 
        question: str, 
        answer: str, 
        ground_truth: str
    ) -> Dict[str, Any]:
        """
        Evaluate factual correctness
        
        Args:
            question: Question text
            answer: Generated answer
            ground_truth: Ground truth answer
            
        Returns:
            Score and reasoning
        """
        prompt = self.CORRECTNESS_PROMPT.format(
            question=question,
            answer=answer,
            ground_truth=ground_truth
        )
        response = self.llm.generate(prompt)
        return self._parse_llm_response(response)
    
    def evaluate_overall(
        self, 
        question: str, 
        answer: str, 
        ground_truth: str
    ) -> Dict[str, Any]:
        """
        Evaluate overall quality
        
        Args:
            question: Question text
            answer: Generated answer
            ground_truth: Ground truth answer
            
        Returns:
            Score and reasoning
        """
        prompt = self.OVERALL_PROMPT.format(
            question=question,
            answer=answer,
            ground_truth=ground_truth
        )
        response = self.llm.generate(prompt)
        return self._parse_llm_response(response)
    
    def evaluate_single(
        self,
        question: str,
        answer: str,
        ground_truth: str,
        context: Optional[str] = None,
        criteria: List[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single Q&A pair on multiple criteria
        
        Args:
            question: Question text
            answer: Generated answer
            ground_truth: Ground truth answer
            context: Retrieved context (optional)
            criteria: List of criteria to evaluate
                     Options: ['faithfulness', 'relevance', 'completeness', 'correctness', 'overall']
                     Default: ['relevance', 'correctness', 'overall']
        
        Returns:
            Dictionary with scores for each criterion
        """
        if criteria is None:
            criteria = ['relevance', 'correctness', 'overall']
        
        results = {}
        
        if 'faithfulness' in criteria and context:
            results['faithfulness'] = self.evaluate_faithfulness(question, answer, context)
        
        if 'relevance' in criteria:
            results['relevance'] = self.evaluate_relevance(question, answer)
        
        if 'completeness' in criteria:
            results['completeness'] = self.evaluate_completeness(question, answer, ground_truth)
        
        if 'correctness' in criteria:
            results['correctness'] = self.evaluate_correctness(question, answer, ground_truth)
        
        if 'overall' in criteria:
            results['overall'] = self.evaluate_overall(question, answer, ground_truth)
        
        return results
    
    def evaluate_batch(
        self,
        samples: List[Dict[str, Any]],
        criteria: List[str] = None,
        include_context: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of Q&A pairs
        
        Args:
            samples: List of samples, each containing:
                    - question: str
                    - answer: str (generated)
                    - ground_truth: str
                    - context: str (optional, if include_context=True)
            criteria: List of criteria to evaluate
            include_context: Whether context is provided for faithfulness evaluation
        
        Returns:
            Dictionary with:
            - detailed_results: List of per-sample results
            - aggregate_scores: Average scores for each criterion
        """
        if criteria is None:
            criteria = ['relevance', 'correctness', 'overall']
            if include_context:
                criteria.insert(0, 'faithfulness')
        
        print(f"\n{'='*60}")
        print(f"LLM-AS-JUDGE EVALUATION")
        print(f"{'='*60}")
        print(f"Criteria: {', '.join(criteria)}")
        print(f"Samples: {len(samples)}")
        print()
        
        detailed_results = []
        aggregate_scores = {criterion: [] for criterion in criteria}
        
        for sample in tqdm(samples, desc="Evaluating with LLM Judge"):
            question = sample['question']
            answer = sample['answer']
            ground_truth = sample['ground_truth']
            context = sample.get('context', None) if include_context else None
            
            # Evaluate this sample
            evaluation = self.evaluate_single(
                question=question,
                answer=answer,
                ground_truth=ground_truth,
                context=context,
                criteria=criteria
            )
            
            # Store detailed results
            result = {
                'question': question,
                'answer': answer,
                'ground_truth': ground_truth,
                'evaluation': evaluation
            }
            detailed_results.append(result)
            
            # Aggregate scores
            for criterion, result in evaluation.items():
                aggregate_scores[criterion].append(result['score'])
        
        # Calculate averages
        avg_scores = {
            criterion: sum(scores) / len(scores) if scores else 0
            for criterion, scores in aggregate_scores.items()
        }
        
        # Print summary
        print(f"\n{'='*60}")
        print("LLM JUDGE RESULTS")
        print(f"{'='*60}")
        for criterion, score in avg_scores.items():
            print(f"{criterion.capitalize():<20}: {score:.2f} / 5.0")
        print()
        
        return {
            'detailed_results': detailed_results,
            'aggregate_scores': avg_scores,
            'criteria': criteria,
            'num_samples': len(samples)
        }


if __name__ == "__main__":
    print("LLM-as-Judge Evaluation Module")
    print("\nSupported criteria:")
    print("  - faithfulness: Answer is faithful to context")
    print("  - relevance: Answer is relevant to question")
    print("  - completeness: Answer covers all key points")
    print("  - correctness: Answer is factually correct")
    print("  - overall: Overall quality assessment")
