"""
RAG Pipeline Module
End-to-end RAG system integrating retrieval and generation
"""

import os
import yaml
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI
from .query_rewriter import rewrite_query

try:
    from sentence_transformers import CrossEncoder
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    CrossEncoder = None


class LLMGenerator:
    """LLM-based answer generator"""
    
    def __init__(
        self,
        provider: str = "openai",
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        max_tokens: int = 512,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        Initialize LLM Generator
        
        Args:
            provider: LLM provider ('openai', 'deepseek', 'anthropic')
            model_name: Model name
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            api_key: API key (or loaded from env)
            base_url: Base URL for API (for DeepSeek or compatible APIs)
        """
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url
        
        # Load API key and base URL
        load_dotenv()
        if api_key:
            self.api_key = api_key
        else:
            if provider == "deepseek":
                self.api_key = os.getenv("DEEPSEEK_API_KEY")
            elif provider == "openai":
                self.api_key = os.getenv("OPENAI_API_KEY")
            elif provider == "anthropic":
                self.api_key = os.getenv("ANTHROPIC_API_KEY")
        
        # Set base URL for DeepSeek or custom endpoints
        if not self.base_url:
            if provider == "deepseek":
                self.base_url = "https://api.deepseek.com"
        
        # Initialize OpenAI-compatible client for OpenAI and DeepSeek
        if provider in ["openai", "deepseek"]:
            # Build client kwargs, only include non-None values
            client_kwargs = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            
            try:
                self.client = OpenAI(**client_kwargs)
            except TypeError as e:
                # Fallback for compatibility issues
                print(f"Warning: OpenAI client init error: {e}")
                print("Trying with minimal parameters...")
                self.client = OpenAI(api_key=self.api_key)
                if self.base_url:
                    self.client.base_url = self.base_url
        
        print(f"✓ Initialized {provider} generator with model {model_name}")
        if self.base_url:
            print(f"  Base URL: {self.base_url}")
    
    def generate(self, prompt: str) -> str:
        """
        Generate response using LLM
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        if self.provider in ["openai", "deepseek"]:
            return self._generate_openai_compatible(prompt)
        elif self.provider == "anthropic":
            return self._generate_anthropic(prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def _generate_openai_compatible(self, prompt: str) -> str:
        """Generate using OpenAI-compatible API (works for OpenAI and DeepSeek)"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=False
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _generate_anthropic(self, prompt: str) -> str:
        """Generate using Anthropic API"""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            message = client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"


class RAGPipeline:
    """End-to-end RAG pipeline"""
    
    def __init__(
        self,
        retriever,
        generator: LLMGenerator,
        rag_prompt_template: str,
        closed_book_prompt_template: str = None,
        top_k: int = 5,
        use_query_rewriting: bool = True,
        rewriting_method: str = "simple",
        use_reranker: bool = False,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        reranker_top_k: int = 3
    ):
        """
        Initialize RAG Pipeline
        
        Args:
            retriever: Retriever instance (Dense, BM25, or Hybrid)
            generator: LLM generator
            rag_prompt_template: Prompt template for RAG
            closed_book_prompt_template: Prompt template for closed-book
            top_k: Number of chunks to retrieve
            use_query_rewriting: Whether to use query rewriting for better retrieval
            rewriting_method: Query rewriting method ('simple', 'hyde', 'dense')
            use_reranker: Whether to use re-ranker
            reranker_model: Cross-encoder model for re-ranking
            reranker_top_k: Number of chunks to keep after re-ranking
        """
        self.retriever = retriever
        self.generator = generator
        self.rag_prompt_template = rag_prompt_template
        self.closed_book_prompt_template = closed_book_prompt_template
        self.top_k = top_k
        self.use_query_rewriting = use_query_rewriting
        self.rewriting_method = rewriting_method
        self.use_reranker = use_reranker
        self.reranker_top_k = reranker_top_k
        
        # Initialize re-ranker if enabled
        self.reranker = None
        if use_reranker:
            if not RERANKER_AVAILABLE:
                print("⚠️  Warning: sentence-transformers not installed, re-ranker disabled")
                self.use_reranker = False
            else:
                try:
                    self.reranker = CrossEncoder(reranker_model)
                    print(f"✓ Re-ranker initialized: {reranker_model}")
                    print(f"  Will re-rank top {top_k} → keep top {reranker_top_k}")
                except Exception as e:
                    print(f"⚠️  Warning: Failed to load re-ranker: {e}")
                    self.use_reranker = False
        
        print("✓ RAG Pipeline initialized")
        if use_query_rewriting:
            print(f"  Query rewriting: enabled (method={rewriting_method})")
    
    def query(
        self, 
        question: str, 
        return_context: bool = False
    ) -> Dict[str, Any]:
        """
        Query the RAG system
        
        Args:
            question: User question
            return_context: Whether to return retrieved context
            
        Returns:
            Dictionary with answer and optional context
        """
        # Retrieve relevant chunks with optional query rewriting
        if self.use_query_rewriting:
            chunks, scores = self._retrieve_with_rewriting(question)
        else:
            chunks, scores = self.retriever.retrieve(question, self.top_k)
        
        # Re-rank if enabled
        if self.use_reranker and self.reranker is not None:
            chunks, scores = self._rerank(question, chunks, scores)
        
        # Format context
        context = self._format_context(chunks)
        
        # Create prompt
        prompt = self.rag_prompt_template.format(
            context=context,
            question=question
        )
        
        # Generate answer
        answer = self.generator.generate(prompt)
        
        result = {
            "question": question,
            "answer": answer,
        }
        
        if return_context:
            result["context"] = chunks
            result["scores"] = scores
        
        return result
    
    def _retrieve_with_rewriting(self, question: str):
        """
        Retrieve using multiple rewritten queries and fuse results
        
        Args:
            question: Original question
            
        Returns:
            Tuple of (merged chunks, merged scores)
        """
        # Generate query variants using specified method
        query_variants = rewrite_query(question, method=self.rewriting_method)
        
        # Retrieve with each variant
        all_chunks = []
        all_scores = {}
        
        for variant in query_variants[:3]:  # Use top 3 variants
            chunks, scores = self.retriever.retrieve(variant, self.top_k)
            
            for chunk, score in zip(chunks, scores):
                chunk_id = chunk.get('chunk_id', id(chunk))
                
                # Keep the best score for each chunk
                if chunk_id not in all_scores or score > all_scores[chunk_id]['score']:
                    all_scores[chunk_id] = {
                        'chunk': chunk,
                        'score': score
                    }
        
        # Sort by score and return top_k
        sorted_results = sorted(
            all_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )[:self.top_k]
        
        final_chunks = [r['chunk'] for r in sorted_results]
        final_scores = [r['score'] for r in sorted_results]
        
        return final_chunks, final_scores
    
    def _rerank(self, question: str, chunks: List[Dict[str, Any]], scores: List[float]):
        """
        Re-rank retrieved chunks using cross-encoder
        
        Args:
            question: Original question
            chunks: Retrieved chunks
            scores: Initial retrieval scores
            
        Returns:
            Tuple of (re-ranked chunks, re-ranked scores)
        """
        if not chunks:
            return chunks, scores
        
        # Prepare query-document pairs
        texts = [chunk.get('text', '') for chunk in chunks]
        pairs = [[question, text] for text in texts]
        
        # Get re-ranking scores
        rerank_scores = self.reranker.predict(pairs)
        
        # Combine chunks with new scores
        ranked_results = list(zip(chunks, rerank_scores))
        ranked_results.sort(key=lambda x: x[1], reverse=True)
        
        # Keep top_k after re-ranking
        top_results = ranked_results[:self.reranker_top_k]
        
        reranked_chunks = [r[0] for r in top_results]
        reranked_scores = [float(r[1]) for r in top_results]
        
        return reranked_chunks, reranked_scores
    
    def closed_book_query(self, question: str) -> str:
        """
        Query without RAG (closed-book)
        
        Args:
            question: User question
            
        Returns:
            Generated answer
        """
        if not self.closed_book_prompt_template:
            raise ValueError("Closed-book prompt template not provided")
        
        prompt = self.closed_book_prompt_template.format(question=question)
        answer = self.generator.generate(prompt)
        
        return answer
    
    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks into context string
        
        Args:
            chunks: Retrieved chunks
            
        Returns:
            Formatted context string
        """
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get('text', '')
            context_parts.append(f"[{i}] {text}")
        
        return "\n\n".join(context_parts)
    
    @classmethod
    def from_config(cls, config_path: str, retriever):
        """
        Create RAG pipeline from config file
        
        Args:
            config_path: Path to config YAML file
            retriever: Pre-initialized retriever
            
        Returns:
            RAGPipeline instance
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Create generator
        llm_config = config['llm']
        generator = LLMGenerator(
            provider=llm_config['provider'],
            model_name=llm_config['model_name'],
            temperature=llm_config['temperature'],
            max_tokens=llm_config['max_tokens']
        )
        
        # Get prompts
        rag_prompt = config['prompts']['rag_prompt']
        closed_book_prompt = config['prompts'].get('closed_book_prompt')
        
        # Get retrieval settings
        top_k = config['retrieval']['top_k']
        
        # Query rewriting config
        qr_config = config['advanced'].get('query_rewriting', {})
        use_query_rewriting = qr_config.get('enabled', True)
        rewriting_method = qr_config.get('method', 'simple')
        
        # Re-ranker config
        reranker_config = config.get('reranker', {})
        use_reranker = reranker_config.get('enabled', False)
        reranker_model = reranker_config.get('model_name', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
        reranker_top_k = reranker_config.get('top_k', 3)
        
        return cls(
            retriever=retriever,
            generator=generator,
            rag_prompt_template=rag_prompt,
            closed_book_prompt_template=closed_book_prompt,
            top_k=top_k,
            use_query_rewriting=use_query_rewriting,
            rewriting_method=rewriting_method,
            use_reranker=use_reranker,
            reranker_model=reranker_model,
            reranker_top_k=reranker_top_k
        )


if __name__ == "__main__":
    # Example usage
    print("RAG Pipeline example")
    print("Note: Requires API keys and initialized retriever to run")
    
    # Example prompt template
    rag_prompt = """Answer the question based ONLY on the provided context.

Context:
{context}

Question: {question}

Answer:"""
    
    print(f"\nPrompt template:\n{rag_prompt}")
