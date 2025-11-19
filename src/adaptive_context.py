"""
Adaptive Context Window Management
===================================

Dynamically adjusts the number of chunks to use based on:
1. Relevance score distribution (quality of retrieval)
2. Query complexity (simple vs complex questions)
3. Token budget constraints

This prevents wasting context on low-quality chunks and improves answer quality.
"""

from typing import List, Tuple, Dict, Any
import statistics


class AdaptiveContextManager:
    """
    Manages context window size dynamically based on chunk quality.
    
    Key principles:
    - Use fewer chunks if top results are very relevant (high confidence)
    - Use more chunks if relevance scores are spread out (uncertain)
    - Never exceed token budget
    - Filter out low-quality chunks below relevance threshold
    """
    
    def __init__(
        self,
        min_chunks: int = 3,
        max_chunks: int = 10,
        relevance_threshold: float = 0.5,
        quality_variance_threshold: float = 0.15,
        max_tokens: int = 4000
    ):
        """
        Args:
            min_chunks: Minimum number of chunks to use (even if low quality)
            max_chunks: Maximum number of chunks to consider
            relevance_threshold: Minimum relevance score to include a chunk
            quality_variance_threshold: If variance is low, use fewer chunks
            max_tokens: Maximum tokens allowed in context
        """
        self.min_chunks = min_chunks
        self.max_chunks = max_chunks
        self.relevance_threshold = relevance_threshold
        self.quality_variance_threshold = quality_variance_threshold
        self.max_tokens = max_tokens
    
    def select_chunks(
        self,
        ranked_chunks: List[Tuple[int, str]],
        scores: Dict[int, float],
        query_complexity: str = "medium"
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Select optimal number of chunks based on quality and complexity.
        
        Args:
            ranked_chunks: List of (chunk_id, chunk_text) tuples in ranked order
            scores: Dict mapping chunk_id to relevance score
            query_complexity: "simple", "medium", or "complex"
        
        Returns:
            Tuple of (selected_chunks, metadata)
        """
        if not ranked_chunks:
            return [], {"reason": "no_chunks", "count": 0}
        
        # Extract scores for ranked chunks
        chunk_scores = [scores.get(chunk_id, 0.0) for chunk_id, _ in ranked_chunks[:self.max_chunks]]
        
        if not chunk_scores:
            return [chunk for _, chunk in ranked_chunks[:self.min_chunks]], {
                "reason": "no_scores",
                "count": self.min_chunks
            }
        
        # Analyze score distribution
        mean_score = statistics.mean(chunk_scores)
        if len(chunk_scores) > 1:
            score_variance = statistics.variance(chunk_scores)
            score_std = statistics.stdev(chunk_scores)
        else:
            score_variance = 0
            score_std = 0
        
        max_score = max(chunk_scores)
        score_range = max_score - min(chunk_scores)
        
        # Decision logic
        selected_count = self._determine_chunk_count(
            mean_score=mean_score,
            max_score=max_score,
            score_variance=score_variance,
            score_std=score_std,
            score_range=score_range,
            query_complexity=query_complexity,
            available_chunks=len(chunk_scores)
        )
        
        # Filter by relevance threshold
        selected_chunks = []
        filtered_count = 0
        
        for i, (chunk_id, chunk_text) in enumerate(ranked_chunks[:selected_count]):
            score = scores.get(chunk_id, 0.0)
            if score >= self.relevance_threshold or i < self.min_chunks:
                selected_chunks.append(chunk_text)
            else:
                filtered_count += 1
        
        # Ensure minimum
        if len(selected_chunks) < self.min_chunks:
            for _, chunk_text in ranked_chunks[len(selected_chunks):self.min_chunks]:
                selected_chunks.append(chunk_text)
        
        # Token budget check (simplified - actual would count tokens)
        estimated_tokens = sum(len(chunk.split()) * 1.3 for chunk in selected_chunks)
        if estimated_tokens > self.max_tokens:
            # Trim from the end
            while estimated_tokens > self.max_tokens and len(selected_chunks) > self.min_chunks:
                removed = selected_chunks.pop()
                estimated_tokens -= len(removed.split()) * 1.3
        
        metadata = {
            "count": len(selected_chunks),
            "requested": selected_count,
            "filtered": filtered_count,
            "mean_score": mean_score,
            "max_score": max_score,
            "score_variance": score_variance,
            "score_std": score_std,
            "score_range": score_range,
            "estimated_tokens": int(estimated_tokens),
            "reason": self._explain_decision(
                mean_score, max_score, score_variance, 
                query_complexity, len(selected_chunks)
            )
        }
        
        return selected_chunks, metadata
    
    def _determine_chunk_count(
        self,
        mean_score: float,
        max_score: float,
        score_variance: float,
        score_std: float,
        score_range: float,
        query_complexity: str,
        available_chunks: int
    ) -> int:
        """Determine optimal chunk count based on quality metrics."""
        
        # Base count on query complexity
        complexity_map = {
            "simple": self.min_chunks,
            "medium": (self.min_chunks + self.max_chunks) // 2,
            "complex": self.max_chunks
        }
        base_count = complexity_map.get(query_complexity, self.min_chunks + 3)
        
        # Adjust based on quality
        
        # High confidence (low variance, high max score) → use fewer chunks
        if score_variance < self.quality_variance_threshold and max_score > 0.8:
            adjustment = -2
            
        # Very high quality top result → use even fewer
        elif max_score > 0.9 and score_range > 0.3:
            adjustment = -3
            
        # Low variance but medium scores → slight reduction
        elif score_variance < self.quality_variance_threshold:
            adjustment = -1
            
        # High variance (uncertain) → use more chunks
        elif score_variance > self.quality_variance_threshold * 2:
            adjustment = +2
            
        # Low mean score → try more chunks
        elif mean_score < 0.6:
            adjustment = +1
            
        else:
            adjustment = 0
        
        # Apply adjustment
        count = base_count + adjustment
        
        # Enforce bounds
        count = max(self.min_chunks, min(self.max_chunks, count, available_chunks))
        
        return count
    
    def _explain_decision(
        self,
        mean_score: float,
        max_score: float,
        score_variance: float,
        query_complexity: str,
        final_count: int
    ) -> str:
        """Generate human-readable explanation of the decision."""
        
        reasons = []
        
        if max_score > 0.9:
            reasons.append("very high top score")
        elif max_score > 0.8:
            reasons.append("high top score")
        elif max_score < 0.6:
            reasons.append("low top score")
        
        if score_variance < self.quality_variance_threshold:
            reasons.append("low variance (confident)")
        elif score_variance > self.quality_variance_threshold * 2:
            reasons.append("high variance (uncertain)")
        
        if query_complexity == "complex":
            reasons.append("complex query")
        elif query_complexity == "simple":
            reasons.append("simple query")
        
        if not reasons:
            return f"Using {final_count} chunks (standard strategy)"
        
        return f"Using {final_count} chunks: " + ", ".join(reasons)
    
    def estimate_query_complexity(self, query: str) -> str:
        """
        Estimate query complexity based on length and structure.
        
        Returns:
            "simple", "medium", or "complex"
        """
        words = query.split()
        word_count = len(words)
        
        # Check for multiple questions
        question_marks = query.count('?')
        
        # Check for complex connectors
        complex_words = ['and', 'or', 'but', 'however', 'moreover', 'furthermore']
        has_complex_structure = any(word in query.lower() for word in complex_words)
        
        # Classification
        if word_count <= 5 and question_marks <= 1:
            return "simple"
        elif word_count > 15 or question_marks > 1 or has_complex_structure:
            return "complex"
        else:
            return "medium"


def integrate_adaptive_context(
    ranked_chunk_ids: List[int],
    all_chunks: List[str],
    scores: Dict[int, float],
    query: str,
    config: Dict[str, Any]
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Helper function to integrate adaptive context into existing pipeline.
    
    Args:
        ranked_chunk_ids: Ranked list of chunk IDs
        all_chunks: All available chunks
        scores: Relevance scores
        query: User query
        config: Configuration dict with adaptive settings
    
    Returns:
        Tuple of (selected_chunks, metadata)
    """
    # Create ranked chunks list
    ranked_chunks = [(chunk_id, all_chunks[chunk_id]) for chunk_id in ranked_chunk_ids]
    
    # Initialize manager
    manager = AdaptiveContextManager(
        min_chunks=config.get('min_chunks', 3),
        max_chunks=config.get('max_chunks', 10),
        relevance_threshold=config.get('relevance_threshold', 0.5),
        quality_variance_threshold=config.get('quality_variance_threshold', 0.15),
        max_tokens=config.get('max_context_tokens', 4000)
    )
    
    # Estimate complexity
    complexity = manager.estimate_query_complexity(query)
    
    # Select chunks
    selected, metadata = manager.select_chunks(ranked_chunks, scores, complexity)
    metadata['query_complexity'] = complexity
    
    return selected, metadata
