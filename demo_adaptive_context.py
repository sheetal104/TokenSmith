"""
Demo: Adaptive Context Window Management
=========================================

Shows how the system dynamically adjusts the number of chunks based on:
- Query complexity
- Relevance score distribution
- Quality metrics
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.adaptive_context import AdaptiveContextManager
from typing import List, Dict, Tuple
import random


def generate_mock_chunks(count: int = 20) -> List[str]:
    """Generate mock document chunks."""
    topics = [
        "Machine learning algorithms use training data to improve performance",
        "Neural networks consist of interconnected layers of nodes",
        "Gradient descent optimizes model parameters iteratively",
        "Overfitting occurs when models memorize training data",
        "Regularization techniques prevent overfitting in machine learning",
        "Transfer learning reuses pre-trained models for new tasks",
        "Ensemble methods combine multiple models for better predictions",
        "Feature engineering transforms raw data into useful inputs",
        "Cross-validation splits data to evaluate model performance",
        "Hyperparameter tuning optimizes model configuration",
    ]
    
    chunks = []
    for i in range(count):
        base = topics[i % len(topics)]
        chunks.append(f"Chunk {i+1}: {base}. This chunk contains additional context and details.")
    
    return chunks


def simulate_scores_high_confidence() -> Dict[int, float]:
    """Simulate scores where top result is very relevant, others drop off."""
    scores = {}
    scores[0] = 0.95  # Very high top score
    for i in range(1, 10):
        scores[i] = max(0.3, 0.9 - i * 0.1)  # Rapid drop-off
    return scores


def simulate_scores_uncertain() -> Dict[int, float]:
    """Simulate scores where many chunks have similar relevance."""
    scores = {}
    for i in range(10):
        scores[i] = 0.6 + random.uniform(-0.15, 0.15)  # High variance
    return scores


def simulate_scores_low_quality() -> Dict[int, float]:
    """Simulate scores where nothing is very relevant."""
    scores = {}
    for i in range(10):
        scores[i] = random.uniform(0.3, 0.5)  # All mediocre
    return scores


def demo_scenario(
    name: str,
    query: str,
    chunks: List[str],
    scores: Dict[int, float],
    manager: AdaptiveContextManager
):
    """Run one scenario and show results."""
    print(f"\n{'='*70}")
    print(f"Scenario: {name}")
    print(f"{'='*70}")
    print(f"Query: \"{query}\"")
    print(f"Query complexity: {manager.estimate_query_complexity(query)}")
    print()
    
    # Prepare ranked chunks
    ranked_chunk_ids = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
    ranked_chunks = [(i, chunks[i]) for i in ranked_chunk_ids]
    
    # Select chunks
    complexity = manager.estimate_query_complexity(query)
    selected, metadata = manager.select_chunks(ranked_chunks, scores, complexity)
    
    print(f"📊 Score Distribution:")
    print(f"   Mean: {metadata['mean_score']:.3f}")
    print(f"   Max:  {metadata['max_score']:.3f}")
    print(f"   Std:  {metadata['score_std']:.3f}")
    print(f"   Variance: {metadata['score_variance']:.3f}")
    print()
    
    print(f"✅ Decision: {metadata['reason']}")
    print(f"   Chunks selected: {metadata['count']}")
    print(f"   Chunks requested: {metadata['requested']}")
    print(f"   Chunks filtered: {metadata['filtered']}")
    print(f"   Estimated tokens: {metadata['estimated_tokens']}")
    print()
    
    print(f"📄 Selected Chunks (top 3):")
    for i, chunk in enumerate(selected[:3], 1):
        preview = chunk[:80] + "..." if len(chunk) > 80 else chunk
        print(f"   {i}. {preview}")


def main():
    """Run comprehensive demo."""
    print("\n🎯 Adaptive Context Window Management Demo")
    print("=" * 70)
    
    # Generate mock data
    chunks = generate_mock_chunks(20)
    
    # Initialize manager with default settings
    manager = AdaptiveContextManager(
        min_chunks=3,
        max_chunks=10,
        relevance_threshold=0.5,
        quality_variance_threshold=0.15,
        max_tokens=4000
    )
    
    print(f"\nManager Configuration:")
    print(f"  Min chunks: {manager.min_chunks}")
    print(f"  Max chunks: {manager.max_chunks}")
    print(f"  Relevance threshold: {manager.relevance_threshold}")
    print(f"  Quality variance threshold: {manager.quality_variance_threshold}")
    
    # Scenario 1: High confidence (clear winner)
    demo_scenario(
        "High Confidence - Clear Best Result",
        "What is gradient descent?",
        chunks,
        simulate_scores_high_confidence(),
        manager
    )
    
    # Scenario 2: Uncertain (many similar results)
    demo_scenario(
        "Uncertain - Many Similar Results",
        "Compare machine learning algorithms and explain differences",
        chunks,
        simulate_scores_uncertain(),
        manager
    )
    
    # Scenario 3: Low quality (nothing great)
    demo_scenario(
        "Low Quality - No Great Matches",
        "What is quantum computing?",
        chunks,
        simulate_scores_low_quality(),
        manager
    )
    
    # Scenario 4: Simple query, high confidence
    demo_scenario(
        "Simple Query + High Confidence",
        "Define overfitting",
        chunks,
        simulate_scores_high_confidence(),
        manager
    )
    
    # Scenario 5: Complex query, uncertain
    demo_scenario(
        "Complex Query + Uncertainty",
        "How do ensemble methods, regularization, and cross-validation work together?",
        chunks,
        simulate_scores_uncertain(),
        manager
    )
    
    print("\n" + "="*70)
    print("✨ Demo Complete!")
    print("="*70)
    print("\nKey Takeaways:")
    print("  • High confidence queries use fewer chunks (avoid noise)")
    print("  • Complex queries use more chunks (gather comprehensive info)")
    print("  • Low quality results try more chunks (search harder)")
    print("  • Score variance indicates uncertainty level")
    print("  • Token budget prevents context overflow")
    print()


if __name__ == "__main__":
    main()
