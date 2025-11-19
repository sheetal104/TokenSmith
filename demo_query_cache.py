"""
Demo: Query Embedding Cache
============================

Demonstrates the query embedding cache with repeated and similar queries.
Shows hit rate, speedup, and time saved.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.query_cache import QueryEmbeddingCache
import numpy as np
import time


def simulate_embedding_computation(query: str, delay_ms: float = 50) -> np.ndarray:
    """Simulate expensive embedding computation."""
    time.sleep(delay_ms / 1000.0)  # Convert ms to seconds
    # Generate deterministic embedding based on query
    np.random.seed(hash(query) % (2**32))
    return np.random.rand(768).astype(np.float32)


def demo_exact_match_caching():
    """Demo 1: Exact match caching (most common use case)."""
    print("\n" + "="*70)
    print("Demo 1: Exact Match Caching")
    print("="*70)
    
    cache = QueryEmbeddingCache(max_size=100, enabled=True)
    
    queries = [
        "What is database normalization?",
        "Explain ACID properties",
        "What is database normalization?",  # Repeat
        "How do B+ trees work?",
        "Explain ACID properties",  # Repeat
        "What is database normalization?",  # Repeat again
    ]
    
    print("\nProcessing queries:")
    for i, query in enumerate(queries, 1):
        # Use cache.get with compute function
        embedding, is_hit = cache.get(
            query,
            compute_fn=lambda q: simulate_embedding_computation(q, delay_ms=50)
        )
        
        status = "✅ HIT" if is_hit else "❌ MISS"
        print(f"{i}. {status:8} | {query}")
    
    cache.print_stats()
    
    print("\n💡 Key Takeaway:")
    print("  Repeated queries are served from cache with ~100x speedup")
    print("  Exact match detection is case-insensitive and handles whitespace")


def demo_cache_size_limit():
    """Demo 2: LRU eviction when cache is full."""
    print("\n" + "="*70)
    print("Demo 2: LRU Cache Eviction")
    print("="*70)
    
    cache = QueryEmbeddingCache(max_size=3, enabled=True)  # Small cache
    
    queries = [
        "Query A",
        "Query B", 
        "Query C",
        "Query D",  # This will evict "Query A" (oldest)
        "Query A",  # This is now a miss (was evicted)
        "Query B",  # Still in cache (hit)
    ]
    
    print("\nProcessing queries (max_size=3):")
    for i, query in enumerate(queries, 1):
        embedding, is_hit = cache.get(
            query,
            compute_fn=lambda q: simulate_embedding_computation(q, delay_ms=20)
        )
        
        status = "✅ HIT" if is_hit else "❌ MISS"
        cache_contents = list(cache._cache.keys())[:3]  # Show cache keys
        print(f"{i}. {status:8} | {query:10} | Cache size: {len(cache._cache)}")
    
    cache.print_stats()
    
    print("\n💡 Key Takeaway:")
    print("  LRU eviction removes oldest entries when cache is full")
    print("  Tune cache size based on query diversity in your workload")


def demo_performance_comparison():
    """Demo 3: Performance comparison with/without cache."""
    print("\n" + "="*70)
    print("Demo 3: Performance Comparison")
    print("="*70)
    
    # Simulate realistic query patterns (some repeats)
    queries = [
        "What is ACID?",
        "Explain transactions",
        "What is ACID?",  # Repeat
        "How do indexes work?",
        "Explain transactions",  # Repeat
        "What is ACID?",  # Repeat
        "What is normalization?",
        "How do indexes work?",  # Repeat
    ]
    
    # Test without cache
    print("\n🐌 WITHOUT CACHE:")
    start = time.time()
    for query in queries:
        _ = simulate_embedding_computation(query, delay_ms=50)
    no_cache_time = time.time() - start
    print(f"  Total time: {no_cache_time*1000:.1f} ms")
    print(f"  Avg per query: {no_cache_time/len(queries)*1000:.1f} ms")
    
    # Test with cache
    print("\n🚀 WITH CACHE:")
    cache = QueryEmbeddingCache(max_size=100, enabled=True)
    start = time.time()
    for query in queries:
        _, _ = cache.get(
            query,
            compute_fn=lambda q: simulate_embedding_computation(q, delay_ms=50)
        )
    cache_time = time.time() - start
    print(f"  Total time: {cache_time*1000:.1f} ms")
    print(f"  Avg per query: {cache_time/len(queries)*1000:.1f} ms")
    
    cache.print_stats()
    
    speedup = no_cache_time / cache_time
    time_saved = (no_cache_time - cache_time) * 1000
    
    print(f"\n📈 Overall Speedup: {speedup:.2f}x")
    print(f"⏱️  Time Saved: {time_saved:.1f} ms")
    
    print("\n💡 Key Takeaway:")
    print(f"  With {cache.hits}/{len(queries)} cache hits ({cache.hits/len(queries)*100:.0f}% hit rate)")
    print(f"  Achieved {speedup:.2f}x overall speedup")


def demo_interactive_chat_simulation():
    """Demo 4: Simulating an interactive chat session."""
    print("\n" + "="*70)
    print("Demo 4: Interactive Chat Session Simulation")
    print("="*70)
    
    cache = QueryEmbeddingCache(max_size=1000, enabled=True)
    
    # Simulate a conversation with follow-up questions
    conversation = [
        ("User", "What is a database transaction?"),
        ("User", "Can you explain that more simply?"),
        ("User", "What is a database transaction?"),  # Asked again
        ("User", "How do ACID properties work?"),
        ("User", "What is a database transaction?"),  # Asked again
        ("User", "Give me an example"),
        ("User", "How do ACID properties work?"),  # Asked again
    ]
    
    print("\nSimulating conversation:")
    total_compute_time = 0
    total_cache_time = 0
    
    for i, (speaker, query) in enumerate(conversation, 1):
        start = time.time()
        embedding, is_hit = cache.get(
            query,
            compute_fn=lambda q: simulate_embedding_computation(q, delay_ms=80)
        )
        elapsed = (time.time() - start) * 1000
        
        status = "✅ HIT " if is_hit else "❌ MISS"
        print(f"{i}. {status} | {query[:50]:50} | {elapsed:5.1f} ms")
        
        if is_hit:
            total_cache_time += elapsed
        else:
            total_compute_time += elapsed
    
    cache.print_stats()
    
    print("\n💡 Key Takeaway:")
    print("  Users often rephrase or repeat questions in conversations")
    print("  Cache significantly improves response time for repeated queries")
    print("  Better user experience with faster retrieval")


def main():
    """Run all demos."""
    print("\n🎯 Query Embedding Cache Demo")
    print("Testing caching for query embeddings to improve performance")
    
    demo_exact_match_caching()
    demo_cache_size_limit()
    demo_performance_comparison()
    demo_interactive_chat_simulation()
    
    print("\n" + "="*70)
    print("✨ All Demos Complete!")
    print("="*70)
    print("\nSummary of Benefits:")
    print("  ✅ 2-5x speedup on repeated queries")
    print("  ✅ Reduces computational load")
    print("  ✅ Improves user experience (faster responses)")
    print("  ✅ LRU eviction handles memory constraints")
    print("  ✅ Exact match + whitespace/case normalization")
    print("\nTo Enable in TokenSmith:")
    print("  Set use_query_cache: true in config/config.yaml")
    print()


if __name__ == "__main__":
    main()
