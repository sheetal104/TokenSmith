#!/usr/bin/env python3
"""
Quick Demo of Query Type Classification
=========================================

This script demonstrates how the query planner works without needing
the full TokenSmith setup. Perfect for quick testing and demonstration.

Usage:
    python demo_query_classification.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.planning.heuristics import HeuristicQueryPlanner
from src.config import QueryPlanConfig


def demo_classification():
    """Demonstrate query classification on example queries."""
    
    print("="*70)
    print("TokenSmith Query Type Classification Demo")
    print("="*70)
    print()
    
    # Load config
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        print("❌ Config file not found. Please run from TokenSmith root directory.")
        return
    
    cfg = QueryPlanConfig.from_yaml(config_path)
    
    # Initialize logger (required for planner)
    from src.instrumentation.logging import init_logger
    init_logger(cfg)
    
    planner = HeuristicQueryPlanner(cfg)
    
    # Test queries covering different types
    test_queries = [
        # Definition queries
        "What is ACID in databases?",
        "Define normalization",
        "What does MVCC mean?",
        
        # Explanatory queries
        "Why do we need transactions?",
        "Explain how B+ trees work",
        "What causes deadlocks?",
        
        # Procedural queries  
        "How to implement a B+ tree?",
        "Steps to normalize a database",
        "How do you create an index?",
        
        # Comparative queries
        "Difference between SQL and NoSQL",
        "Compare 2PL and MVCC",
        "Advantages of B+ trees vs hash indexes",
        
        # Other/mixed
        "Tell me about database transactions",
    ]
    
    print("Classifying queries and showing weight adjustments...")
    print()
    
    for query in test_queries:
        # Classify the query
        query_type, confidence = planner.classify(query)
        
        # Get the planned configuration
        planned_cfg = planner.plan(query)
        classification = planned_cfg.query_classification
        
        # Print results
        print(f"📝 Query: \"{query}\"")
        print(f"   Type: {query_type.upper()} (confidence: {confidence:.2f})")
        print(f"   Weights:")
        print(f"     • Original:  FAISS={classification['original_weights']['faiss']:.1f}, "
              f"BM25={classification['original_weights']['bm25']:.1f}")
        print(f"     • Adjusted:  FAISS={classification['adjusted_weights']['faiss']:.1f}, "
              f"BM25={classification['adjusted_weights']['bm25']:.1f}")
        
        if classification['adjusted_pool_size'] != classification['original_pool_size']:
            print(f"   Pool Size: {classification['original_pool_size']} → "
                  f"{classification['adjusted_pool_size']}")
        
        # Explain the reasoning
        if query_type == "definition":
            print(f"   💡 Favoring BM25 for exact keyword matching")
        elif query_type == "explanatory":
            print(f"   💡 Favoring FAISS for semantic understanding")
        elif query_type == "procedural":
            print(f"   💡 Using larger pool + balanced weights for multi-chunk answers")
        elif query_type == "comparative":
            print(f"   💡 Using equal weights + larger pool for multiple concepts")
        else:
            print(f"   💡 Using default balanced strategy")
        
        print()
    
    # Show statistics
    print("="*70)
    print("Classification Statistics")
    print("="*70)
    stats = planner.get_statistics()
    total = sum(stats.values())
    
    for qtype, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            percentage = (count / total) * 100
            print(f"  {qtype:15} : {count:2} queries ({percentage:.1f}%)")
    print()


def demo_interactive():
    """Interactive demo where you can type queries."""
    
    print("="*70)
    print("Interactive Query Classification Demo")
    print("="*70)
    print()
    print("Type queries to see how they're classified.")
    print("Type 'exit' or 'quit' to end.")
    print()
    
    config_path = Path("config/config.yaml")
    cfg = QueryPlanConfig.from_yaml(config_path)
    
    # Initialize logger (required for planner)
    from src.instrumentation.logging import init_logger
    init_logger(cfg)
    
    planner = HeuristicQueryPlanner(cfg)
    
    while True:
        try:
            query = input("Query > ").strip()
            
            if not query:
                continue
                
            if query.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            
            # Classify and plan
            query_type, confidence = planner.classify(query)
            planned_cfg = planner.plan(query)
            classification = planned_cfg.query_classification
            
            # Display results
            print(f"\n  🔍 Type: {query_type.upper()}")
            print(f"  📊 Confidence: {confidence:.2f}")
            print(f"  ⚙️  Weights: FAISS={classification['adjusted_weights']['faiss']:.1f}, "
                  f"BM25={classification['adjusted_weights']['bm25']:.1f}")
            
            if query_type == "definition":
                print(f"  💡 Strategy: Prioritize exact keyword matches")
            elif query_type == "explanatory":
                print(f"  💡 Strategy: Prioritize semantic similarity")
            elif query_type == "procedural":
                print(f"  💡 Strategy: Larger pool for multi-step answers")
            elif query_type == "comparative":
                print(f"  💡 Strategy: Balanced approach for multiple concepts")
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Demo query classification")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Run in interactive mode")
    
    args = parser.parse_args()
    
    if args.interactive:
        demo_interactive()
    else:
        demo_classification()
