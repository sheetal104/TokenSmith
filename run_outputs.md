===========================================
TokenSmith Q/A Benchmark
===========================================
Questions: 12 from data/starter_questions.y
Baseline config: config/config.yaml
Improved config: config/config_all_improvem
Output directory: results

===========================================
Running BASELINE mode...
===========================================
Loading model with n_ctx=8192, n_threads=No
llama_context: n_ctx_per_seq (8192) < n_ctx
init: embeddings required but some input to
Model loaded successfully. Embedding dimens
✓ Loaded 1752 chunks from 1752 sources
  [1/12] What is database normalization?...
Encoding 1 texts with batch_size=32
init: embeddings required but some input to
      ⏱️  26.01s
  [2/12] Define ACID properties...
Encoding 1 texts with batch_size=32
init: embeddings required but some input to
      ⏱️  28.00s
  [3/12] What does MVCC mean?...
Encoding 1 texts with batch_size=32
init: embeddings required but some input to
      ⏱️  25.23s
  [4/12] Why are indexes important?...
Encoding 1 texts with batch_size=32
init: embeddings required but some input to
      ⏱️  31.41s
  [5/12] Explain how B+ trees work...
Encoding 1 texts with batch_size=32
init: embeddings required but some input to
      ⏱️  40.91s
  [6/12] How does query optimization improv
Encoding 1 texts with batch_size=32
init: embeddings required but some input to
      ⏱️  35.75s
  [7/12] How to design a relational schema?
Encoding 1 texts with batch_size=32
init: embeddings required but some input to
      ⏱️  33.41s
  [8/12] Steps to implement a transaction..
Encoding 1 texts with batch_size=32
init: embeddings required but some input to
      ⏱️  46.93s
  [9/12] Compare OLTP and OLAP databases...
Encoding 1 texts with batch_size=32
init: embeddings required but some input to
      ⏱️  56.45s
  [10/12] What's the difference between MyS
Encoding 1 texts with batch_size=32
init: embeddings required but some input to
      ⏱️  31.68s
  [11/12] What is database normalization?..
Encoding 1 texts with batch_size=32
init: embeddings required but some input to
      ⏱️  31.42s
  [12/12] Explain how B+ trees work...
Encoding 1 texts with batch_size=32
init: embeddings required but some input to
      ⏱️  42.04s

===========================================
Running IMPROVED mode...
===========================================
✓ Query planner enabled
Loading model with n_ctx=8192, n_threads=No
llama_context: n_ctx_per_seq (8192) < n_ctx
✅ Query cache enabled (size: 1000)
init: embeddings required but some input to
Model loaded successfully. Embedding dimens
✓ Loaded 1752 chunks from 1752 sources
  [1/12] What is database normalization?...
init: embeddings required but some input to
      ⏱️  35.81s
  [2/12] Define ACID properties...
init: embeddings required but some input to
      ⏱️  38.14s
  [3/12] What does MVCC mean?...
init: embeddings required but some input to
      ⏱️  32.95s
  [4/12] Why are indexes important?...
init: embeddings required but some input to
      ⏱️  52.44s
  [5/12] Explain how B+ trees work...
init: embeddings required but some input to
      ⏱️  41.55s
  [6/12] How does query optimization improv
init: embeddings required but some input to
      ⏱️  46.22s
  [7/12] How to design a relational schema?
init: embeddings required but some input to
      ⏱️  39.68s
  [8/12] Steps to implement a transaction..
init: embeddings required but some input to
      ⏱️  51.48s
  [9/12] Compare OLTP and OLAP databases...
init: embeddings required but some input to
      ⏱️  45.74s
  [10/12] What's the difference between MyS
init: embeddings required but some input to
      ⏱️  50.12s
  [11/12] What is database normalization?..
      ⏱️  23.04s
  [12/12] Explain how B+ trees work...
      ⏱️  41.92s

✅ Benchmark complete!              16.7% h
============================================================
📄 Saved: results/qa_baseline.json
📄 Saved: results/qa_improved.json

(tokensmith) sheetal@DESKTOP-PK1KIFD:/mnt/e/Projects/TokenSmith$ python3 compare_baseline_vs_planner.py
======================================================================
A/B Comparison: Baseline vs Query Planner
======================================================================

📊 Baseline Configuration (No Planner):
   Fixed weights: FAISS=0.6, BM25=0.4

🧠 Query Planner Configuration:
   Adaptive weights based on query type

======================================================================
Per-Query Comparison
======================================================================

📈 What is ACID?
   Type: definition (expected: definition)
   Baseline: 0.817 (FAISS=0.6, BM25=0.4)
   Planner:  0.892 (FAISS=0.3, BM25=0.7)
   Improvement: +9.2%

📈 Define normalization
   Type: definition (expected: definition)
   Baseline: 0.807 (FAISS=0.6, BM25=0.4)
   Planner:  0.882 (FAISS=0.3, BM25=0.7)
   Improvement: +9.3%

➖ What does MVCC mean?
   Type: other (expected: definition)
   Baseline: 0.831 (FAISS=0.6, BM25=0.4)
   Planner:  0.831 (FAISS=0.6, BM25=0.4)
   Improvement: +0.0%

📈 Why do we need transactions?
   Type: explanatory (expected: explanatory)
   Baseline: 0.882 (FAISS=0.6, BM25=0.4)
   Planner:  0.907 (FAISS=0.7, BM25=0.3)
   Improvement: +2.8%

📈 Explain how B+ trees work
   Type: explanatory (expected: explanatory)
   Baseline: 0.881 (FAISS=0.6, BM25=0.4)
   Planner:  0.906 (FAISS=0.7, BM25=0.3)
   Improvement: +2.8%

📈 What causes deadlocks?
   Type: explanatory (expected: explanatory)
   Baseline: 0.872 (FAISS=0.6, BM25=0.4)
   Planner:  0.897 (FAISS=0.7, BM25=0.3)
   Improvement: +2.9%

➖ How to implement a B+ tree?
   Type: procedural (expected: procedural)
   Baseline: 0.916 (FAISS=0.6, BM25=0.4)
   Planner:  0.916 (FAISS=0.6, BM25=0.4)
   Improvement: +0.0%

➖ Steps to normalize a database
   Type: procedural (expected: procedural)
   Baseline: 0.917 (FAISS=0.6, BM25=0.4)
   Planner:  0.917 (FAISS=0.6, BM25=0.4)
   Improvement: +0.0%

📈 Difference between SQL and NoSQL
   Type: comparative (expected: comparative)
   Baseline: 0.895 (FAISS=0.6, BM25=0.4)
   Planner:  0.920 (FAISS=0.5, BM25=0.5)
   Improvement: +2.8%

======================================================================
OVERALL RESULTS
======================================================================

📊 Average Scores:
   Baseline (fixed weights):  0.869
   Planner (adaptive):        0.896
   Overall Improvement:       +3.2%

📈 Improvement by Query Type:
   comparative     : +2.8%
   definition      : +6.2%
   explanatory     : +2.8%
   procedural      : +0.0%

📊 Win/Loss Breakdown:
   Better with planner:  6/9 (66.7%)
   Worse with planner:   0/9 (0.0%)
   No change:            3/9

✅ Comparison report saved to: tests/validation_results/comparison_report_20251118_200749.html

✅ Query planner shows positive improvement
(tokensmith) sheetal@DESKTOP-PK1KIFD:/mnt/e/Projects/TokenSmith$ python3 demo_query_classification.py
======================================================================
TokenSmith Query Type Classification Demo
======================================================================

Classifying queries and showing weight adjustments...

📝 Query: "What is ACID in databases?"
   Type: DEFINITION (confidence: 0.20)
   Weights:
     • Original:  FAISS=1.0, BM25=0.0
     • Adjusted:  FAISS=0.3, BM25=0.7
   💡 Favoring BM25 for exact keyword matching

📝 Query: "Define normalization"
   Type: DEFINITION (confidence: 0.50)
   Weights:
     • Original:  FAISS=1.0, BM25=0.0
     • Adjusted:  FAISS=0.3, BM25=0.7
   💡 Favoring BM25 for exact keyword matching

📝 Query: "What does MVCC mean?"
   Type: OTHER (confidence: 0.00)
   Weights:
     • Original:  FAISS=1.0, BM25=0.0
     • Adjusted:  FAISS=0.6, BM25=0.4
   💡 Using default balanced strategy

📝 Query: "Why do we need transactions?"
   Type: EXPLANATORY (confidence: 0.20)
   Weights:
     • Original:  FAISS=1.0, BM25=0.0
     • Adjusted:  FAISS=0.7, BM25=0.3
   💡 Favoring FAISS for semantic understanding

📝 Query: "Explain how B+ trees work"
   Type: EXPLANATORY (confidence: 0.20)
   Weights:
     • Original:  FAISS=1.0, BM25=0.0
     • Adjusted:  FAISS=0.7, BM25=0.3
   💡 Favoring FAISS for semantic understanding

📝 Query: "What causes deadlocks?"
   Type: EXPLANATORY (confidence: 0.33)
   Weights:
     • Original:  FAISS=1.0, BM25=0.0
     • Adjusted:  FAISS=0.7, BM25=0.3
   💡 Favoring FAISS for semantic understanding

📝 Query: "How to implement a B+ tree?"
   Type: PROCEDURAL (confidence: 0.33)
   Weights:
     • Original:  FAISS=1.0, BM25=0.0
     • Adjusted:  FAISS=0.6, BM25=0.4
   💡 Using larger pool + balanced weights for multi-chunk answers

📝 Query: "Steps to normalize a database"
   Type: PROCEDURAL (confidence: 0.20)
   Weights:
     • Original:  FAISS=1.0, BM25=0.0
     • Adjusted:  FAISS=0.6, BM25=0.4
   💡 Using larger pool + balanced weights for multi-chunk answers

📝 Query: "How do you create an index?"
   Type: OTHER (confidence: 0.00)
   Weights:
     • Original:  FAISS=1.0, BM25=0.0
     • Adjusted:  FAISS=0.6, BM25=0.4
   💡 Using default balanced strategy

📝 Query: "Difference between SQL and NoSQL"
   Type: COMPARATIVE (confidence: 0.20)
   Weights:
     • Original:  FAISS=1.0, BM25=0.0
     • Adjusted:  FAISS=0.5, BM25=0.5
   💡 Using equal weights + larger pool for multiple concepts

📝 Query: "Compare 2PL and MVCC"
   Type: COMPARATIVE (confidence: 0.25)
   Weights:
     • Original:  FAISS=1.0, BM25=0.0
     • Adjusted:  FAISS=0.5, BM25=0.5
   💡 Using equal weights + larger pool for multiple concepts

📝 Query: "Advantages of B+ trees vs hash indexes"
   Type: COMPARATIVE (confidence: 0.29)
   Weights:
     • Original:  FAISS=1.0, BM25=0.0
     • Adjusted:  FAISS=0.5, BM25=0.5
   💡 Using equal weights + larger pool for multiple concepts

📝 Query: "Tell me about database transactions"
   Type: OTHER (confidence: 0.00)
   Weights:
     • Original:  FAISS=1.0, BM25=0.0
     • Adjusted:  FAISS=0.6, BM25=0.4
   💡 Using default balanced strategy

======================================================================
Classification Statistics
======================================================================
  explanatory     :  6 queries (30.0%)
  comparative     :  6 queries (30.0%)
  definition      :  4 queries (20.0%)
  procedural      :  4 queries (20.0%)

(tokensmith) sheetal@DESKTOP-PK1KIFD:/mnt/e/Projects/TokenSmith$ python3 demo_adaptive_context.py

🎯 Adaptive Context Window Management Demo
======================================================================

Manager Configuration:
  Min chunks: 3
  Max chunks: 10
  Relevance threshold: 0.5
  Quality variance threshold: 0.15

======================================================================
Scenario: High Confidence - Clear Best Result
======================================================================
Query: "What is gradient descent?"
Query complexity: simple

📊 Score Distribution:
   Mean: 0.515
   Max:  0.950
   Std:  0.238
   Variance: 0.057

✅ Decision: Using 3 chunks: very high top score, low variance (confident), simple query
   Chunks selected: 3
   Chunks requested: 3
   Chunks filtered: 0
   Estimated tokens: 65

📄 Selected Chunks (top 3):
   1. Chunk 1: Machine learning algorithms use training data to improve performance. T...
   2. Chunk 2: Neural networks consist of interconnected layers of nodes. This chunk c...
   3. Chunk 3: Gradient descent optimizes model parameters iteratively. This chunk con...

======================================================================
Scenario: Uncertain - Many Similar Results
======================================================================
Query: "Compare machine learning algorithms and explain differences"
Query complexity: complex

📊 Score Distribution:
   Mean: 0.562
   Max:  0.694
   Std:  0.077
   Variance: 0.006

✅ Decision: Using 8 chunks: low variance (confident), complex query
   Chunks selected: 8
   Chunks requested: 9
   Chunks filtered: 1
   Estimated tokens: 169

📄 Selected Chunks (top 3):
   1. Chunk 1: Machine learning algorithms use training data to improve performance. T...
   2. Chunk 7: Ensemble methods combine multiple models for better predictions. This c...
   3. Chunk 3: Gradient descent optimizes model parameters iteratively. This chunk con...

======================================================================
Scenario: Low Quality - No Great Matches
======================================================================
Query: "What is quantum computing?"
Query complexity: simple

📊 Score Distribution:
   Mean: 0.396
   Max:  0.497
   Std:  0.056
   Variance: 0.003

✅ Decision: Using 3 chunks: low top score, low variance (confident), simple query
   Chunks selected: 3
   Chunks requested: 3
   Chunks filtered: 0
   Estimated tokens: 65

📄 Selected Chunks (top 3):
   1. Chunk 5: Regularization techniques prevent overfitting in machine learning. This...
   2. Chunk 8: Feature engineering transforms raw data into useful inputs. This chunk ...
   3. Chunk 7: Ensemble methods combine multiple models for better predictions. This c...

======================================================================
Scenario: Simple Query + High Confidence
======================================================================
Query: "Define overfitting"
Query complexity: simple

📊 Score Distribution:
   Mean: 0.515
   Max:  0.950
   Std:  0.238
   Variance: 0.057

✅ Decision: Using 3 chunks: very high top score, low variance (confident), simple query
   Chunks selected: 3
   Chunks requested: 3
   Chunks filtered: 0
   Estimated tokens: 65

📄 Selected Chunks (top 3):
   1. Chunk 1: Machine learning algorithms use training data to improve performance. T...
   2. Chunk 2: Neural networks consist of interconnected layers of nodes. This chunk c...
   3. Chunk 3: Gradient descent optimizes model parameters iteratively. This chunk con...

======================================================================
Scenario: Complex Query + Uncertainty
======================================================================
Query: "How do ensemble methods, regularization, and cross-validation work together?"
Query complexity: complex

📊 Score Distribution:
   Mean: 0.572
   Max:  0.723
   Std:  0.088
   Variance: 0.008

✅ Decision: Using 7 chunks: low variance (confident), complex query
   Chunks selected: 7
   Chunks requested: 9
   Chunks filtered: 2
   Estimated tokens: 145

📄 Selected Chunks (top 3):
   1. Chunk 9: Cross-validation splits data to evaluate model performance. This chunk ...
   2. Chunk 5: Regularization techniques prevent overfitting in machine learning. This...
   3. Chunk 4: Overfitting occurs when models memorize training data. This chunk conta...

======================================================================
✨ Demo Complete!
======================================================================

Key Takeaways:
  • High confidence queries use fewer chunks (avoid noise)
  • Complex queries use more chunks (gather comprehensive info)
  • Low quality results try more chunks (search harder)
  • Score variance indicates uncertainty level
  • Token budget prevents context overflow

(tokensmith) sheetal@DESKTOP-PK1KIFD:/mnt/e/Projects/TokenSmith$ python3 demo_query_cache.py

🎯 Query Embedding Cache Demo
Testing caching for query embeddings to improve performance

======================================================================
Demo 1: Exact Match Caching
======================================================================

Processing queries:
1. ❌ MISS   | What is database normalization?
2. ❌ MISS   | Explain ACID properties
3. ✅ HIT    | What is database normalization?
4. ❌ MISS   | How do B+ trees work?
5. ✅ HIT    | Explain ACID properties
6. ✅ HIT    | What is database normalization?

📊 Query Embedding Cache Statistics
============================================================
Status: ✅ Enabled
Cache size: 3 / 100
Total requests: 6
  Hits: 3 (50.0%)
  Misses: 3
Performance:
  Avg compute time: 54.97 ms
  Avg cache time: 0.02 ms
  Speedup: 2360.8x
  Time saved: 164.8 ms
============================================================

💡 Key Takeaway:
  Repeated queries are served from cache with ~100x speedup
  Exact match detection is case-insensitive and handles whitespace

======================================================================
Demo 2: LRU Cache Eviction
======================================================================

Processing queries (max_size=3):
1. ❌ MISS   | Query A    | Cache size: 1
2. ❌ MISS   | Query B    | Cache size: 2
3. ❌ MISS   | Query C    | Cache size: 3
4. ❌ MISS   | Query D    | Cache size: 3
5. ❌ MISS   | Query A    | Cache size: 3
6. ❌ MISS   | Query B    | Cache size: 3

📊 Query Embedding Cache Statistics
============================================================
Status: ✅ Enabled
Cache size: 3 / 3
Total requests: 6
  Hits: 0 (0.0%)
  Misses: 6
Performance:
  Avg compute time: 20.43 ms
  Avg cache time: 0.00 ms
  Speedup: 1.0x
  Time saved: 0.0 ms
============================================================

💡 Key Takeaway:
  LRU eviction removes oldest entries when cache is full
  Tune cache size based on query diversity in your workload

======================================================================
Demo 3: Performance Comparison
======================================================================

🐌 WITHOUT CACHE:
  Total time: 402.2 ms
  Avg per query: 50.3 ms

🚀 WITH CACHE:
  Total time: 201.9 ms
  Avg per query: 25.2 ms

📊 Query Embedding Cache Statistics
============================================================
Status: ✅ Enabled
Cache size: 4 / 100
Total requests: 8
  Hits: 4 (50.0%)
  Misses: 4
Performance:
  Avg compute time: 50.45 ms
  Avg cache time: 0.01 ms
  Speedup: 4575.0x
  Time saved: 201.7 ms
============================================================

📈 Overall Speedup: 1.99x
⏱️  Time Saved: 200.3 ms

💡 Key Takeaway:
  With 4/8 cache hits (50% hit rate)
  Achieved 1.99x overall speedup

======================================================================
Demo 4: Interactive Chat Session Simulation
======================================================================

Simulating conversation:
1. ❌ MISS | What is a database transaction?                    |  80.4 ms
2. ❌ MISS | Can you explain that more simply?                  |  80.3 ms
3. ✅ HIT  | What is a database transaction?                    |   0.0 ms
4. ❌ MISS | How do ACID properties work?                       |  80.2 ms
5. ✅ HIT  | What is a database transaction?                    |   0.0 ms
6. ❌ MISS | Give me an example                                 |  80.5 ms
7. ✅ HIT  | How do ACID properties work?                       |   0.0 ms

📊 Query Embedding Cache Statistics
============================================================
Status: ✅ Enabled
Cache size: 4 / 1000
Total requests: 7
  Hits: 3 (42.9%)
  Misses: 4
Performance:
  Avg compute time: 80.33 ms
  Avg cache time: 0.01 ms
  Speedup: 10639.6x
  Time saved: 241.0 ms
============================================================


💡 Key Takeaway:
  Users often rephrase or repeat questions in conversations
  Cache significantly improves response time for repeated queries
  Better user experience with faster retrieval

======================================================================
✨ All Demos Complete!
======================================================================

Summary of Benefits:
  ✅ 2-5x speedup on repeated queries
  ✅ Reduces computational load
  ✅ Improves user experience (faster responses)
  ✅ LRU eviction handles memory constraints
  ✅ Exact match + whitespace/case normalization

To Enable in TokenSmith:
  Set use_query_cache: true in config/config.yaml