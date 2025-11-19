**Modifications to TokenSmith RAG system**

Query Type Classification - Dynamic retrieval weights based on query intent
Adaptive Context Window - Smart chunk selection (3-10 vs fixed 5)
Query Embedding Cache - LRU cache for repeated queries

src/planning/\
├── planner.py              # Abstract query planner interface \
└── heuristics.py           # Keyword-based classifier 

src/\
├── adaptive_context.py     # Dynamic chunk selection \
└── query_cache.py          # LRU embedding cache 

**Integration Points**

src/main.py\
Apply query planner before retrieval\
Apply adaptive context after ranking\
Initialize cache-enabled retrievers

src/config.py \
Add use_query_planner: bool\
Add use_adaptive_context: bool\
Add use_query_cache: bool\
Add adaptive context parameters

src/embedder.py \
Cache-aware encoding method\
Pass cache to encoding calls

src/retriever.py \
Accept optional cache parameter\
Use cache in retrieve methods

src/instrumentation/logging.py \
Added log_event() method\
Support arbitrary event metadata

*Configuration Toggles*
```
# config/config_all_improvements.yaml
use_query_planner: true      # Feature #1
use_adaptive_context: true   # Feature #2
use_query_cache: true        # Feature #3
```
Rollback: Set all to false → reverts to baseline

*Run benchmark:*
```
python run_qa_benchmark.py \
  --questions data/starter_questions.yaml \
  --index_prefix textbook_index \
  --baseline_config config/config.yaml \
  --improved_config config/config_all_improvements.yaml \
  --outdir results
```