from src.config import QueryPlanConfig
from copy import deepcopy
from typing import Dict, Tuple
import re

from src.planning.planner import QueryPlanner

"""
Heuristic Query Planner
-----------------------
Automatically adjusts retrieval strategy based on query type:
  • Definition queries → Favor BM25 (keyword matching for precise terms)
  • Explanatory queries → Favor FAISS (semantic similarity for concepts)
  • Procedural queries → Larger pool + balanced weights (scattered information)
  • Comparative queries → Balanced weights + larger pool (multiple concepts)
"""

class HeuristicQueryPlanner(QueryPlanner):
    # Enhanced classification patterns
    DEFINITION_PATTERNS = [
        r'\bwhat\s+is\b', r'\bdefine\b', r'\bdefinition\b', 
        r'\bmeaning\s+of\b', r'\bterm\b', r'\brefers?\s+to\b'
    ]
    
    EXPLANATORY_PATTERNS = [
        r'\bwhy\b', r'\bexplain\b', r'\bhow\s+does\b', r'\bwhat\s+causes\b',
        r'\breason\b', r'\bpurpose\b', r'\bbenefits?\b'
    ]
    
    PROCEDURAL_PATTERNS = [
        r'\bhow\s+to\b', r'\bsteps\b', r'\bprocedure\b', r'\balgorithm\b',
        r'\bprocess\b', r'\bmethod\b', r'\bimplement\b'
    ]
    
    COMPARATIVE_PATTERNS = [
        r'\bdifference\b', r'\bcompare\b', r'\bversus\b', r'\bvs\.?\b',
        r'\bbetter\b', r'\badvantages?\b', r'\bdisadvantages?\b'
    ]

    @property
    def name(self) -> str:
        return "HeuristicBasedPlanner"

    def __init__(self, base_cfg: QueryPlanConfig):
        super().__init__(base_cfg)
        self.base_cfg = deepcopy(base_cfg)
        self.classification_stats = {
            "definition": 0,
            "explanatory": 0,
            "procedural": 0,
            "comparative": 0,
            "other": 0
        }

    def classify(self, query: str) -> Tuple[str, float]:
        """
        Classify query type with confidence score.
        
        Returns:
            Tuple[str, float]: (query_type, confidence)
        """
        q = query.lower()
        scores = {
            "definition": 0,
            "explanatory": 0,
            "procedural": 0,
            "comparative": 0
        }
        
        # Score each category based on pattern matches
        for pattern in self.DEFINITION_PATTERNS:
            if re.search(pattern, q):
                scores["definition"] += 1
        
        for pattern in self.EXPLANATORY_PATTERNS:
            if re.search(pattern, q):
                scores["explanatory"] += 1
        
        for pattern in self.PROCEDURAL_PATTERNS:
            if re.search(pattern, q):
                scores["procedural"] += 1
        
        for pattern in self.COMPARATIVE_PATTERNS:
            if re.search(pattern, q):
                scores["comparative"] += 1
        
        # Get the type with highest score
        max_score = max(scores.values())
        if max_score == 0:
            return "other", 0.0
        
        query_type = max(scores.items(), key=lambda x: x[1])[0]
        confidence = max_score / len(query.split())  # Normalize by query length
        
        # Update stats
        self.classification_stats[query_type] += 1
        
        return query_type, min(confidence, 1.0)

    def plan(self, query: str) -> QueryPlanConfig:
        """
        Generate optimized query plan based on query type.
        
        Returns plan with adjusted weights, pool size, and metadata for validation.
        """
        kind, confidence = self.classify(query)
        cfg = deepcopy(self.base_cfg)
        
        # Store classification metadata for analysis
        cfg.query_classification = {
            "type": kind,
            "confidence": confidence,
            "original_weights": dict(self.base_cfg.ranker_weights),
            "original_pool_size": self.base_cfg.pool_size
        }

        if kind == "definition":
            # Definitions need precise keyword matching
            cfg.ranker_weights = {"faiss": 0.3, "bm25": 0.7}
            cfg.pool_size = self.base_cfg.pool_size  # Standard pool

        elif kind == "explanatory":
            # Explanations benefit from semantic similarity
            cfg.ranker_weights = {"faiss": 0.7, "bm25": 0.3}
            cfg.pool_size = self.base_cfg.pool_size

        elif kind == "procedural":
            # Procedures may span multiple chunks
            cfg.pool_size = max(cfg.pool_size, cfg.top_k * 5)
            cfg.ranker_weights = {"faiss": 0.6, "bm25": 0.4}

        elif kind == "comparative":
            # Comparisons need multiple relevant chunks
            cfg.pool_size = max(cfg.pool_size, cfg.top_k * 4)
            cfg.ranker_weights = {"faiss": 0.5, "bm25": 0.5}

        else:
            # Default to balanced approach
            cfg.ranker_weights = {"faiss": 0.6, "bm25": 0.4}
            cfg.pool_size = self.base_cfg.pool_size

        # Store adjusted values for validation
        cfg.query_classification["adjusted_weights"] = dict(cfg.ranker_weights)
        cfg.query_classification["adjusted_pool_size"] = cfg.pool_size

        self._log_decision(cfg)
        return cfg
    
    def get_statistics(self) -> Dict[str, int]:
        """Return classification statistics for validation."""
        return self.classification_stats.copy()
