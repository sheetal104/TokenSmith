#!/usr/bin/env python3
"""
Run 12 questions against TokenSmith in two modes:
  (A) Baseline (improvements disabled)
  (B) Improvements enabled (planner + adaptive context + cache)

Outputs:
  - results/qa_baseline.json
  - results/qa_improved.json
  - results/qa_comparison.md (side-by-side for your report)
  - results/qa_summary.md (setup details: dataset, model, configs)

Usage (inside WSL venv):
  python run_qa_benchmark.py \
    --questions data/starter_questions.yaml \
    --index_prefix textbook_index \
    --baseline_config config/config.yaml \
    --improved_config config/config_all_improvements.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from datetime import datetime

from src.config import QueryPlanConfig
from src.instrumentation.logging import init_logger, get_logger
from src.ranking.ranker import EnsembleRanker
from src.retriever import load_artifacts, BM25Retriever, FAISSRetriever
from src.main import get_answer


def read_questions(path: Path) -> List[str]:
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "questions" in data:
            qs = [str(q).strip() for q in data["questions"] if str(q).strip()]
        else:
            qs = [str(q).strip() for q in data if str(q).strip()]
    else:
        qs = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return qs[:12]


def load_artifacts_once(cfg: QueryPlanConfig, index_prefix: str) -> Dict[str, Any]:
    artifacts_dir = cfg.make_artifacts_directory()
    faiss_index, bm25_index, chunks, sources = load_artifacts(
        artifacts_dir=artifacts_dir, index_prefix=index_prefix
    )
    retrievers = [
        FAISSRetriever(
            faiss_index, cfg.embed_model, n_ctx=cfg.embed_n_ctx, enable_cache=cfg.use_query_cache
        ),
        BM25Retriever(bm25_index),
    ]
    ranker = EnsembleRanker(
        ensemble_method=cfg.ensemble_method,
        weights=cfg.ranker_weights,
        rrf_k=int(cfg.rrf_k),
    )
    return {
        "chunks": chunks, 
        "sources": sources, 
        "retrievers": retrievers, 
        "ranker": ranker,
        "num_chunks": len(chunks),
        "num_sources": len(sources)
    }


def make_args(model_path: str | None, system_prompt_mode: str, index_prefix: str):
    class A: ...
    a = A()
    a.model_path = model_path
    a.system_prompt_mode = system_prompt_mode
    a.index_prefix = index_prefix
    return a


def run_mode(cfg_path: Path, questions: List[str], index_prefix: str, mode_name: str) -> Tuple[QueryPlanConfig, List[Dict[str, Any]], Dict[str, Any]]:
    print(f"\n{'='*60}")
    print(f"Running {mode_name} mode...")
    print(f"{'='*60}")
    
    cfg = QueryPlanConfig.from_yaml(cfg_path)
    init_logger(cfg)
    logger = get_logger()
    
    planner = None
    if cfg.use_query_planner:
        from src.planning.heuristics import HeuristicQueryPlanner
        planner = HeuristicQueryPlanner(cfg)
        print(f"✓ Query planner enabled")
    
    artifacts = load_artifacts_once(cfg, index_prefix)
    print(f"✓ Loaded {artifacts['num_chunks']} chunks from {artifacts['num_sources']} sources")
    
    args = make_args(None, cfg.system_prompt_mode, index_prefix)
    
    results = []
    query_timings = []  # Track timing for each query
    
    for i, q in enumerate(questions, 1):
        print(f"  [{i}/{len(questions)}] {q[:60]}...")
        
        start_time = time.time()
        qcfg = planner.plan(q) if planner else cfg
        ans = get_answer(q, qcfg, args, logger, artifacts=artifacts)
        elapsed = time.time() - start_time
        
        results.append({
            "question": q,
            "answer": ans or "",
            "time_seconds": elapsed
        })
        query_timings.append(elapsed)
        
        print(f"      ⏱️  {elapsed:.2f}s")
    
    # Get cache stats if available
    cache_stats = {}
    if cfg.use_query_cache and artifacts.get('retrievers'):
        faiss_retriever = artifacts['retrievers'][0]  # FAISSRetriever
        if hasattr(faiss_retriever, 'embedder') and hasattr(faiss_retriever.embedder, 'cache'):
            cache = faiss_retriever.embedder.cache
            if cache and hasattr(cache, 'hits'):
                cache_stats = {
                    "hits": cache.hits,
                    "misses": cache.misses,
                    "hit_rate": cache.hits / (cache.hits + cache.misses) if (cache.hits + cache.misses) > 0 else 0,
                    "total_queries": cache.hits + cache.misses
                }
                print(f"\n📊 Cache Stats: {cache.hits} hits / {cache.misses} misses ({cache_stats['hit_rate']*100:.1f}% hit rate)")
    
    stats = {
        "num_chunks": artifacts['num_chunks'],
        "num_sources": artifacts['num_sources'],
        "planner_enabled": cfg.use_query_planner,
        "adaptive_context_enabled": cfg.use_adaptive_context,
        "cache_enabled": cfg.use_query_cache,
        "cache_stats": cache_stats,
        "query_timings": query_timings,
        "avg_time": sum(query_timings) / len(query_timings) if query_timings else 0,
        "total_time": sum(query_timings)
    }
    
    return cfg, results, stats


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_comparison_md(path: Path, baseline: List[Dict[str, str]], improved: List[Dict[str, str]]) -> None:
    lines: List[str] = []
    lines.append("# TokenSmith Q/A Comparison (Baseline vs Improvements)")
    lines.append("")
    lines.append("This document contains side-by-side answers from TokenSmith with and without improvements enabled.")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    for i, (b, m) in enumerate(zip(baseline, improved), start=1):
        # Check if this is a repeated question (cache test)
        is_repeat = False
        repeat_note = ""
        if i > 2:  # Only check after first few questions
            for j in range(i-1):
                if baseline[j].get("question") == b.get("question"):
                    is_repeat = True
                    repeat_note = f" (REPEAT of Q{j+1} - Cache Test)"
                    break
        
        lines.append(f"## Q{i}{repeat_note}")
        lines.append("")
        lines.append("### Question")
        lines.append(b.get("question", ""))
        lines.append("")
        
        # Show timing if available
        b_time = b.get("time_seconds", 0)
        m_time = m.get("time_seconds", 0)
        if b_time > 0 or m_time > 0:
            lines.append("### Execution Time")
            lines.append(f"- **Baseline:** {b_time:.2f}s")
            lines.append(f"- **Improved:** {m_time:.2f}s")
            if is_repeat and m_time > 0:
                lines.append(f"- **Note:** Cache should make this faster (embedding reused)")
            lines.append("")
        
        lines.append("### Baseline Answer")
        lines.append("```")
        lines.append((b.get("answer") or "").strip() or "(no output)")
        lines.append("```")
        lines.append("")
        
        lines.append("### Improved Answer")
        lines.append("```")
        lines.append((m.get("answer") or "").strip() or "(no output)")
        lines.append("```")
        lines.append("")
        
        lines.append("### Your Evaluation")
        lines.append("- **Baseline score (1-5):** ")
        lines.append("- **Improved score (1-5):** ")
        lines.append("- **Comments:** ")
        lines.append("")
        lines.append("---")
        lines.append("")
    
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary_md(path: Path, baseline_cfg: QueryPlanConfig, improved_cfg: QueryPlanConfig, 
                     baseline_stats: Dict, improved_stats: Dict, questions_file: Path,
                     baseline_results: List[Dict] = None, improved_results: List[Dict] = None) -> None:
    lines: List[str] = []
    lines.append("# TokenSmith Q/A Benchmark - Setup Summary")
    lines.append("")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    lines.append("## 📚 Dataset")
    lines.append("")
    lines.append(f"- **Source document:** `data/book_without_image.md`")
    lines.append(f"- **Number of chunks:** {baseline_stats['num_chunks']}")
    lines.append(f"- **Number of sources:** {baseline_stats['num_sources']}")
    lines.append(f"- **Chunking strategy:** {baseline_cfg.chunk_config.to_string()}")
    lines.append("")
    
    lines.append("## 🤖 Models")
    lines.append("")
    lines.append(f"- **Generation model:** `{baseline_cfg.model_path}`")
    lines.append(f"- **Embedding model:** `{baseline_cfg.embed_model}`")
    lines.append(f"- **Embedding context window:** {baseline_cfg.embed_n_ctx} tokens")
    lines.append("")
    
    lines.append("## ❓ Questions")
    lines.append("")
    lines.append(f"- **Questions file:** `{questions_file}`")
    lines.append(f"- **Number of questions:** 12")
    lines.append("- **Question types:** True/False, Definitions, Comparisons, Design choices, Procedural")
    lines.append("")
    
    lines.append("## ⚙️ Configuration Comparison")
    lines.append("")
    lines.append("| Setting | Baseline | Improved |")
    lines.append("|---------|----------|----------|")
    lines.append(f"| **Top-k chunks** | {baseline_cfg.top_k} | {improved_cfg.top_k} |")
    lines.append(f"| **Pool size** | {baseline_cfg.pool_size} | {improved_cfg.pool_size} |")
    lines.append(f"| **Ensemble method** | {baseline_cfg.ensemble_method} | {improved_cfg.ensemble_method} |")
    lines.append(f"| **FAISS weight** | {baseline_cfg.ranker_weights.get('faiss', 0):.1f} | {improved_cfg.ranker_weights.get('faiss', 0):.1f} |")
    lines.append(f"| **BM25 weight** | {baseline_cfg.ranker_weights.get('bm25', 0):.1f} | {improved_cfg.ranker_weights.get('bm25', 0):.1f} |")
    lines.append(f"| **Max generation tokens** | {baseline_cfg.max_gen_tokens} | {improved_cfg.max_gen_tokens} |")
    lines.append(f"| **System prompt** | {baseline_cfg.system_prompt_mode} | {improved_cfg.system_prompt_mode} |")
    lines.append("")
    
    lines.append("## 🚀 Improvements Enabled")
    lines.append("")
    lines.append("| Improvement | Baseline | Improved |")
    lines.append("|-------------|----------|----------|")
    lines.append(f"| **Query Type Classification** | {'✅' if baseline_stats['planner_enabled'] else '❌'} | {'✅' if improved_stats['planner_enabled'] else '❌'} |")
    lines.append(f"| **Adaptive Context Window** | {'✅' if baseline_stats['adaptive_context_enabled'] else '❌'} | {'✅' if improved_stats['adaptive_context_enabled'] else '❌'} |")
    lines.append(f"| **Query Embedding Cache** | {'✅' if baseline_stats['cache_enabled'] else '❌'} | {'✅' if improved_stats['cache_enabled'] else '❌'} |")
    lines.append("")

    if improved_stats.get('cache_stats'):
        cache = improved_stats['cache_stats']
        lines.append("### Cache Performance (Improved Mode)")
        lines.append("")
        lines.append(f"- **Total queries:** {cache['total_queries']}")
        lines.append(f"- **Cache hits:** {cache['hits']}")
        lines.append(f"- **Cache misses:** {cache['misses']}")
        lines.append(f"- **Hit rate:** {cache['hit_rate']*100:.1f}%")
        lines.append("")
        lines.append("**Expected Results:**")
        lines.append("- Questions 11-12 should hit cache (repeats of Q1 & Q5)")
        lines.append("- Cache hits mean ~50-100ms faster embedding computation")
        lines.append("- Note: Final answers may still vary due to LLM non-determinism")
        lines.append("")
    
    # Add timing comparison
    if baseline_stats.get('query_timings') and improved_stats.get('query_timings'):
        lines.append("## ⏱️ Performance Comparison")
        lines.append("")
        lines.append("| Metric | Baseline | Improved | Speedup |")
        lines.append("|--------|----------|----------|---------|")
        
        b_avg = baseline_stats['avg_time']
        m_avg = improved_stats['avg_time']
        speedup = b_avg / m_avg if m_avg > 0 else 1.0
        
        lines.append(f"| **Avg time per query** | {b_avg:.2f}s | {m_avg:.2f}s | {speedup:.2f}x |")
        lines.append(f"| **Total time** | {baseline_stats['total_time']:.2f}s | {improved_stats['total_time']:.2f}s | {baseline_stats['total_time']/improved_stats['total_time'] if improved_stats['total_time'] > 0 else 1:.2f}x |")
        lines.append("")
        
        # Show per-query timings for cache analysis
        lines.append("### Per-Query Timings (Improved Mode)")
        lines.append("")
        lines.append("| Q# | Time (s) | Notes |")
        lines.append("|----|----------|-------|")
        if improved_results:
            for i, result in enumerate(improved_results, 1):
                timing = result.get('time_seconds', 0)
                note = ""
                if i in [11, 12]:
                    note = "⚡ Cache test"
                lines.append(f"| Q{i} | {timing:.2f} | {note} |")
        else:
            for i, timing in enumerate(improved_stats['query_timings'], 1):
                note = ""
                if i in [11, 12]:
                    note = "⚡ Cache test"
                lines.append(f"| Q{i} | {timing:.2f} | {note} |")
    
    if improved_stats['adaptive_context_enabled']:
        lines.append("### Adaptive Context Settings (Improved)")
        lines.append("")
        lines.append(f"- **Min chunks:** {improved_cfg.adaptive_min_chunks}")
        lines.append(f"- **Max chunks:** {improved_cfg.adaptive_max_chunks}")
        lines.append(f"- **Relevance threshold:** {improved_cfg.adaptive_relevance_threshold}")
        lines.append(f"- **Variance threshold:** {improved_cfg.adaptive_quality_variance_threshold}")
        lines.append(f"- **Max tokens:** {improved_cfg.adaptive_max_tokens}")
        lines.append("")
    
    if improved_stats['cache_enabled']:
        lines.append("### Cache Settings (Improved)")
        lines.append("")
        lines.append(f"- **Cache size:** {improved_cfg.query_cache_size} entries")
        lines.append("")
    
    lines.append("## 📊 Output Files")
    lines.append("")
    lines.append("- `results/qa_baseline.json` - Baseline answers (JSON)")
    lines.append("- `results/qa_improved.json` - Improved answers (JSON)")
    lines.append("- `results/qa_comparison.md` - Side-by-side comparison")
    lines.append("- `results/qa_summary.md` - This file")
    lines.append("")
    
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run 12 questions with and without improvements")
    ap.add_argument("--questions", type=Path, required=True, help="Questions file (.yaml or .txt)")
    ap.add_argument("--index_prefix", type=str, default="textbook_index")
    ap.add_argument("--baseline_config", type=Path, default=Path("config/config.yaml"))
    ap.add_argument("--improved_config", type=Path, default=Path("config/config_all_improvements.yaml"))
    ap.add_argument("--outdir", type=Path, default=Path("results"))
    args = ap.parse_args()

    if not args.questions.exists():
        print(f"ERROR: Questions file not found: {args.questions}")
        sys.exit(1)
    
    if not args.baseline_config.exists():
        print(f"ERROR: Baseline config not found: {args.baseline_config}")
        sys.exit(1)
    
    if not args.improved_config.exists():
        print(f"ERROR: Improved config not found: {args.improved_config}")
        sys.exit(1)

    questions = read_questions(args.questions)
    if not questions:
        print("ERROR: No questions found.")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"TokenSmith Q/A Benchmark")
    print(f"{'='*60}")
    print(f"Questions: {len(questions)} from {args.questions}")
    print(f"Baseline config: {args.baseline_config}")
    print(f"Improved config: {args.improved_config}")
    print(f"Output directory: {args.outdir}")

    # Run baseline
    baseline_cfg, baseline_results, baseline_stats = run_mode(
        args.baseline_config, questions, args.index_prefix, "BASELINE"
    )
    
    # Run improved
    improved_cfg, improved_results, improved_stats = run_mode(
        args.improved_config, questions, args.index_prefix, "IMPROVED"
    )

    # Save JSON
    write_json(args.outdir / "qa_baseline.json", {
        "config": str(args.baseline_config),
        "stats": baseline_stats,
        "results": baseline_results
    })
    write_json(args.outdir / "qa_improved.json", {
        "config": str(args.improved_config),
        "stats": improved_stats,
        "results": improved_results
    })
    
    # Save comparison Markdown
    write_comparison_md(args.outdir / "qa_comparison.md", baseline_results, improved_results)
    
    # Save summary Markdown
    write_summary_md(
        args.outdir / "qa_summary.md",
        baseline_cfg, improved_cfg,
        baseline_stats, improved_stats,
        args.questions,
        baseline_results, improved_results
    )

    print(f"\n{'='*60}")
    print("✅ Benchmark complete!")
    print(f"{'='*60}")
    print(f"📄 Saved: {args.outdir/'qa_baseline.json'}")
    print(f"📄 Saved: {args.outdir/'qa_improved.json'}")
    print(f"📝 Saved: {args.outdir/'qa_comparison.md'}")
    print(f"📋 Saved: {args.outdir/'qa_summary.md'}")
    print(f"\n👉 Review '{args.outdir/'qa_comparison.md'}' and fill in your evaluations!")


if __name__ == "__main__":
    main()
