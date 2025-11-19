import argparse
import pathlib
import sys
from typing import Dict, Optional

from src.config import QueryPlanConfig
from src.generator import answer
from src.index_builder import build_index
from src.instrumentation.logging import init_logger, get_logger, RunLogger
from src.ranking.ranker import EnsembleRanker
from src.preprocessing.chunking import DocumentChunker
from src.retriever import apply_seg_filter, BM25Retriever, FAISSRetriever, load_artifacts
from src.query_enhancement import generate_hypothetical_document


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the application."""
    parser = argparse.ArgumentParser(
        description="Welcome to TokenSmith!"
    )

    # Required arguments
    parser.add_argument(
        "mode",
        choices=["index", "chat"],
        help="operation mode: 'index' to build index, 'chat' to query"
    )

    # Common arguments
    parser.add_argument(
        "--pdf_dir",
        default="data/chapters/",
        help="directory containing PDF files (default: %(default)s)"
    )
    parser.add_argument(
        "--index_prefix",
        default="textbook_index",
        help="prefix for generated index files (default: %(default)s)"
    )
    parser.add_argument(
        "--model_path",
        help="path to generation model (uses config default if not specified)"
    )
    parser.add_argument(
        "--system_prompt_mode",
        choices=["baseline", "tutor", "concise", "detailed"],
        default="baseline",
        help="system prompt mode (choices: baseline, tutor, concise, detailed)"
    )
    
    # Indexing-specific arguments
    indexing_group = parser.add_argument_group("indexing options")
    indexing_group.add_argument(
        "--pdf_range",
        metavar="START-END",
        help="specific range of PDFs to index (e.g., '27-33')"
    )
    indexing_group.add_argument(
        "--keep_tables",
        action="store_true",
        help="include tables in the index"
    )
    indexing_group.add_argument(
        "--visualize",
        action="store_true",
        help="generate visualizations during indexing"
    )

    return parser.parse_args()


def run_index_mode(args: argparse.Namespace, cfg: QueryPlanConfig):
    """Handles the logic for building the index."""

    # Robust range filtering
    try:
        if args.pdf_range:
            start, end = map(int, args.pdf_range.split("-"))
            pdf_paths = [f"{i}.pdf" for i in range(start, end + 1)] # Inclusive range
            print(f"Indexing PDFs in range: {start}-{end}")
        else:
            pdf_paths = None
    except ValueError:
        print(f"ERROR: Invalid format for --pdf_range. Expected 'start-end', but got '{args.pdf_range}'.")
        sys.exit(1)
    
    strategy = cfg.make_strategy()
    chunker = DocumentChunker(strategy=strategy, keep_tables=args.keep_tables)
    
    artifacts_dir = cfg.make_artifacts_directory()

    build_index(
        markdown_file="data/book_without_image.md",
        chunker=chunker,
        chunk_config=cfg.chunk_config,
        embedding_model_path=cfg.embed_model,
        artifacts_dir=artifacts_dir,
        index_prefix=args.index_prefix,
        do_visualize=args.visualize,
    )


def get_answer(
    question: str,
    cfg: QueryPlanConfig,
    args: argparse.Namespace,
    logger: "RunLogger",
    artifacts: Optional[Dict] = None,
    golden_chunks: Optional[list] = None,
    is_test_mode: bool = False
) -> str:
    """
    Run a single query through the pipeline.
    """    
    chunks = artifacts["chunks"]
    sources = artifacts["sources"]
    retrievers = artifacts["retrievers"]
    ranker = artifacts["ranker"]
    
    logger.log_query_start(question)
    
    # Step 1: Get chunks (golden, retrieved, or none)
    chunks_info = None
    hyde_query = None
    if golden_chunks and cfg.use_golden_chunks:
        # Use provided golden chunks
        ranked_chunks = golden_chunks
    elif cfg.disable_chunks:
        # No chunks - baseline mode
        ranked_chunks = []
    else:
        # Step 0: Query Enhancement (HyDE)
        retrieval_query = question
        if cfg.use_hyde:
            model_path = args.model_path or cfg.model_path
            hypothetical_doc = generate_hypothetical_document(
                question, model_path, max_tokens=cfg.hyde_max_tokens
            )
            retrieval_query = hypothetical_doc
            hyde_query = hypothetical_doc
            # print(f"🔍 HyDE query: {hypothetical_doc}")
        
        # Step 1: Retrieval
        pool_n = max(cfg.pool_size, cfg.top_k + 10)
        raw_scores: Dict[str, Dict[int, float]] = {}
        for retriever in retrievers:
            raw_scores[retriever.name] = retriever.get_scores(retrieval_query, pool_n, chunks)
        # TODO: Fix retrieval logging.
        
        # Step 2: Ranking
        ordered = ranker.rank(raw_scores=raw_scores)
        topk_idxs = apply_seg_filter(cfg, chunks, ordered)
        
        # Step 2.5: Adaptive Context Window (if enabled)
        adaptive_metadata = None
        if cfg.use_adaptive_context:
            from src.adaptive_context import AdaptiveContextManager
            
            # Combine scores for adaptive selection
            combined_scores = {}
            for retriever_name, scores in raw_scores.items():
                weight = cfg.ranker_weights.get(retriever_name, 0.5)
                for idx, score in scores.items():
                    combined_scores[idx] = combined_scores.get(idx, 0) + weight * score
            
            # Create manager
            manager = AdaptiveContextManager(
                min_chunks=cfg.adaptive_min_chunks,
                max_chunks=cfg.adaptive_max_chunks,
                relevance_threshold=cfg.adaptive_relevance_threshold,
                quality_variance_threshold=cfg.adaptive_quality_variance_threshold,
                max_tokens=cfg.adaptive_max_tokens
            )
            
            # Estimate complexity
            complexity = manager.estimate_query_complexity(question)
            
            # Prepare ranked chunks with scores
            ranked_with_text = [(idx, chunks[idx]) for idx in topk_idxs]
            
            # Select adaptive chunks
            ranked_chunks, adaptive_metadata = manager.select_chunks(
                ranked_with_text, combined_scores, complexity
            )
            
            # Update topk_idxs to reflect adaptive selection
            # (for logging purposes - use indices of selected chunks)
            topk_idxs = topk_idxs[:len(ranked_chunks)]
            
            # Log adaptive decision
            logger.log_event(
                "adaptive_context",
                f"Selected {len(ranked_chunks)} chunks ({adaptive_metadata['reason']})"
            )
        else:
            ranked_chunks = [chunks[i] for i in topk_idxs]
        
        logger.log_chunks_used(topk_idxs, chunks, sources)
        
        # Capture chunk info if in test mode
        if is_test_mode:
            # Compute individual ranker ranks
            faiss_scores = raw_scores.get("faiss", {})
            bm25_scores = raw_scores.get("bm25", {})
            
            faiss_ranked = sorted(faiss_scores.keys(), key=lambda i: faiss_scores[i], reverse=True)  # Higher score = better
            bm25_ranked = sorted(bm25_scores.keys(), key=lambda i: bm25_scores[i], reverse=True)  # Higher score = better
            
            faiss_ranks = {idx: rank + 1 for rank, idx in enumerate(faiss_ranked)}
            bm25_ranks = {idx: rank + 1 for rank, idx in enumerate(bm25_ranked)}
            
            chunks_info = []
            for rank, idx in enumerate(topk_idxs, 1):
                chunks_info.append({
                    "rank": rank,
                    "chunk_id": idx,
                    "content": chunks[idx],
                    "faiss_score": faiss_scores.get(idx, 0),
                    "faiss_rank": faiss_ranks.get(idx, 0),
                    "bm25_score": bm25_scores.get(idx, 0),
                    "bm25_rank": bm25_ranks.get(idx, 0),
                })
        
        # Step 3: Final Re-ranking (if enabled)
        # Disabled till we fix the core pipeline
        # ranked_chunks = rerank(question, ranked_chunks, mode=cfg.rerank_mode, top_n=cfg.top_k)
    
    # Step 4: Generation
    model_path = args.model_path or cfg.model_path
    system_prompt = args.system_prompt_mode or cfg.system_prompt_mode
    ans = answer(
        question, 
        ranked_chunks, 
        model_path, 
        max_tokens=cfg.max_gen_tokens, 
        system_prompt_mode=system_prompt
    )
    
    if is_test_mode:
        return ans, chunks_info, hyde_query
    return ans


def run_chat_session(args: argparse.Namespace, cfg: QueryPlanConfig):
    """
    Initializes artifacts and runs the main interactive chat loop.
    """
    logger = get_logger()
    
    # Initialize query planner if enabled
    planner = None
    if cfg.use_query_planner:
        from src.planning.heuristics import HeuristicQueryPlanner
        planner = HeuristicQueryPlanner(cfg)
        print("🧠 Query planner enabled - will adapt strategy based on question type")

    # Load artifacts, initialize retrievers and rankers once before the loop.
    print("Welcome to Tokensmith! Initializing chat...")
    try:
        # Disabled till we fix the core pipeline
        # cfg = planner.plan(q)
        artifacts_dir = cfg.make_artifacts_directory()
        faiss_index, bm25_index, chunks, sources = load_artifacts(
            artifacts_dir=artifacts_dir, 
            index_prefix=args.index_prefix
        )

        retrievers = [
            FAISSRetriever(
                faiss_index, 
                cfg.embed_model, 
                n_ctx=cfg.embed_n_ctx,
                enable_cache=cfg.use_query_cache
            ),
            BM25Retriever(bm25_index)
        ]
        ranker = EnsembleRanker(
            ensemble_method=cfg.ensemble_method,
            weights=cfg.ranker_weights,
            rrf_k=int(cfg.rrf_k)
        )
        
        # Package artifacts for reuse
        artifacts = {
            "chunks": chunks,
            "sources": sources,
            "retrievers": retrievers,
            "ranker": ranker
        }
    except Exception as e:
        print(f"ERROR: Failed to initialize chat artifacts: {e}")
        print("Please ensure you have run 'index' mode first.")
        sys.exit(1)

    print("Initialization complete. You can start asking questions!")
    print("Type 'exit' or 'quit' to end the session.")
    while True:
        try:
            q = input("\nAsk > ").strip()
            if not q:
                continue
            if q.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break

            # Apply query planner if enabled
            query_cfg = cfg
            if planner:
                query_cfg = planner.plan(q)
                classification = query_cfg.query_classification
                print(f"  🔍 Detected: {classification['type']} query "
                      f"(confidence: {classification['confidence']:.2f})")
                print(f"  ⚙️  Adjusted weights: FAISS={classification['adjusted_weights']['faiss']:.1f}, "
                      f"BM25={classification['adjusted_weights']['bm25']:.1f}")

            # Use the single query function
            ans = get_answer(q, query_cfg, args, logger=logger, artifacts=artifacts)

            print("\n=================== START OF ANSWER ===================")
            print(ans.strip() if ans and ans.strip() else "(No output from model)")
            print("\n==================== END OF ANSWER ====================")
            logger.log_generation(ans, {"max_tokens": cfg.max_gen_tokens, "model_path": args.model_path or cfg.model_path})

        except KeyboardInterrupt:
            print("\nGoodbye!")
            
            # Print cache stats if enabled
            if cfg.use_query_cache and retrievers:
                for retriever in retrievers:
                    if hasattr(retriever, 'embedder') and hasattr(retriever.embedder, 'print_cache_stats'):
                        retriever.embedder.print_cache_stats()
            
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            logger.log_error(str(e))
            break

    # TODO: Fix completion logging.
    # logger.log_query_complete()


def main():
    """Main entry point for the script."""
    args = parse_args()

    # Config loading
    config_path = pathlib.Path("config/config.yaml")
    cfg = None
    if config_path.exists():
        cfg = QueryPlanConfig.from_yaml(config_path)

    if cfg is None:
        raise FileNotFoundError(
            "No config file provided and no fallback found at config/ or ~/.config/tokensmith/"
        )

    init_logger(cfg)

    if args.mode == "index":
        run_index_mode(args, cfg)
    elif args.mode == "chat":
        run_chat_session(args, cfg)


if __name__ == "__main__":
    main()
