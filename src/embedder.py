import numpy as np
from typing import List, Union, Optional
from llama_cpp import Llama
from tqdm import tqdm

# Import cache 
try:
    from src.query_cache import QueryEmbeddingCache
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    QueryEmbeddingCache = None

class SentenceTransformer:
    def __init__(self, model_path: str, n_ctx: int = 40960, n_threads: int = None, 
                 enable_cache: bool = False, cache_size: int = 1000):
        """
        Initialize with a local GGUF model file path.
        
        Args:
            model_path: Path to your local .gguf file
            n_ctx: Context window size (increased to match Qwen3 training context)
            n_threads: Number of threads to use (None = auto-detect)
            enable_cache: Whether to enable query embedding cache
            cache_size: Maximum number of cached embeddings
        """
        print(f"Loading model with n_ctx={n_ctx}, n_threads={n_threads}")
        
        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            embedding=True,
            verbose=False,
            n_batch=512,
            use_mmap=True,
            logits_all=True
        )
        self._embedding_dimension = None
        
        # Initialize cache if available and enabled
        self.cache: Optional[QueryEmbeddingCache] = None
        if enable_cache and CACHE_AVAILABLE:
            self.cache = QueryEmbeddingCache(max_size=cache_size, enabled=True)
            print(f"✅ Query cache enabled (size: {cache_size})")
        elif enable_cache and not CACHE_AVAILABLE:
            print("⚠️  Cache requested but query_cache module not available")
        
        _ = self.embedding_dimension
        print(f"Model loaded successfully. Embedding dimension: {self._embedding_dimension}")

    @property
    def embedding_dimension(self) -> int:
        """Get embedding dimension (cached after first call)."""
        if self._embedding_dimension is None:
            test_embedding = self.model.create_embedding("test")['data'][0]['embedding']
            self._embedding_dimension = len(test_embedding)
        return self._embedding_dimension

    def encode(self, 
               texts: Union[str, List[str]], 
               batch_size: int = 32,
               normalize: bool = False,
               device: str = None,
               show_progress_bar: bool = False,
               use_cache: bool = True,
               **kwargs) -> np.ndarray:
        """
        Encode texts to embeddings with batch processing.
        
        Args:
            texts: Single text or list of texts to encode
            batch_size: Number of texts to process at once
            normalize: Whether to normalize embeddings
            device: Compatibility param (ignored, CPU only)
            show_progress_bar: Whether to show progress bar
            use_cache: Whether to use cache for single-query encoding
            
        Returns:
            numpy.ndarray: Float32 embeddings array
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Single query - try cache
        if len(texts) == 1 and self.cache and use_cache:
            query = texts[0]
            
            def compute_embedding(q):
                embedding = self.model.create_embedding(q)['data'][0]['embedding']
                return np.array(embedding, dtype=np.float32)
            
            embedding, is_hit = self.cache.get(query, compute_fn=compute_embedding)
            
            if is_hit:
                # Cache hit - return cached embedding
                result = np.array([embedding], dtype=np.float32)
            else:
                # Cache miss - embedding was computed and cached
                result = np.array([embedding], dtype=np.float32)
            
            if normalize:
                norms = np.linalg.norm(result, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1e-12, norms)
                result = result / norms
            
            return result
            
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, -1)
        
        print(f"Encoding {len(texts)} texts with batch_size={batch_size}")
        
        embeddings = []
        
        # Process in batches (no cache for batch encoding)
        num_batches = (len(texts) + batch_size - 1) // batch_size

        for i in tqdm(range(num_batches), desc="Encoding", disable=not show_progress_bar):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]
            
            batch_embeddings = []
            for text in batch_texts:
                try:
                    embedding = self.model.create_embedding(text)['data'][0]['embedding']
                    batch_embeddings.append(embedding)
                except Exception as e:
                    print(f"Error encoding text: {e}")
                    batch_embeddings.append([0.0] * self.embedding_dimension)
			
            if len(batch_embeddings) != len(batch_texts):
                batch_embeddings.extend([[0.0] * self.embedding_dimension] * (len(batch_texts) - len(batch_embeddings)))
			
            embeddings.extend(batch_embeddings)
                
        vecs = np.array(embeddings, dtype=np.float32)
        
        # Normalize if requested
        if normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1e-12, norms)
            vecs = vecs / norms
            
        return vecs

    def embed_one(self, text: str, normalize: bool = False) -> List[float]:
        """Encode single text and return as list."""
        return self.encode([text], normalize=normalize)[0].tolist()

    def get_sentence_embedding_dimension(self) -> int:
        """Get the dimension of embeddings (compatibility method)."""
        return self.embedding_dimension

    def get_cache_stats(self):
        """Get cache statistics if cache is enabled."""
        if self.cache:
            return self.cache.get_stats()
        return None
    
    def print_cache_stats(self):
        """Print cache statistics if cache is enabled."""
        if self.cache:
            self.cache.print_stats()
        else:
            print("Cache is not enabled")
    
    def clear_cache(self):
        """Clear cache if enabled."""
        if self.cache:
            self.cache.clear()
            print("✅ Cache cleared")
        else:
            print("Cache is not enabled")
