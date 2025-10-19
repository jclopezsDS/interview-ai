"""
Unified RAG pipeline implementation.

This module consolidates all RAG components including:
- Vector store operations (embedding, chunking, retrieval)
- Context processing (compression)
- Relevance filtering
- Context fusion
- Semantic caching

This provides a complete, cohesive pipeline for retrieval-augmented generation.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional, Dict, Callable, Iterator, Any
import faiss
import chromadb
from chromadb.config import Settings
from pathlib import Path
from functools import lru_cache
import numpy as np
import hashlib
import time
import re
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from concurrent.futures import ThreadPoolExecutor, as_completed
import nltk
from nltk.tokenize import sent_tokenize
import tiktoken
from difflib import SequenceMatcher
import threading

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')


# ================================
# Vector Store Components
# ================================

@dataclass
class VectorStoreConfig:
    faiss_index_path: str
    chroma_persist_dir: str
    embedding_dim: int = 1536
    faiss_index_type: str = "IVF"
    n_clusters: int = 100
    metadata_fields: List[str] = None


def setup_hybrid_vector_store(config: VectorStoreConfig) -> Tuple[faiss.Index, chromadb.Collection]:
    """Configure FAISS with hierarchical indexing and Chroma for metadata filtering.
    
    Args:
        config: VectorStoreConfig with paths and parameters
        
    Returns:
        Tuple[faiss.Index, chromadb.Collection]: FAISS index and Chroma collection
    """
    print("[RAG] Initializing hybrid vector store configuration")
    
    # Setup FAISS hierarchical index
    if config.faiss_index_type == "IVF":
        quantizer = faiss.IndexFlatL2(config.embedding_dim)
        faiss_index = faiss.IndexIVFFlat(quantizer, config.embedding_dim, config.n_clusters)
        print(f"[RAG] FAISS IVF index created: {config.n_clusters} clusters, {config.embedding_dim}D")
    else:
        faiss_index = faiss.IndexFlatL2(config.embedding_dim)
        print(f"[RAG] FAISS flat index created: {config.embedding_dim}D")
    
    # Load existing index if available
    if Path(config.faiss_index_path).exists():
        faiss_index = faiss.read_index(config.faiss_index_path)
        print(f"[RAG] Loaded existing FAISS index: {faiss_index.ntotal} vectors")
    
    # Setup Chroma for metadata filtering
    chroma_client = chromadb.PersistentClient(
        path=config.chroma_persist_dir,
        settings=Settings(anonymized_telemetry=False)
    )
    
    collection = chroma_client.get_or_create_collection(
        name="job_descriptions",
        metadata={"hnsw:space": "cosine"}
    )
    print(f"[RAG] Chroma collection ready: {collection.count()} documents")
    
    print("[SUCCESS] Hybrid vector store configured successfully")
    return faiss_index, collection


@dataclass
class EmbeddingConfig:
    primary_model: str = "text-embedding-3-large"
    fallback_model: str = "BAAI/bge-large-en-v1.5"
    openai_client: Optional[OpenAI] = None
    cache_size: int = 1000
    timeout_seconds: int = 10


class AdaptiveEmbeddings:
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.openai_client = config.openai_client
        self.fallback_model = None
        self._load_fallback_model()
        
    def _load_fallback_model(self) -> None:
        """Load local sentence transformer model for fallback."""
        try:
            self.fallback_model = SentenceTransformer(self.config.fallback_model)
            print(f"[RAG] Fallback model loaded: {self.config.fallback_model}")
        except Exception as e:
            print(f"[RAG] Warning: Fallback model failed to load: {e}")
    
    @lru_cache(maxsize=1000)
    def _get_cache_key(self, text: str, model: str) -> str:
        """Generate cache key for embeddings."""
        return hashlib.md5(f"{text}:{model}".encode()).hexdigest()
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings with fallback strategy."""
        print(f"[RAG] Embedding {len(texts)} texts with adaptive strategy")
        
        # Try primary OpenAI model
        try:
            start_time = time.time()
            response = self.openai_client.embeddings.create(
                input=texts,
                model=self.config.primary_model,
                timeout=self.config.timeout_seconds
            )
            embeddings = np.array([data.embedding for data in response.data])
            elapsed = time.time() - start_time
            print(f"[RAG] Primary embeddings completed: {elapsed:.2f}s, {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            print(f"[RAG] Primary model failed, using fallback: {str(e)[:50]}...")
            
            # Fallback to local model
            if self.fallback_model:
                start_time = time.time()
                embeddings = self.fallback_model.encode(texts, convert_to_numpy=True)
                elapsed = time.time() - start_time
                print(f"[RAG] Fallback embeddings completed: {elapsed:.2f}s, {embeddings.shape}")
                return embeddings
            else:
                raise Exception("Both primary and fallback embedding models failed")


def implement_adaptive_embeddings(openai_client: OpenAI) -> AdaptiveEmbeddings:
    """Deploy multiple embedding models with fallback strategy.
    
    Args:
        openai_client: Configured OpenAI client
        
    Returns:
        AdaptiveEmbeddings: Configured embedding system with fallback
    """
    print("[RAG] Initializing adaptive embeddings system")
    
    config = EmbeddingConfig(
        primary_model="text-embedding-3-large",
        fallback_model="BAAI/bge-large-en-v1.5",
        openai_client=openai_client,
        cache_size=1000,
        timeout_seconds=10
    )
    
    embedder = AdaptiveEmbeddings(config)
    
    # Test with sample texts
    test_texts = [
        "Senior Software Engineer position at tech startup",
        "Python developer with 3+ years experience required"
    ]
    
    test_embeddings = embedder.embed_texts(test_texts)
    print(f"[RAG] Test embeddings shape: {test_embeddings.shape}")
    print("[SUCCESS] Adaptive embeddings system ready")
    
    return embedder


@dataclass
class ChunkingConfig:
    similarity_threshold: float = 0.7
    min_chunk_size: int = 100
    max_chunk_size: int = 1000
    overlap_sentences: int = 1
    embedding_model: str = "all-MiniLM-L6-v2"


class SemanticChunker:
    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.embedding_model = SentenceTransformer(config.embedding_model)
        print(f"[RAG] Semantic chunker initialized with {config.embedding_model}")
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK."""
        sentences = sent_tokenize(text)
        # Clean and filter sentences
        cleaned = [s.strip() for s in sentences if len(s.strip()) > 10]
        return cleaned
    
    def _find_semantic_boundaries(self, sentences: List[str]) -> List[int]:
        """Find semantic boundaries using sentence similarity."""
        if len(sentences) < 2:
            return [0, len(sentences)]
        
        # Get sentence embeddings
        embeddings = self.embedding_model.encode(sentences)
        similarities = []
        
        # Calculate consecutive sentence similarities
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            similarities.append(sim)
        
        # Find boundaries where similarity drops below threshold
        boundaries = [0]
        for i, sim in enumerate(similarities):
            if sim < self.config.similarity_threshold:
                boundaries.append(i + 1)
        boundaries.append(len(sentences))
        
        return sorted(list(set(boundaries)))
    
    def _create_chunks_from_boundaries(self, sentences: List[str], boundaries: List[int]) -> List[str]:
        """Create text chunks from semantic boundaries."""
        chunks = []
        
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]
            
            # Add overlap from previous chunk
            if i > 0 and self.config.overlap_sentences > 0:
                overlap_start = max(0, boundaries[i] - self.config.overlap_sentences)
                chunk_sentences = sentences[overlap_start:end_idx]
            else:
                chunk_sentences = sentences[start_idx:end_idx]
            
            chunk_text = ' '.join(chunk_sentences)
            
            # Validate chunk size
            if (self.config.min_chunk_size <= len(chunk_text) <= self.config.max_chunk_size):
                chunks.append(chunk_text)
            elif len(chunk_text) > self.config.max_chunk_size:
                # Split large chunks further
                sub_chunks = self._split_large_chunk(chunk_text)
                chunks.extend(sub_chunks)
        
        return chunks
    
    def _split_large_chunk(self, text: str) -> List[str]:
        """Split chunks that exceed maximum size."""
        words = text.split()
        sub_chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            word_size = len(word) + 1  # +1 for space
            if current_size + word_size > self.config.max_chunk_size and current_chunk:
                sub_chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_size = word_size
            else:
                current_chunk.append(word)
                current_size += word_size
        
        if current_chunk:
            sub_chunks.append(' '.join(current_chunk))
        
        return sub_chunks
    
    def chunk_text(self, text: str) -> List[str]:
        """Perform semantic chunking on input text."""
        sentences = self._split_into_sentences(text)
        boundaries = self._find_semantic_boundaries(sentences)
        chunks = self._create_chunks_from_boundaries(sentences, boundaries)
        return chunks


def create_semantic_chunking() -> SemanticChunker:
    """Implement sentence-transformer-based semantic boundary detection for optimal chunk sizing.
    
    Returns:
        SemanticChunker: Configured semantic chunking system
    """
    print("[RAG] Initializing semantic chunking system")
    
    config = ChunkingConfig(
        similarity_threshold=0.7,
        min_chunk_size=100,
        max_chunk_size=1000,
        overlap_sentences=1,
        embedding_model="all-MiniLM-L6-v2"
    )
    
    chunker = SemanticChunker(config)
    
    # Test with sample job description
    test_text = """
    We are seeking a Senior Software Engineer to join our dynamic team. The ideal candidate will have 5+ years of experience in Python development and strong knowledge of machine learning frameworks. 
    
    Responsibilities include designing scalable systems, mentoring junior developers, and collaborating with cross-functional teams. You will work on cutting-edge AI applications and contribute to our data pipeline architecture.
    
    Requirements: Bachelor's degree in Computer Science, experience with AWS, Docker, and Kubernetes. Strong communication skills and ability to work in an agile environment are essential.
    """
    
    chunks = chunker.chunk_text(test_text)
    print(f"[RAG] Test chunking: {len(chunks)} chunks created")
    for i, chunk in enumerate(chunks):
        print(f"[RAG] Chunk {i+1}: {len(chunk)} chars")
    
    print("[SUCCESS] Semantic chunking system ready")
    return chunker


@dataclass
class CoarseRetrievalConfig:
    n_candidates: int = 100
    n_probe: int = 10
    metric_type: str = "L2"
    use_gpu: bool = False
    min_docs_for_ivf: int = 100


class CoarseRetriever:
    def __init__(self, config: CoarseRetrievalConfig, embedding_dim: int):
        self.config = config
        self.embedding_dim = embedding_dim
        self.document_store = {}
        self.metadata_store = {}
        self._create_index()
        print(f"[RAG] Coarse retriever initialized: {self.index.ntotal} vectors, {embedding_dim}D")
    
    def _create_index(self) -> None:
        """Create optimized FAISS index with adaptive clustering."""
        # Start with simple flat index - will upgrade to IVF when we have enough data
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        print(f"[RAG] Index created: {self.embedding_dim}D Flat index (will upgrade to IVF)")
        
        if self.config.use_gpu and faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)
            print("[RAG] Index moved to GPU")
    
    def _upgrade_to_ivf(self, embeddings: np.ndarray) -> None:
        """Upgrade to IVF index when we have enough data."""
        if isinstance(self.index, faiss.IndexFlatL2) and len(embeddings) >= self.config.min_docs_for_ivf:
            # Create IVF index with appropriate number of clusters
            n_clusters = min(100, len(embeddings) // 4)  # Conservative cluster count
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            new_index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, n_clusters)
            new_index.nprobe = min(self.config.n_probe, n_clusters)
            
            # Train new index
            new_index.train(embeddings.astype(np.float32))
            
            # Transfer existing vectors
            if self.index.ntotal > 0:
                old_vectors = self.index.reconstruct_n(0, self.index.ntotal)
                new_index.add(old_vectors)
            
            self.index = new_index
            print(f"[RAG] Upgraded to IVF index: {n_clusters} clusters, nprobe={new_index.nprobe}")
    
    def add_documents(self, embeddings: np.ndarray, documents: List[str], metadata: List[Dict]) -> None:
        """Add documents and embeddings to the index."""
        start_idx = self.index.ntotal
        
        # Normalize embeddings for cosine similarity if needed
        if self.config.metric_type == "IP":
            faiss.normalize_L2(embeddings)
        
        # Check if we should upgrade to IVF
        total_vectors = self.index.ntotal + len(embeddings)
        if total_vectors >= self.config.min_docs_for_ivf:
            # Collect all embeddings for training
            all_embeddings = embeddings
            if self.index.ntotal > 0:
                existing = self.index.reconstruct_n(0, self.index.ntotal)
                all_embeddings = np.vstack([existing, embeddings])
            self._upgrade_to_ivf(all_embeddings)
        
        # Train index if needed (IVF only)
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            self.index.train(embeddings.astype(np.float32))
            print(f"[RAG] Index trained with {len(embeddings)} vectors")
        
        self.index.add(embeddings.astype(np.float32))
        
        # Store documents and metadata
        for i, (doc, meta) in enumerate(zip(documents, metadata)):
            doc_id = start_idx + i
            self.document_store[doc_id] = doc
            self.metadata_store[doc_id] = meta
        
        print(f"[RAG] Added {len(embeddings)} documents, total: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, k: Optional[int] = None) -> Tuple[List[int], List[float], List[str]]:
        """Perform fast approximate nearest neighbor search."""
        k = min(k or self.config.n_candidates, self.index.ntotal)
        
        # Ensure query has correct dimensions and type
        query_embedding = query_embedding.astype(np.float32)
        
        # Normalize query if using cosine similarity
        if self.config.metric_type == "IP":
            query_embedding = query_embedding.copy()
            faiss.normalize_L2(query_embedding.reshape(1, -1))
        
        start_time = time.time()
        
        # Search FAISS index
        scores, indices = self.index.search(query_embedding.reshape(1, -1), k)
        
        # Filter out invalid indices (-1)
        valid_mask = indices[0] >= 0
        valid_indices = indices[0][valid_mask].tolist()
        valid_scores = scores[0][valid_mask].tolist()
        
        # Retrieve documents
        documents = [self.document_store.get(idx, "") for idx in valid_indices]
        
        elapsed = time.time() - start_time
        print(f"[RAG] Coarse retrieval: {len(valid_indices)} candidates in {elapsed:.3f}s")
        
        return valid_indices, valid_scores, documents
    
    def get_metadata(self, doc_ids: List[int]) -> List[Dict]:
        """Retrieve metadata for document IDs."""
        return [self.metadata_store.get(doc_id, {}) for doc_id in doc_ids]


def implement_coarse_retrieval(adaptive_embedder: AdaptiveEmbeddings) -> CoarseRetriever:
    """Fast first-stage retrieval using approximate nearest neighbors with 100+ candidates.
    
    Args:
        adaptive_embedder: Adaptive embedding system
        
    Returns:
        CoarseRetriever: Configured coarse retrieval system
    """
    print("[RAG] Initializing coarse retrieval system")
    
    # Get embedding dimension from test embedding
    test_embedding = adaptive_embedder.embed_texts(["test"])[0]
    embedding_dim = len(test_embedding)
    print(f"[RAG] Detected embedding dimension: {embedding_dim}")
    
    config = CoarseRetrievalConfig(
        n_candidates=100,
        n_probe=10,
        metric_type="L2",
        use_gpu=False,
        min_docs_for_ivf=100
    )
    
    retriever = CoarseRetriever(config, embedding_dim)
    
    # Test with sample job descriptions
    test_documents = [
        "Senior Python Developer with machine learning experience required for AI startup",
        "Frontend React developer needed for e-commerce platform development",
        "Data Scientist position open for predictive analytics and model deployment",
        "DevOps Engineer with Kubernetes and AWS experience for cloud infrastructure"
    ]
    
    test_metadata = [
        {"role": "senior", "domain": "ai", "skills": ["python", "ml"]},
        {"role": "mid", "domain": "frontend", "skills": ["react", "javascript"]},
        {"role": "senior", "domain": "data", "skills": ["python", "statistics"]},
        {"role": "senior", "domain": "devops", "skills": ["kubernetes", "aws"]}
    ]
    
    # Generate embeddings and add to index
    test_embeddings = adaptive_embedder.embed_texts(test_documents)
    retriever.add_documents(test_embeddings, test_documents, test_metadata)
    
    # Test retrieval
    query_text = "Python machine learning engineer position"
    query_embedding = adaptive_embedder.embed_texts([query_text])[0]
    
    indices, scores, documents = retriever.search(query_embedding, k=3)
    print(f"[RAG] Test query results: {len(documents)} matches")
    for i, (score, doc) in enumerate(zip(scores[:3], documents[:3])):
        print(f"[RAG] Match {i+1}: score={score:.3f}, doc='{doc[:50]}...'")
    
    print("[SUCCESS] Coarse retrieval system ready")
    return retriever


@dataclass
class FineGrainedConfig:
    similarity_metric: str = "cosine"  # cosine, euclidean, dot_product
    top_k_refinement: int = 20
    score_threshold: float = 0.1
    normalize_scores: bool = True


class FineGrainedFilter:
    def __init__(self, config: FineGrainedConfig, adaptive_embedder: AdaptiveEmbeddings):
        self.config = config
        self.embedder = adaptive_embedder
        self.similarity_func = self._get_similarity_function()
        print(f"[RAG] Fine-grained filter initialized: {config.similarity_metric} similarity")
    
    def _get_similarity_function(self) -> Callable:
        """Get similarity computation function based on metric."""
        if self.config.similarity_metric == "cosine":
            return self._cosine_similarity
        elif self.config.similarity_metric == "euclidean":
            return self._euclidean_similarity
        elif self.config.similarity_metric == "dot_product":
            return self._dot_product_similarity
        else:
            raise ValueError(f"Unsupported similarity metric: {self.config.similarity_metric}")
    
    def _cosine_similarity(self, query_emb: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and documents."""
        return cosine_similarity([query_emb], doc_embeddings)[0]
    
    def _euclidean_similarity(self, query_emb: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
        """Compute euclidean similarity (converted to similarity score)."""
        distances = euclidean_distances([query_emb], doc_embeddings)[0]
        # Convert distance to similarity: higher similarity = lower distance
        return 1.0 / (1.0 + distances)
    
    def _dot_product_similarity(self, query_emb: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
        """Compute dot product similarity."""
        return np.dot(doc_embeddings, query_emb)
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range."""
        if len(scores) == 0:
            return scores
        
        min_score, max_score = scores.min(), scores.max()
        if max_score - min_score == 0:
            return np.ones_like(scores)
        
        return (scores - min_score) / (max_score - min_score)
    
    def refine_candidates(self, 
                         query_text: str,
                         candidate_docs: List[str], 
                         candidate_indices: List[int],
                         candidate_scores: List[float]) -> Tuple[List[int], List[float], List[str]]:
        """Perform fine-grained filtering with exact similarity computation.
        
        Args:
            query_text: Original query text
            candidate_docs: Documents from coarse retrieval
            candidate_indices: Document indices from coarse retrieval
            candidate_scores: Scores from coarse retrieval
            
        Returns:
            Tuple of refined indices, scores, and documents
        """
        if not candidate_docs:
            return [], [], []
        
        print(f"[RAG] Fine-grained filtering: {len(candidate_docs)} candidates")
        start_time = time.time()
        
        # Generate embeddings for query and candidates
        all_texts = [query_text] + candidate_docs
        embeddings = self.embedder.embed_texts(all_texts)
        
        query_embedding = embeddings[0]
        doc_embeddings = embeddings[1:]
        
        # Compute exact similarities
        similarities = self.similarity_func(query_embedding, doc_embeddings)
        
        # Normalize scores if requested
        if self.config.normalize_scores:
            similarities = self._normalize_scores(similarities)
        
        # Filter by threshold
        valid_mask = similarities >= self.config.score_threshold
        
        # Create results
        refined_indices = [candidate_indices[i] for i in range(len(similarities)) if valid_mask[i]]
        refined_scores = similarities[valid_mask].tolist()
        refined_docs = [candidate_docs[i] for i in range(len(similarities)) if valid_mask[i]]
        
        # Sort by similarity (descending)
        sorted_data = sorted(zip(refined_indices, refined_scores, refined_docs), 
                           key=lambda x: x[1], reverse=True)
        
        # Take top-k
        top_k = min(self.config.top_k_refinement, len(sorted_data))
        final_indices = [x[0] for x in sorted_data[:top_k]]
        final_scores = [x[1] for x in sorted_data[:top_k]]
        final_docs = [x[2] for x in sorted_data[:top_k]]
        
        elapsed = time.time() - start_time
        print(f"[RAG] Fine-grained filtering completed: {len(final_docs)} refined in {elapsed:.3f}s")
        
        return final_indices, final_scores, final_docs
    
    def compute_relevance_scores(self, query_text: str, documents: List[str]) -> List[float]:
        """Compute relevance scores for a list of documents."""
        if not documents:
            return []
        
        # Generate embeddings
        all_texts = [query_text] + documents
        embeddings = self.embedder.embed_texts(all_texts)
        
        query_embedding = embeddings[0]
        doc_embeddings = embeddings[1:]
        
        # Compute similarities
        similarities = self.similarity_func(query_embedding, doc_embeddings)
        
        if self.config.normalize_scores:
            similarities = self._normalize_scores(similarities)
        
        return similarities.tolist()


def create_fine_grained_filtering(adaptive_embedder: AdaptiveEmbeddings) -> FineGrainedFilter:
    """Second-stage dense retrieval with exact similarity computation.
    
    Args:
        adaptive_embedder: Adaptive embedding system
        
    Returns:
        FineGrainedFilter: Configured fine-grained filtering system
    """
    print("[RAG] Initializing fine-grained filtering system")
    
    config = FineGrainedConfig(
        similarity_metric="cosine",
        top_k_refinement=20,
        score_threshold=0.1,
        normalize_scores=True
    )
    
    filter_system = FineGrainedFilter(config, adaptive_embedder)
    
    # Test with sample candidates from coarse retrieval
    test_query = "Python machine learning engineer position"
    test_candidates = [
        "Senior Python Developer with machine learning experience required for AI startup",
        "Data Scientist position open for predictive analytics and model deployment",
        "Frontend React developer needed for e-commerce platform development"
    ]
    test_indices = [0, 2, 1]
    test_scores = [0.656, 0.719, 1.250]
    
    # Perform fine-grained filtering
    refined_indices, refined_scores, refined_docs = filter_system.refine_candidates(
        test_query, test_candidates, test_indices, test_scores
    )
    
    print(f"[RAG] Test refinement: {len(refined_docs)} documents refined")
    for i, (idx, score, doc) in enumerate(zip(refined_indices, refined_scores, refined_docs)):
        print(f"[RAG] Refined {i+1}: idx={idx}, score={score:.3f}, doc='{doc[:40]}...'")
    
    print("[SUCCESS] Fine-grained filtering system ready")
    return filter_system


@dataclass
class MetadataFusionConfig:
    similarity_weight: float = 0.6
    role_weight: float = 0.2
    domain_weight: float = 0.15
    skills_weight: float = 0.05
    exact_match_boost: float = 0.2
    partial_match_boost: float = 0.1


class MetadataFusion:
    def __init__(self, config: MetadataFusionConfig):
        self.config = config
        self.role_hierarchy = {
            "junior": 1, "mid": 2, "senior": 3, "lead": 4, "principal": 5, "staff": 6
        }
        print(f"[RAG] Metadata fusion initialized: sim={config.similarity_weight}, role={config.role_weight}")
    
    def _extract_query_metadata(self, query_text: str) -> Dict:
        """Extract metadata features from query text."""
        query_lower = query_text.lower()
        
        # Extract role level
        role = "mid"  # default
        if any(term in query_lower for term in ["junior", "entry", "associate"]):
            role = "junior"
        elif any(term in query_lower for term in ["senior", "sr", "lead", "principal"]):
            role = "senior"
        
        # Extract domain
        domain = "general"
        if any(term in query_lower for term in ["ai", "ml", "machine learning", "data science"]):
            domain = "ai"
        elif any(term in query_lower for term in ["frontend", "react", "vue", "angular"]):
            domain = "frontend"
        elif any(term in query_lower for term in ["backend", "api", "server"]):
            domain = "backend"
        elif any(term in query_lower for term in ["devops", "kubernetes", "aws", "cloud"]):
            domain = "devops"
        elif any(term in query_lower for term in ["data", "analytics", "statistics"]):
            domain = "data"
        
        # Extract skills
        skills = []
        skill_patterns = {
            "python": r"\bpython\b",
            "javascript": r"\b(javascript|js)\b",
            "react": r"\breact\b",
            "ml": r"\b(machine learning|ml)\b",
            "aws": r"\baws\b",
            "kubernetes": r"\b(kubernetes|k8s)\b"
        }
        
        for skill, pattern in skill_patterns.items():
            if re.search(pattern, query_lower):
                skills.append(skill)
        
        return {
            "role": role,
            "domain": domain,
            "skills": skills
        }
    
    def _compute_role_similarity(self, query_role: str, doc_role: str) -> float:
        """Compute role-based similarity score."""
        if query_role == doc_role:
            return 1.0
        
        query_level = self.role_hierarchy.get(query_role, 2)
        doc_level = self.role_hierarchy.get(doc_role, 2)
        
        # Closer levels get higher similarity
        level_diff = abs(query_level - doc_level)
        if level_diff == 0:
            return 1.0
        elif level_diff == 1:
            return 0.7
        elif level_diff == 2:
            return 0.4
        else:
            return 0.1
    
    def _compute_domain_similarity(self, query_domain: str, doc_domain: str) -> float:
        """Compute domain-based similarity score."""
        if query_domain == doc_domain:
            return 1.0
        
        # Domain relationships
        related_domains = {
            "ai": ["data", "backend"],
            "data": ["ai", "backend"],
            "frontend": ["backend"],
            "backend": ["frontend", "ai", "data"],
            "devops": ["backend"]
        }
        
        if doc_domain in related_domains.get(query_domain, []):
            return 0.5
        
        return 0.1
    
    def _compute_skills_similarity(self, query_skills: List[str], doc_skills: List[str]) -> float:
        """Compute skills-based similarity score."""
        if not query_skills or not doc_skills:
            return 0.5
        
        query_set = set(query_skills)
        doc_set = set(doc_skills)
        
        intersection = query_set & doc_set
        union = query_set | doc_set
        
        if not union:
            return 0.5
        
        # Jaccard similarity
        jaccard = len(intersection) / len(union)
        return jaccard
    
    def fuse_scores(self, 
                   query_text: str,
                   similarity_scores: List[float],
                   document_metadata: List[Dict],
                   document_indices: List[int]) -> Tuple[List[int], List[float]]:
        """Combine vector similarity with metadata scoring.
        
        Args:
            query_text: Original query text
            similarity_scores: Vector similarity scores
            document_metadata: Metadata for each document
            document_indices: Document indices
            
        Returns:
            Tuple of reranked indices and fused scores
        """
        print(f"[RAG] Metadata fusion: processing {len(similarity_scores)} candidates")
        
        # Extract query metadata
        query_meta = self._extract_query_metadata(query_text)
        print(f"[RAG] Query metadata: {query_meta}")
        
        fused_scores = []
        
        for i, (sim_score, doc_meta, doc_idx) in enumerate(zip(similarity_scores, document_metadata, document_indices)):
            # Compute metadata similarities
            role_sim = self._compute_role_similarity(
                query_meta.get("role", "mid"), 
                doc_meta.get("role", "mid")
            )
            
            domain_sim = self._compute_domain_similarity(
                query_meta.get("domain", "general"), 
                doc_meta.get("domain", "general")
            )
            
            skills_sim = self._compute_skills_similarity(
                query_meta.get("skills", []), 
                doc_meta.get("skills", [])
            )
            
            # Apply boosts for exact matches
            boost = 0.0
            if query_meta.get("role") == doc_meta.get("role"):
                boost += self.config.exact_match_boost
            if query_meta.get("domain") == doc_meta.get("domain"):
                boost += self.config.exact_match_boost
            
            # Compute weighted fusion score
            fused_score = (
                self.config.similarity_weight * sim_score +
                self.config.role_weight * role_sim +
                self.config.domain_weight * domain_sim +
                self.config.skills_weight * skills_sim +
                boost
            )
            
            fused_scores.append((doc_idx, fused_score, sim_score, role_sim, domain_sim, skills_sim))
        
        # Sort by fused score (descending)
        fused_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Extract sorted indices and scores
        sorted_indices = [x[0] for x in fused_scores]
        sorted_scores = [x[1] for x in fused_scores]
        
        # Log top results
        for i, (idx, fused, sim, role, domain, skills) in enumerate(fused_scores[:3]):
            print(f"[RAG] Rank {i+1}: idx={idx}, fused={fused:.3f} (sim={sim:.3f}, role={role:.3f}, domain={domain:.3f}, skills={skills:.3f})")
        
        return sorted_indices, sorted_scores


def build_metadata_fusion(coarse_retriever: CoarseRetriever) -> MetadataFusion:
    """Combine vector similarity with job role, seniority, and domain metadata scoring.
    
    Args:
        coarse_retriever: Coarse retrieval system for metadata access
        
    Returns:
        MetadataFusion: Configured metadata fusion system
    """
    print("[RAG] Initializing metadata fusion system")
    
    config = MetadataFusionConfig(
        similarity_weight=0.6,
        role_weight=0.2,
        domain_weight=0.15,
        skills_weight=0.05,
        exact_match_boost=0.2,
        partial_match_boost=0.1
    )
    
    fusion_system = MetadataFusion(config)
    
    # Test with sample data
    test_query = "Senior Python machine learning engineer position"
    test_similarity_scores = [0.85, 0.72, 0.68]
    test_metadata = [
        {"role": "senior", "domain": "ai", "skills": ["python", "ml"]},
        {"role": "mid", "domain": "frontend", "skills": ["react", "javascript"]},
        {"role": "senior", "domain": "data", "skills": ["python", "statistics"]}
    ]
    test_indices = [0, 1, 2]
    
    # Perform metadata fusion
    fused_indices, fused_scores = fusion_system.fuse_scores(
        test_query, test_similarity_scores, test_metadata, test_indices
    )
    
    print(f"[RAG] Test fusion: {len(fused_indices)} documents reranked")
    print(f"[RAG] Original order: {test_indices}")
    print(f"[RAG] Fused order: {fused_indices}")
    print(f"[RAG] Score improvement: {max(fused_scores) - max(test_similarity_scores):.3f}")
    
    print("[SUCCESS] Metadata fusion system ready")
    return fusion_system


@dataclass
class RerankingConfig:
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    batch_size: int = 16
    max_length: int = 512
    device: str = "auto"
    score_threshold: float = 0.1
    top_k: int = 5


class CrossEncoderReranker:
    def __init__(self, config: RerankingConfig):
        self.config = config
        self.device = self._get_device()
        self.model = None
        self._load_model()
        print(f"[RAG] Cross-encoder reranker initialized: {config.model_name} on {self.device}")
    
    def _get_device(self) -> str:
        """Determine optimal device for inference."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return self.config.device
    
    def _load_model(self) -> None:
        """Load and configure cross-encoder model."""
        try:
            self.model = CrossEncoder(
                self.config.model_name,
                max_length=self.config.max_length,
                device=self.device
            )
            print(f"[RAG] Model loaded successfully: {self.config.model_name}")
        except Exception as e:
            print(f"[RAG] Warning: Failed to load cross-encoder, using fallback: {e}")
            # Fallback to a lighter model
            self.model = CrossEncoder(
                "cross-encoder/ms-marco-TinyBERT-L-2-v2",
                max_length=256,
                device=self.device
            )
            print("[RAG] Fallback model loaded: ms-marco-TinyBERT-L-2-v2")
    
    def rerank_passages(self, 
                       query: str, 
                       passages: List[str], 
                       passage_indices: List[int],
                       initial_scores: Optional[List[float]] = None) -> Tuple[List[int], List[float]]:
        """Rerank passages using cross-encoder relevance scoring.
        
        Args:
            query: Search query
            passages: List of candidate passages
            passage_indices: Original indices of passages
            initial_scores: Optional initial similarity scores
            
        Returns:
            Tuple of reranked indices and cross-encoder scores
        """
        if not passages:
            return [], []
        
        print(f"[RAG] Cross-encoder reranking: {len(passages)} passages")
        start_time = time.time()
        
        # Prepare query-passage pairs
        query_passage_pairs = [[query, passage] for passage in passages]
        
        # Compute cross-encoder scores in batches
        try:
            cross_scores = self.model.predict(
                query_passage_pairs,
                batch_size=self.config.batch_size,
                show_progress_bar=False
            )
        except Exception as e:
            print(f"[RAG] Cross-encoder prediction failed: {e}")
            # Fallback to initial scores if available
            if initial_scores:
                cross_scores = np.array(initial_scores)
            else:
                cross_scores = np.ones(len(passages))
        
        # Convert to numpy array if not already
        if not isinstance(cross_scores, np.ndarray):
            cross_scores = np.array(cross_scores)
        
        # Filter by threshold
        valid_mask = cross_scores >= self.config.score_threshold
        
        # Create results with valid scores
        results = []
        for i, (idx, score) in enumerate(zip(passage_indices, cross_scores)):
            if valid_mask[i]:
                results.append((idx, score, passages[i]))
        
        # Sort by cross-encoder score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Take top-k
        top_k = min(self.config.top_k, len(results))
        final_indices = [x[0] for x in results[:top_k]]
        final_scores = [float(x[1]) for x in results[:top_k]]
        
        elapsed = time.time() - start_time
        print(f"[RAG] Cross-encoder reranking completed: {len(final_indices)} results in {elapsed:.3f}s")
        
        # Log top results
        for i, (idx, score) in enumerate(zip(final_indices[:3], final_scores[:3])):
            print(f"[RAG] Rerank {i+1}: idx={idx}, score={score:.3f}")
        
        return final_indices, final_scores
    
    def score_pairs(self, query_passage_pairs: List[List[str]]) -> List[float]:
        """Score query-passage pairs directly."""
        if not query_passage_pairs:
            return []
        
        try:
            scores = self.model.predict(
                query_passage_pairs,
                batch_size=self.config.batch_size,
                show_progress_bar=False
            )
            return scores.tolist() if hasattr(scores, 'tolist') else list(scores)
        except Exception as e:
            print(f"[RAG] Scoring failed: {e}")
            return [0.5] * len(query_passage_pairs)


def setup_reranking_model() -> CrossEncoderReranker:
    """Configure cross-encoder (ms-marco-MiniLM) for passage-query relevance scoring.
    
    Returns:
        CrossEncoderReranker: Configured cross-encoder reranking system
    """
    print("[RAG] Initializing cross-encoder reranking system")
    
    config = RerankingConfig(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size=16,
        max_length=512,
        device="auto",
        score_threshold=0.1,
        top_k=5
    )
    
    reranker = CrossEncoderReranker(config)
    
    # Test with sample query and passages
    test_query = "Senior Python machine learning engineer position"
    test_passages = [
        "Senior Python Developer with machine learning experience required for AI startup",
        "Frontend React developer needed for e-commerce platform development",
        "Data Scientist position open for predictive analytics and model deployment"
    ]
    test_indices = [0, 1, 2]
    test_scores = [0.85, 0.72, 0.68]
    
    # Perform reranking
    reranked_indices, reranked_scores = reranker.rerank_passages(
        test_query, test_passages, test_indices, test_scores
    )
    
    print(f"[RAG] Test reranking: {len(reranked_indices)} passages reranked")
    print(f"[RAG] Original order: {test_indices}")
    print(f"[RAG] Reranked order: {reranked_indices}")
    
    # Test direct scoring
    test_pairs = [[test_query, passage] for passage in test_passages[:2]]
    pair_scores = reranker.score_pairs(test_pairs)
    print(f"[RAG] Direct scoring test: {len(pair_scores)} pairs scored")
    
    print("[SUCCESS] Cross-encoder reranking system ready")
    return reranker


@dataclass
class ListwiseRerankingConfig:
    max_candidates: int = 20
    final_top_k: int = 5
    temperature: float = 1.0
    score_aggregation: str = "weighted_sum"  # weighted_sum, max_pool, attention
    diversity_weight: float = 0.1
    relevance_weight: float = 0.9


class ListwiseReranker:
    def __init__(self, config: ListwiseRerankingConfig, cross_encoder: CrossEncoderReranker):
        self.config = config
        self.cross_encoder = cross_encoder
        self.device = cross_encoder.device
        print(f"[RAG] Listwise reranker initialized: max_candidates={config.max_candidates}, final_k={config.final_top_k}")
    
    def _compute_pairwise_similarities(self, passages: List[str]) -> np.ndarray:
        """Compute pairwise similarities between passages for diversity scoring."""
        n_passages = len(passages)
        similarity_matrix = np.zeros((n_passages, n_passages))
        
        # Use cross-encoder to compute passage-passage similarities
        pairs = []
        indices = []
        
        for i in range(n_passages):
            for j in range(i + 1, n_passages):
                pairs.append([passages[i], passages[j]])
                indices.append((i, j))
        
        if pairs:
            similarities = self.cross_encoder.score_pairs(pairs)
            
            for (i, j), sim in zip(indices, similarities):
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
        
        # Set diagonal to 1.0 (self-similarity)
        np.fill_diagonal(similarity_matrix, 1.0)
        
        return similarity_matrix
    
    def _compute_diversity_scores(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """Compute diversity scores based on pairwise similarities."""
        n_passages = similarity_matrix.shape[0]
        diversity_scores = np.zeros(n_passages)
        
        for i in range(n_passages):
            # Diversity is inversely related to maximum similarity with other passages
            max_sim_with_others = np.max(similarity_matrix[i, np.arange(n_passages) != i])
            diversity_scores[i] = 1.0 - max_sim_with_others
        
        return diversity_scores
    
    def _compute_attention_weights(self, relevance_scores: np.ndarray, diversity_scores: np.ndarray) -> np.ndarray:
        """Compute attention weights for listwise scoring."""
        # Combine relevance and diversity
        combined_scores = (
            self.config.relevance_weight * relevance_scores +
            self.config.diversity_weight * diversity_scores
        )
        
        # Apply temperature scaling
        scaled_scores = combined_scores / self.config.temperature
        
        # Softmax to get attention weights
        exp_scores = np.exp(scaled_scores - np.max(scaled_scores))
        attention_weights = exp_scores / np.sum(exp_scores)
        
        return attention_weights
    
    def _aggregate_scores(self, relevance_scores: np.ndarray, diversity_scores: np.ndarray, attention_weights: np.ndarray) -> np.ndarray:
        """Aggregate relevance and diversity scores using specified method."""
        if self.config.score_aggregation == "weighted_sum":
            final_scores = (
                self.config.relevance_weight * relevance_scores +
                self.config.diversity_weight * diversity_scores
            )
        
        elif self.config.score_aggregation == "max_pool":
            stacked = np.stack([relevance_scores, diversity_scores], axis=0)
            final_scores = np.max(stacked, axis=0)
        
        elif self.config.score_aggregation == "attention":
            # Use attention weights to combine scores
            final_scores = attention_weights * relevance_scores + (1 - attention_weights) * diversity_scores
        
        else:
            # Default to weighted sum
            final_scores = (
                self.config.relevance_weight * relevance_scores +
                self.config.diversity_weight * diversity_scores
            )
        
        return final_scores
    
    def rerank_listwise(self, 
                       query: str,
                       passages: List[str],
                       passage_indices: List[int],
                       initial_scores: List[float]) -> Tuple[List[int], List[float]]:
        """Rerank top-20 candidates using cross-attention for final top-5 selection.
        
        Args:
            query: Search query
            passages: List of candidate passages (up to max_candidates)
            passage_indices: Original indices of passages
            initial_scores: Initial relevance scores
            
        Returns:
            Tuple of final reranked indices and listwise scores
        """
        if not passages:
            return [], []
        
        # Limit to max candidates
        n_candidates = min(len(passages), self.config.max_candidates)
        passages = passages[:n_candidates]
        passage_indices = passage_indices[:n_candidates]
        initial_scores = initial_scores[:n_candidates]
        
        print(f"[RAG] Listwise reranking: {len(passages)} candidates -> top {self.config.final_top_k}")
        start_time = time.time()
        
        # Step 1: Get cross-encoder relevance scores
        query_passage_pairs = [[query, passage] for passage in passages]
        relevance_scores = np.array(self.cross_encoder.score_pairs(query_passage_pairs))
        
        # Normalize relevance scores
        if len(relevance_scores) > 1:
            relevance_scores = (relevance_scores - np.min(relevance_scores)) / (np.max(relevance_scores) - np.min(relevance_scores) + 1e-8)
        
        # Step 2: Compute diversity scores
        if len(passages) > 1:
            similarity_matrix = self._compute_pairwise_similarities(passages)
            diversity_scores = self._compute_diversity_scores(similarity_matrix)
        else:
            diversity_scores = np.array([1.0])
        
        # Normalize diversity scores
        if len(diversity_scores) > 1:
            diversity_scores = (diversity_scores - np.min(diversity_scores)) / (np.max(diversity_scores) - np.min(diversity_scores) + 1e-8)
        
        # Step 3: Compute attention weights
        attention_weights = self._compute_attention_weights(relevance_scores, diversity_scores)
        
        # Step 4: Aggregate final scores
        final_scores = self._aggregate_scores(relevance_scores, diversity_scores, attention_weights)
        
        # Step 5: Select top-k with diversity consideration
        selected_indices = []
        selected_scores = []
        available_indices = list(range(len(passages)))
        
        for _ in range(min(self.config.final_top_k, len(passages))):
            if not available_indices:
                break
            
            # Find best remaining candidate
            best_idx = max(available_indices, key=lambda i: final_scores[i])
            selected_indices.append(passage_indices[best_idx])
            selected_scores.append(float(final_scores[best_idx]))
            
            # Remove selected candidate
            available_indices.remove(best_idx)
            
            # Penalize similar candidates (diversity enforcement)
            if available_indices and len(passages) > 1:
                for remaining_idx in available_indices:
                    similarity = similarity_matrix[best_idx, remaining_idx] if 'similarity_matrix' in locals() else 0.0
                    penalty = similarity * self.config.diversity_weight
                    final_scores[remaining_idx] -= penalty
        
        elapsed = time.time() - start_time
        print(f"[RAG] Listwise reranking completed: {len(selected_indices)} results in {elapsed:.3f}s")
        
        # Log results
        for i, (idx, score) in enumerate(zip(selected_indices, selected_scores)):
            print(f"[RAG] Listwise {i+1}: idx={idx}, score={score:.3f}")
        
        return selected_indices, selected_scores


def implement_listwise_reranking(cross_encoder: CrossEncoderReranker) -> ListwiseReranker:
    """Rerank top-20 candidates using cross-attention for final top-5 selection.
    
    Args:
        cross_encoder: Configured cross-encoder reranker
        
    Returns:
        ListwiseReranker: Configured listwise reranking system
    """
    print("[RAG] Initializing listwise reranking system")
    
    config = ListwiseRerankingConfig(
        max_candidates=20,
        final_top_k=5,
        temperature=1.0,
        score_aggregation="weighted_sum",
        diversity_weight=0.1,
        relevance_weight=0.9
    )
    
    listwise_reranker = ListwiseReranker(config, cross_encoder)
    
    # Test with sample candidates
    test_query = "Senior Python machine learning engineer position"
    test_passages = [
        "Senior Python Developer with machine learning experience required for AI startup",
        "Python Data Scientist with ML expertise for tech company",
        "Machine Learning Engineer position using Python and TensorFlow",
        "Senior Data Engineer with Python background for analytics team",
        "Frontend React developer needed for e-commerce platform development",
        "DevOps Engineer with Kubernetes and AWS experience for cloud infrastructure"
    ]
    test_indices = list(range(len(test_passages)))
    test_scores = [0.9, 0.85, 0.88, 0.75, 0.3, 0.4]
    
    # Perform listwise reranking
    final_indices, final_scores = listwise_reranker.rerank_listwise(
        test_query, test_passages, test_indices, test_scores
    )
    
    print(f"[RAG] Test listwise reranking: {len(final_indices)} final selections")
    print(f"[RAG] Original candidates: {len(test_passages)}")
    print(f"[RAG] Final selection: {final_indices}")
    
    print("[SUCCESS] Listwise reranking system ready")
    return listwise_reranker


@dataclass
class BatchingConfig:
    batch_size: int = 8  # Smaller for CPU/MPS
    max_concurrent_batches: int = 2
    enable_threading: bool = True
    adaptive_batching: bool = True
    memory_threshold_mb: float = 512.0
    timeout_seconds: float = 30.0


class OptimizedBatchProcessor:
    def __init__(self, config: BatchingConfig, cross_encoder: CrossEncoderReranker):
        self.config = config
        self.cross_encoder = cross_encoder
        self.device = cross_encoder.device
        self._adaptive_batch_size = config.batch_size
        self._processing_times = []
        print(f"[RAG] Batch processor initialized: device={self.device}, batch_size={config.batch_size}")
    
    def _estimate_memory_usage(self, num_pairs: int, avg_text_length: int = 256) -> float:
        """Estimate memory usage in MB for batch processing."""
        # Rough estimation based on model and text length
        tokens_per_pair = avg_text_length * 2  # query + passage
        memory_per_token = 4  # bytes (rough estimate)
        total_memory_mb = (num_pairs * tokens_per_pair * memory_per_token) / (1024 * 1024)
        return total_memory_mb
    
    def _adapt_batch_size(self, num_pairs: int, avg_processing_time: float) -> int:
        """Dynamically adapt batch size based on performance and memory."""
        if not self.config.adaptive_batching:
            return self.config.batch_size
        
        # Estimate memory usage
        estimated_memory = self._estimate_memory_usage(self.config.batch_size)
        
        # Reduce batch size if memory usage is too high
        if estimated_memory > self.config.memory_threshold_mb:
            new_batch_size = max(1, int(self.config.batch_size * 0.7))
            print(f"[RAG] Reducing batch size due to memory: {self.config.batch_size} -> {new_batch_size}")
            return new_batch_size
        
        # Increase batch size if processing is fast and memory allows
        if len(self._processing_times) > 3:
            avg_time = np.mean(self._processing_times[-3:])
            if avg_time < 0.5 and estimated_memory < self.config.memory_threshold_mb * 0.5:
                new_batch_size = min(32, int(self.config.batch_size * 1.2))
                return new_batch_size
        
        return self.config.batch_size
    
    def _create_batches(self, query_passage_pairs: List[List[str]]) -> Iterator[List[List[str]]]:
        """Create optimized batches from query-passage pairs."""
        batch_size = self._adapt_batch_size(len(query_passage_pairs), 0.0)
        
        for i in range(0, len(query_passage_pairs), batch_size):
            yield query_passage_pairs[i:i + batch_size]
    
    def _process_single_batch(self, batch: List[List[str]], batch_id: int) -> Tuple[int, List[float]]:
        """Process a single batch and return results with batch ID."""
        start_time = time.time()
        
        try:
            scores = self.cross_encoder.score_pairs(batch)
            processing_time = time.time() - start_time
            
            # Track processing times for adaptive batching
            self._processing_times.append(processing_time)
            if len(self._processing_times) > 10:
                self._processing_times.pop(0)
            
            print(f"[RAG] Batch {batch_id}: {len(batch)} pairs processed in {processing_time:.3f}s")
            return batch_id, scores
            
        except Exception as e:
            print(f"[RAG] Batch {batch_id} failed: {e}")
            return batch_id, [0.0] * len(batch)
    
    def process_batches_sequential(self, query_passage_pairs: List[List[str]]) -> List[float]:
        """Process batches sequentially for memory-constrained environments."""
        print(f"[RAG] Sequential batch processing: {len(query_passage_pairs)} pairs")
        start_time = time.time()
        
        all_scores = []
        batches = list(self._create_batches(query_passage_pairs))
        
        for batch_id, batch in enumerate(batches):
            _, batch_scores = self._process_single_batch(batch, batch_id)
            all_scores.extend(batch_scores)
        
        elapsed = time.time() - start_time
        print(f"[RAG] Sequential processing completed: {len(all_scores)} scores in {elapsed:.3f}s")
        return all_scores
    
    def process_batches_concurrent(self, query_passage_pairs: List[List[str]]) -> List[float]:
        """Process batches concurrently with limited threading."""
        print(f"[RAG] Concurrent batch processing: {len(query_passage_pairs)} pairs")
        start_time = time.time()
        
        batches = list(self._create_batches(query_passage_pairs))
        all_scores = [None] * len(query_passage_pairs)
        
        with ThreadPoolExecutor(max_workers=self.config.max_concurrent_batches) as executor:
            # Submit all batch jobs
            future_to_batch = {
                executor.submit(self._process_single_batch, batch, batch_id): (batch_id, len(batch))
                for batch_id, batch in enumerate(batches)
            }
            
            # Collect results as they complete
            scores_offset = 0
            for future in as_completed(future_to_batch, timeout=self.config.timeout_seconds):
                batch_id, batch_scores = future.result()
                batch_size = future_to_batch[future][1]
                
                # Place scores in correct position
                start_idx = batch_id * self._adaptive_batch_size
                end_idx = start_idx + len(batch_scores)
                all_scores[start_idx:end_idx] = batch_scores
        
        # Filter out None values and flatten
        final_scores = [score for score in all_scores if score is not None]
        
        elapsed = time.time() - start_time
        print(f"[RAG] Concurrent processing completed: {len(final_scores)} scores in {elapsed:.3f}s")
        return final_scores
    
    def process_optimized(self, query_passage_pairs: List[List[str]]) -> List[float]:
        """Process with optimal strategy based on device and data size."""
        if not query_passage_pairs:
            return []
        
        # Choose processing strategy
        if (self.config.enable_threading and 
            len(query_passage_pairs) > self.config.batch_size * 2 and
            self.device in ["cpu", "mps"]):
            return self.process_batches_concurrent(query_passage_pairs)
        else:
            return self.process_batches_sequential(query_passage_pairs)


def optimize_reranking_batching(cross_encoder: CrossEncoderReranker) -> OptimizedBatchProcessor:
    """Batch processing for efficient CPU/MPS utilization in reranking pipeline.
    
    Args:
        cross_encoder: Configured cross-encoder reranker
        
    Returns:
        OptimizedBatchProcessor: Configured batch processing system
    """
    print("[RAG] Initializing optimized batch processing system")
    
    # Adapt configuration based on available device
    device = cross_encoder.device
    
    if device == "mps":
        # MPS-optimized settings
        config = BatchingConfig(
            batch_size=8,
            max_concurrent_batches=2,
            enable_threading=True,
            adaptive_batching=True,
            memory_threshold_mb=512.0,
            timeout_seconds=30.0
        )
        print("[RAG] MPS optimization: moderate batching with threading")
        
    elif device == "cpu":
        # CPU-optimized settings
        config = BatchingConfig(
            batch_size=4,
            max_concurrent_batches=3,
            enable_threading=True,
            adaptive_batching=True,
            memory_threshold_mb=256.0,
            timeout_seconds=45.0
        )
        print("[RAG] CPU optimization: smaller batches with more threads")
        
    else:
        # GPU settings (if available)
        config = BatchingConfig(
            batch_size=16,
            max_concurrent_batches=1,
            enable_threading=False,
            adaptive_batching=True,
            memory_threshold_mb=1024.0,
            timeout_seconds=20.0
        )
        print("[RAG] GPU optimization: larger batches, sequential processing")
    
    batch_processor = OptimizedBatchProcessor(config, cross_encoder)
    
    # Test with sample query-passage pairs
    test_query = "Senior Python machine learning engineer position"
    test_passages = [
        "Senior Python Developer with machine learning experience required for AI startup",
        "Python Data Scientist with ML expertise for tech company",
        "Machine Learning Engineer position using Python and TensorFlow",
        "Senior Data Engineer with Python background for analytics team",
        "Frontend React developer needed for e-commerce platform development",
        "DevOps Engineer with Kubernetes and AWS experience for cloud infrastructure",
        "Full-stack developer with Node.js and React experience",
        "Data Analyst position requiring SQL and Python skills"
    ]
    
    test_pairs = [[test_query, passage] for passage in test_passages]
    
    # Test batch processing
    batch_scores = batch_processor.process_optimized(test_pairs)
    
    print(f"[RAG] Test batch processing: {len(batch_scores)} scores computed")
    print(f"[RAG] Score range: {min(batch_scores):.3f} to {max(batch_scores):.3f}")
    print(f"[RAG] Final batch size: {batch_processor._adaptive_batch_size}")
    
    print("[SUCCESS] Optimized batch processing system ready")
    return batch_processor


# ================================
# Context Processing Components
# ================================

@dataclass
class CompressionConfig:
    max_context_length: int = 2000
    target_compression_ratio: float = 0.4
    min_chunk_overlap: int = 50
    model_name: str = "gpt-4.1-nano"
    temperature: float = 0.3
    max_tokens: int = 500
    cache_ttl_minutes: int = 30


class ContextCompressor:
    def __init__(self, config: CompressionConfig, openai_client: OpenAI):
        self.config = config
        self.client = openai_client
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        self.compression_cache = {}
        self.cache_timestamps = {}
        print(f"[RAG] Context compressor initialized: model={config.model_name}, ratio={config.target_compression_ratio}")
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.tokenizer.encode(text))
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached compression is still valid."""
        if cache_key not in self.cache_timestamps:
            return False
        
        elapsed_minutes = (time.time() - self.cache_timestamps[cache_key]) / 60
        return elapsed_minutes < self.config.cache_ttl_minutes
    
    def _get_cache_key(self, texts: List[str], query: str) -> str:
        """Generate cache key for compression request."""
        combined_text = query + "".join(texts[:3])  # Use first 3 texts for key
        return str(hash(combined_text))
    
    def _create_compression_prompt(self, query: str, contexts: List[str]) -> str:
        """Create prompt for LLM-based context compression."""
        target_tokens = int(self.config.max_context_length * self.config.target_compression_ratio)
        
        prompt = f"""
You are an expert at summarizing and compressing job-related content while preserving key information relevant to a specific query.

QUERY: {query}

TASK: Compress the following job descriptions and requirements into a concise summary of approximately {target_tokens} tokens that maximizes relevance to the query.

REQUIREMENTS:
1. Preserve all information directly relevant to the query
2. Keep specific technical skills, experience levels, and job requirements
3. Maintain company context if mentioned
4. Remove redundant or irrelevant details
5. Use clear, professional language
6. Structure as a coherent summary, not bullet points

CONTEXTS TO COMPRESS:
{chr(10).join([f"Context {{i+1}}: {{ctx}}" for i, ctx in enumerate(contexts)])}

COMPRESSED SUMMARY:"""
        
        return prompt
    
    def compress_contexts(self, query: str, contexts: List[str], context_scores: Optional[List[float]] = None) -> str:
        """Compress multiple contexts into a dense, relevant summary.
        
        Args:
            query: Original search query
            contexts: List of retrieved contexts to compress
            context_scores: Optional relevance scores for prioritization
            
        Returns:
            Compressed context string
        """
        if not contexts:
            return ""
        
        # Check cache first
        cache_key = self._get_cache_key(contexts, query)
        if self._is_cache_valid(cache_key):
            print(f"[RAG] Using cached compression: {cache_key[:8]}")
            return self.compression_cache[cache_key]
        
        print(f"[RAG] Compressing {len(contexts)} contexts for query relevance")
        start_time = time.time()
        
        # Calculate total tokens before compression
        total_tokens_before = sum(self._count_tokens(ctx) for ctx in contexts)
        
        # Sort contexts by relevance score if provided
        if context_scores:
            sorted_contexts = [ctx for _, ctx in sorted(zip(context_scores, contexts), reverse=True)]
        else:
            sorted_contexts = contexts
        
        # Limit contexts if too many tokens
        selected_contexts = []
        cumulative_tokens = 0
        
        for ctx in sorted_contexts:
            ctx_tokens = self._count_tokens(ctx)
            if cumulative_tokens + ctx_tokens <= self.config.max_context_length:
                selected_contexts.append(ctx)
                cumulative_tokens += ctx_tokens
            else:
                break
        
        if not selected_contexts:
            selected_contexts = [sorted_contexts[0][:1000]]  # Take first 1000 chars as fallback
        
        # Create compression prompt
        compression_prompt = self._create_compression_prompt(query, selected_contexts)
        
        try:
            # Call LLM for compression
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": compression_prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            compressed_context = response.choices[0].message.content.strip()
            
            # Calculate compression metrics
            tokens_after = self._count_tokens(compressed_context)
            compression_ratio = tokens_after / total_tokens_before if total_tokens_before > 0 else 0
            
            # Cache the result
            self.compression_cache[cache_key] = compressed_context
            self.cache_timestamps[cache_key] = time.time()
            
            elapsed = time.time() - start_time
            print(f"[RAG] Compression completed: {total_tokens_before} -> {tokens_after} tokens ({compression_ratio:.2f} ratio) in {elapsed:.3f}s")
            
            return compressed_context
            
        except Exception as e:
            print(f"[RAG] Compression failed, using truncated original: {e}")
            # Fallback: concatenate and truncate
            fallback_context = " ".join(selected_contexts)
            max_chars = int(self.config.max_context_length * self.config.target_compression_ratio * 4)  # ~4 chars per token
            return fallback_context[:max_chars] + "..." if len(fallback_context) > max_chars else fallback_context
    
    def compress_with_overlap(self, query: str, contexts: List[str], metadata: Optional[List[Dict]] = None) -> Tuple[str, Dict]:
        """Compress contexts while preserving overlapping information between chunks.
        
        Args:
            query: Search query
            contexts: Context chunks to compress
            metadata: Optional metadata for each context
            
        Returns:
            Tuple of compressed context and compression statistics
        """
        if len(contexts) <= 1:
            return self.compress_contexts(query, contexts), {"compression_ratio": 1.0, "chunks_processed": len(contexts)}
        
        # Group similar contexts to preserve overlap
        grouped_contexts = []
        current_group = [contexts[0]]
        
        for i in range(1, len(contexts)):
            # Simple overlap detection based on word similarity
            prev_words = set(contexts[i-1].lower().split())
            curr_words = set(contexts[i].lower().split())
            overlap_ratio = len(prev_words & curr_words) / len(prev_words | curr_words) if prev_words | curr_words else 0
            
            if overlap_ratio > 0.3:  # High overlap threshold
                current_group.append(contexts[i])
            else:
                grouped_contexts.append(current_group)
                current_group = [contexts[i]]
        
        if current_group:
            grouped_contexts.append(current_group)
        
        # Compress each group separately then combine
        compressed_groups = []
        total_chunks = 0
        
        for group in grouped_contexts:
            group_compressed = self.compress_contexts(query, group)
            compressed_groups.append(group_compressed)
            total_chunks += len(group)
        
        # Combine compressed groups
        final_compressed = " ".join(compressed_groups)
        
        # Calculate final metrics
        original_tokens = sum(self._count_tokens(ctx) for ctx in contexts)
        final_tokens = self._count_tokens(final_compressed)
        compression_ratio = final_tokens / original_tokens if original_tokens > 0 else 0
        
        stats = {
            "compression_ratio": compression_ratio,
            "chunks_processed": total_chunks,
            "groups_created": len(grouped_contexts),
            "original_tokens": original_tokens,
            "final_tokens": final_tokens
        }
        
        return final_compressed, stats


def create_context_compression(openai_client: OpenAI) -> ContextCompressor:
    """LLM-based context summarization to maximize relevant information density.
    
    Args:
        openai_client: Configured OpenAI client
        
    Returns:
        ContextCompressor: Configured context compression system
    """
    print("[RAG] Initializing context compression system")
    
    config = CompressionConfig(
        max_context_length=2000,
        target_compression_ratio=0.4,
        min_chunk_overlap=50,
        model_name="gpt-4.1-nano",
        temperature=0.3,
        max_tokens=500,
        cache_ttl_minutes=30
    )
    
    compressor = ContextCompressor(config, openai_client)
    
    # Test with sample contexts
    test_query = "Senior Python machine learning engineer position"
    test_contexts = [
        "We are seeking a Senior Python Developer with machine learning experience for our AI startup. The ideal candidate will have 5+ years of Python development experience and strong knowledge of ML frameworks like TensorFlow and PyTorch. Experience with cloud platforms (AWS, GCP) is preferred.",
        "Data Scientist position available for predictive analytics team. Requires PhD in Statistics or related field, expertise in Python, R, and machine learning algorithms. Experience with big data tools (Spark, Hadoop) and visualization tools (Tableau, PowerBI) essential.",
        "Senior Machine Learning Engineer role at tech company. Must have experience with deep learning, neural networks, and production ML systems. Python programming, Docker, Kubernetes, and MLOps experience required. Remote work available."
    ]
    test_scores = [0.95, 0.82, 0.91]
    
    # Test basic compression
    compressed_context = compressor.compress_contexts(test_query, test_contexts, test_scores)
    print(f"[RAG] Test compression: {len(' '.join(test_contexts))} -> {len(compressed_context)} chars")
    
    # Test compression with overlap preservation
    overlap_compressed, stats = compressor.compress_with_overlap(test_query, test_contexts)
    print(f"[RAG] Overlap compression: ratio={stats['compression_ratio']:.3f}, groups={stats['groups_created']}")
    
    print("[SUCCESS] Context compression system ready")
    return compressor


# ================================
# Relevance Filtering Components
# ================================

@dataclass
class RelevanceFilterConfig:
    static_threshold: float = 0.3
    dynamic_threshold_method: str = "adaptive"  # adaptive, percentile, std_dev
    min_results: int = 1
    max_results: int = 20
    percentile_cutoff: float = 0.6
    std_dev_multiplier: float = 0.5
    enable_semantic_clustering: bool = True
    diversity_bonus: float = 0.1


class RelevanceFilter:
    def __init__(self, config: RelevanceFilterConfig):
        self.config = config
        self.score_history = []
        self.adaptive_threshold = config.static_threshold
        print(f"[RAG] Relevance filter initialized: method={config.dynamic_threshold_method}, min_results={config.min_results}")
    
    def _compute_dynamic_threshold(self, scores: List[float]) -> float:
        """Compute dynamic threshold based on score distribution."""
        if len(scores) < 2:
            return self.config.static_threshold
        
        if self.config.dynamic_threshold_method == "percentile":
            return np.percentile(scores, self.config.percentile_cutoff * 100)
        
        elif self.config.dynamic_threshold_method == "std_dev":
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            return mean_score - (self.config.std_dev_multiplier * std_score)
        
        elif self.config.dynamic_threshold_method == "adaptive":
            # Adaptive threshold based on score distribution and history
            current_mean = np.mean(scores)
            current_std = np.std(scores)
            
            # Update adaptive threshold with exponential moving average
            alpha = 0.3  # Learning rate
            new_threshold = current_mean - (0.5 * current_std)
            self.adaptive_threshold = (alpha * new_threshold + 
                                     (1 - alpha) * self.adaptive_threshold)
            
            return self.adaptive_threshold
        
        else:
            return self.config.static_threshold
    
    def _compute_semantic_diversity_scores(self, documents: List[str], embeddings: Optional[np.ndarray] = None) -> List[float]:
        """Compute diversity scores based on semantic similarity."""
        if not self.config.enable_semantic_clustering or len(documents) <= 1:
            return [1.0] * len(documents)
        
        # Simple diversity scoring based on text overlap if no embeddings
        if embeddings is None:
            diversity_scores = []
            for i, doc in enumerate(documents):
                doc_words = set(doc.lower().split())
                max_overlap = 0.0
                
                for j, other_doc in enumerate(documents):
                    if i != j:
                        other_words = set(other_doc.lower().split())
                        if doc_words and other_words:
                            overlap = len(doc_words & other_words) / len(doc_words | other_words)
                            max_overlap = max(max_overlap, overlap)
                
                diversity_score = 1.0 - max_overlap
                diversity_scores.append(diversity_score)
            
            return diversity_scores
        
        # Use embeddings for more accurate diversity computation
        diversity_scores = []
        for i in range(len(embeddings)):
            similarities = []
            for j in range(len(embeddings)):
                if i != j:
                    sim = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )
                    similarities.append(sim)
            
            diversity_score = 1.0 - max(similarities) if similarities else 1.0
            diversity_scores.append(diversity_score)
        
        return diversity_scores
    
    def filter_by_relevance(self, 
                           documents: List[str],
                           scores: List[float],
                           indices: List[int],
                           metadata: Optional[List[Dict]] = None,
                           embeddings: Optional[np.ndarray] = None) -> Tuple[List[str], List[float], List[int], List[Dict]]:
        """Filter documents based on relevance thresholds and diversity.
        
        Args:
            documents: List of document texts
            scores: Relevance scores for documents
            indices: Original indices of documents
            metadata: Optional metadata for documents
            embeddings: Optional document embeddings for diversity computation
            
        Returns:
            Tuple of filtered documents, scores, indices, and metadata
        """
        if not documents or not scores:
            return [], [], [], []
        
        print(f"[RAG] Filtering {len(documents)} documents by relevance")
        
        # Ensure equal lengths
        min_length = min(len(documents), len(scores), len(indices))
        documents = documents[:min_length]
        scores = scores[:min_length]
        indices = indices[:min_length]
        if metadata:
            metadata = metadata[:min_length]
        else:
            metadata = [{}] * min_length
        
        # Store scores for adaptive threshold computation
        self.score_history.extend(scores)
        if len(self.score_history) > 1000:  # Keep recent history
            self.score_history = self.score_history[-1000:]
        
        # Compute dynamic threshold
        threshold = self._compute_dynamic_threshold(scores)
        print(f"[RAG] Using threshold: {threshold:.3f} (method: {self.config.dynamic_threshold_method})")
        
        # Apply relevance threshold
        relevant_items = []
        for doc, score, idx, meta in zip(documents, scores, indices, metadata):
            if score >= threshold:
                relevant_items.append((doc, score, idx, meta))
        
        # If too few results, lower threshold
        if len(relevant_items) < self.config.min_results and relevant_items:
            # Take top min_results based on scores
            all_items = list(zip(documents, scores, indices, metadata))
            all_items.sort(key=lambda x: x[1], reverse=True)
            relevant_items = all_items[:self.config.min_results]
            print(f"[RAG] Threshold lowered: keeping top {len(relevant_items)} results")
        
        if not relevant_items:
            return [], [], [], []
        
        # Compute diversity scores if enabled
        if self.config.enable_semantic_clustering and len(relevant_items) > 1:
            filtered_docs = [item[0] for item in relevant_items]
            filtered_embeddings = embeddings[:len(relevant_items)] if embeddings is not None else None
            diversity_scores = self._compute_semantic_diversity_scores(filtered_docs, filtered_embeddings)
            
            # Apply diversity bonus to final scores
            enhanced_items = []
            for i, (doc, score, idx, meta) in enumerate(relevant_items):
                diversity_bonus = diversity_scores[i] * self.config.diversity_bonus
                enhanced_score = score + diversity_bonus
                enhanced_items.append((doc, enhanced_score, idx, meta, diversity_scores[i]))
            
            # Sort by enhanced scores
            enhanced_items.sort(key=lambda x: x[1], reverse=True)
            relevant_items = [(item[0], item[1], item[2], item[3]) for item in enhanced_items]
            
            print(f"[RAG] Applied diversity bonus: avg_diversity={np.mean(diversity_scores):.3f}")
        
        # Limit to max results
        if len(relevant_items) > self.config.max_results:
            relevant_items = relevant_items[:self.config.max_results]
            print(f"[RAG] Limited to top {self.config.max_results} results")
        
        # Unpack results
        filtered_docs = [item[0] for item in relevant_items]
        filtered_scores = [item[1] for item in relevant_items]
        filtered_indices = [item[2] for item in relevant_items]
        filtered_metadata = [item[3] for item in relevant_items]
        
        print(f"[RAG] Filtering completed: {len(documents)} -> {len(filtered_docs)} documents")
        print(f"[RAG] Score range: {min(filtered_scores):.3f} to {max(filtered_scores):.3f}")
        
        return filtered_docs, filtered_scores, filtered_indices, filtered_metadata
    
    def get_threshold_stats(self) -> Dict:
        """Get statistics about threshold computation."""
        if not self.score_history:
            return {}
        
        return {
            "current_adaptive_threshold": self.adaptive_threshold,
            "score_history_size": len(self.score_history),
            "mean_historical_score": np.mean(self.score_history),
            "std_historical_score": np.std(self.score_history),
            "percentile_75": np.percentile(self.score_history, 75),
            "percentile_25": np.percentile(self.score_history, 25)
        }


def implement_relevance_filtering() -> RelevanceFilter:
    """Threshold-based filtering to eliminate low-relevance retrieved chunks.
    
    Returns:
        RelevanceFilter: Configured relevance filtering system
    """
    print("[RAG] Initializing relevance filtering system")
    
    config = RelevanceFilterConfig(
        static_threshold=0.3,
        dynamic_threshold_method="adaptive",
        min_results=1,
        max_results=20,
        percentile_cutoff=0.6,
        std_dev_multiplier=0.5,
        enable_semantic_clustering=True,
        diversity_bonus=0.1
    )
    
    relevance_filter = RelevanceFilter(config)
    
    # Test with sample documents and scores
    test_documents = [
        "Senior Python Developer with machine learning experience required for AI startup",
        "Python Data Scientist with ML expertise for tech company",
        "Frontend React developer needed for e-commerce platform development",
        "Machine Learning Engineer position using Python and TensorFlow",
        "DevOps Engineer with Kubernetes and AWS experience",
        "Junior developer position for mobile app development",
        "Data Analyst role requiring SQL and Python skills"
    ]
    
    test_scores = [0.95, 0.88, 0.25, 0.92, 0.15, 0.08, 0.65]
    test_indices = list(range(len(test_documents)))
    test_metadata = [{"source": f"job_{i}"} for i in range(len(test_documents))]
    
    # Test relevance filtering
    filtered_docs, filtered_scores, filtered_indices, filtered_meta = relevance_filter.filter_by_relevance(
        test_documents, test_scores, test_indices, test_metadata
    )
    
    print(f"[RAG] Test filtering: {len(test_documents)} -> {len(filtered_docs)} documents")
    print(f"[RAG] Filtered indices: {filtered_indices}")
    
    # Test with different score distributions
    low_scores = [0.1, 0.15, 0.2, 0.25, 0.3]
    filtered_low, _, _, _ = relevance_filter.filter_by_relevance(
        test_documents[:5], low_scores, list(range(5))
    )
    print(f"[RAG] Low score test: {len(low_scores)} -> {len(filtered_low)} documents")
    
    # Show threshold statistics
    stats = relevance_filter.get_threshold_stats()
    print(f"[RAG] Threshold stats: adaptive={stats.get('current_adaptive_threshold', 0):.3f}")
    
    print("[SUCCESS] Relevance filtering system ready")
    return relevance_filter


# ================================
# Context Fusion Components
# ================================

@dataclass
class ContextFusionConfig:
    overlap_threshold: float = 0.3
    max_merged_length: int = 3000
    sentence_overlap_weight: float = 0.7
    word_overlap_weight: float = 0.3
    preserve_order: bool = True
    remove_duplicates: bool = True
    merge_strategy: str = "hierarchical"  # hierarchical, sequential, clustered


class ContextFusion:
    def __init__(self, config: ContextFusionConfig):
        self.config = config
        print(f"[RAG] Context fusion initialized: strategy={config.merge_strategy}, overlap_threshold={config.overlap_threshold}")
    
    def _compute_text_overlap(self, text1: str, text2: str) -> float:
        """Compute overlap ratio between two text segments."""
        # Sentence-level overlap
        sentences1 = set(re.split(r'[.!?]+', text1.lower().strip()))
        sentences2 = set(re.split(r'[.!?]+', text2.lower().strip()))
        sentences1 = {s.strip() for s in sentences1 if len(s.strip()) > 5}
        sentences2 = {s.strip() for s in sentences2 if len(s.strip()) > 5}
        
        sentence_overlap = 0.0
        if sentences1 and sentences2:
            sentence_overlap = len(sentences1 & sentences2) / len(sentences1 | sentences2)
        
        # Word-level overlap
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        word_overlap = 0.0
        if words1 and words2:
            word_overlap = len(words1 & words2) / len(words1 | words2)
        
        # Weighted combination
        total_overlap = (self.config.sentence_overlap_weight * sentence_overlap + 
                        self.config.word_overlap_weight * word_overlap)
        
        return total_overlap
    
    def _find_best_merge_position(self, text1: str, text2: str) -> Tuple[int, int]:
        """Find optimal position to merge two texts with minimal overlap."""
        sentences1 = re.split(r'[.!?]+', text1)
        sentences2 = re.split(r'[.!?]+', text2)
        
        best_overlap = 0.0
        best_pos1, best_pos2 = len(text1), 0
        
        # Try different merge positions
        for i in range(max(0, len(sentences1) - 3), len(sentences1)):
            suffix1 = '.'.join(sentences1[i:]).strip()
            if not suffix1:
                continue
                
            for j in range(min(3, len(sentences2))):
                prefix2 = '.'.join(sentences2[:j+1]).strip()
                if not prefix2:
                    continue
                
                overlap = self._compute_text_overlap(suffix1, prefix2)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_pos1 = text1.rfind('.'.join(sentences1[:i]))
                    best_pos2 = text2.find('.'.join(sentences2[j+1:]))
        
        return max(0, best_pos1), max(0, best_pos2)
    
    def _merge_two_contexts(self, text1: str, text2: str, score1: float, score2: float) -> str:
        """Merge two contexts intelligently."""
        overlap_ratio = self._compute_text_overlap(text1, text2)
        
        if overlap_ratio < self.config.overlap_threshold:
            # Low overlap: simple concatenation
            return f"{text1.strip()} {text2.strip()}"
        
        # High overlap: intelligent merging
        pos1, pos2 = self._find_best_merge_position(text1, text2)
        
        # Choose higher scoring text as primary
        if score1 >= score2:
            primary_text = text1[:pos1] if pos1 > 0 else text1
            secondary_text = text2[pos2:] if pos2 < len(text2) else ""
        else:
            primary_text = text1[pos1:] if pos1 < len(text1) else ""
            secondary_text = text2[:pos2] if pos2 > 0 else text2
        
        merged = f"{primary_text.strip()} {secondary_text.strip()}".strip()
        return merged
    
    def _remove_redundant_sentences(self, text: str) -> str:
        """Remove duplicate or highly similar sentences."""
        if not self.config.remove_duplicates:
            return text
        
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 5]
        if len(sentences) <= 1:
            return text
        
        filtered_sentences = [sentences[0]]
        
        for sentence in sentences[1:]:
            is_duplicate = False
            for existing in filtered_sentences:
                similarity = SequenceMatcher(None, sentence.lower(), existing.lower()).ratio()
                if similarity > 0.8:  # High similarity threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_sentences.append(sentence)
        
        return '. '.join(filtered_sentences) + '.' if filtered_sentences else text
    
    def _hierarchical_fusion(self, contexts: List[str], scores: List[float]) -> str:
        """Hierarchical merging strategy - merge highest scoring pairs first."""
        if not contexts:
            return ""
        
        if len(contexts) == 1:
            return self._remove_redundant_sentences(contexts[0])
        
        # Create working lists
        working_contexts = contexts.copy()
        working_scores = scores.copy()
        
        while len(working_contexts) > 1:
            # Find best pair to merge (highest combined score)
            best_score = -1
            best_i, best_j = 0, 1
            
            for i in range(len(working_contexts)):
                for j in range(i + 1, len(working_contexts)):
                    combined_score = working_scores[i] + working_scores[j]
                    if combined_score > best_score:
                        best_score = combined_score
                        best_i, best_j = i, j
            
            # Merge the best pair
            merged_text = self._merge_two_contexts(
                working_contexts[best_i], 
                working_contexts[best_j],
                working_scores[best_i], 
                working_scores[best_j]
            )
            
            merged_score = (working_scores[best_i] + working_scores[best_j]) / 2
            
            # Remove original contexts and add merged one
            working_contexts = [ctx for k, ctx in enumerate(working_contexts) if k not in [best_i, best_j]]
            working_scores = [score for k, score in enumerate(working_scores) if k not in [best_i, best_j]]
            
            working_contexts.append(merged_text)
            working_scores.append(merged_score)
        
        final_text = working_contexts[0]
        return self._remove_redundant_sentences(final_text)
    
    def _sequential_fusion(self, contexts: List[str], scores: List[float]) -> str:
        """Sequential merging strategy - merge in order of relevance."""
        if not contexts:
            return ""
        
        # Sort by scores (descending)
        sorted_pairs = sorted(zip(contexts, scores), key=lambda x: x[1], reverse=True)
        sorted_contexts = [pair[0] for pair in sorted_pairs]
        sorted_scores = [pair[1] for pair in sorted_pairs]
        
        merged_text = sorted_contexts[0]
        
        for i in range(1, len(sorted_contexts)):
            merged_text = self._merge_two_contexts(
                merged_text, 
                sorted_contexts[i], 
                sorted_scores[0],  # Use highest score as primary
                sorted_scores[i]
            )
            
            # Check length limit
            if len(merged_text) > self.config.max_merged_length:
                merged_text = merged_text[:self.config.max_merged_length] + "..."
                break
        
        return self._remove_redundant_sentences(merged_text)
    
    def fuse_contexts(self, 
                     contexts: List[str], 
                     scores: List[float], 
                     metadata: Optional[List[Dict]] = None) -> Tuple[str, Dict]:
        """Intelligently merge multiple retrieved contexts with overlap detection.
        
        Args:
            contexts: List of context texts to merge
            scores: Relevance scores for each context
            metadata: Optional metadata for each context
            
        Returns:
            Tuple of merged context and fusion statistics
        """
        if not contexts or not scores:
            return "", {}
        
        print(f"[RAG] Fusing {len(contexts)} contexts using {self.config.merge_strategy} strategy")
        
        # Ensure equal lengths
        min_length = min(len(contexts), len(scores))
        contexts = contexts[:min_length]
        scores = scores[:min_length]
        
        # Calculate initial statistics
        total_length_before = sum(len(ctx) for ctx in contexts)
        avg_overlap = 0.0
        
        if len(contexts) > 1:
            overlaps = []
            for i in range(len(contexts)):
                for j in range(i + 1, len(contexts)):
                    overlap = self._compute_text_overlap(contexts[i], contexts[j])
                    overlaps.append(overlap)
            avg_overlap = np.mean(overlaps) if overlaps else 0.0
        
        # Apply fusion strategy
        if self.config.merge_strategy == "hierarchical":
            fused_text = self._hierarchical_fusion(contexts, scores)
        elif self.config.merge_strategy == "sequential":
            fused_text = self._sequential_fusion(contexts, scores)
        else:
            # Default to sequential
            fused_text = self._sequential_fusion(contexts, scores)
        
        # Calculate final statistics
        compression_ratio = len(fused_text) / total_length_before if total_length_before > 0 else 0
        
        fusion_stats = {
            "original_contexts": len(contexts),
            "total_length_before": total_length_before,
            "final_length": len(fused_text),
            "compression_ratio": compression_ratio,
            "average_overlap": avg_overlap,
            "merge_strategy": self.config.merge_strategy
        }
        
        print(f"[RAG] Context fusion completed: {len(contexts)} contexts -> {len(fused_text)} chars (compression: {compression_ratio:.3f})")
        print(f"[RAG] Average overlap detected: {avg_overlap:.3f}")
        
        return fused_text, fusion_stats


def build_context_fusion() -> ContextFusion:
    """Intelligent merging of multiple retrieved contexts with overlap detection.
    
    Returns:
        ContextFusion: Configured context fusion system
    """
    print("[RAG] Initializing context fusion system")
    
    config = ContextFusionConfig(
        overlap_threshold=0.3,
        max_merged_length=3000,
        sentence_overlap_weight=0.7,
        word_overlap_weight=0.3,
        preserve_order=True,
        remove_duplicates=True,
        merge_strategy="hierarchical"
    )
    
    context_fusion = ContextFusion(config)
    
    # Test with sample contexts that have overlapping content
    test_contexts = [
        "We are seeking a Senior Python Developer with machine learning experience. The ideal candidate will have 5+ years of Python development experience and strong knowledge of ML frameworks like TensorFlow and PyTorch.",
        "Senior Python Developer position available. Requires 5+ years of Python development experience and expertise in machine learning frameworks. Experience with TensorFlow, PyTorch, and scikit-learn preferred.",
        "Machine Learning Engineer role using Python and TensorFlow. The position requires strong programming skills in Python and experience with deep learning frameworks. Knowledge of MLOps and cloud platforms is a plus.",
        "Data Scientist position for predictive analytics team. Requires PhD in Statistics, expertise in Python and R, experience with machine learning algorithms and big data tools."
    ]
    
    test_scores = [0.95, 0.88, 0.82, 0.75]
    test_metadata = [{"source": f"job_{i}"} for i in range(len(test_contexts))]
    
    # Test context fusion
    fused_context, fusion_stats = context_fusion.fuse_contexts(
        test_contexts, test_scores, test_metadata
    )
    
    print(f"[RAG] Test fusion results:")
    print(f"[RAG] Original contexts: {fusion_stats['original_contexts']}")
    print(f"[RAG] Compression ratio: {fusion_stats['compression_ratio']:.3f}")
    print(f"[RAG] Average overlap: {fusion_stats['average_overlap']:.3f}")
    print(f"[RAG] Final text length: {len(fused_context)} characters")
    
    # Test with low overlap contexts
    diverse_contexts = [
        "Senior Python Developer with machine learning experience required",
        "Frontend React developer needed for e-commerce platform",
        "DevOps Engineer with Kubernetes and AWS experience"
    ]
    diverse_scores = [0.9, 0.3, 0.4]
    
    diverse_fused, diverse_stats = context_fusion.fuse_contexts(diverse_contexts, diverse_scores)
    print(f"[RAG] Diverse contexts test: overlap={diverse_stats['average_overlap']:.3f}, compression={diverse_stats['compression_ratio']:.3f}")
    
    print("[SUCCESS] Context fusion system ready")
    return context_fusion


# ================================
# Semantic Cache Components
# ================================

@dataclass
class CacheConfig:
    """Configuration for semantic caching system."""
    similarity_threshold: float = 0.95
    cache_ttl_seconds: int = 3600  # 1 hour
    max_cache_size: int = 1000
    embedding_model: str = "all-MiniLM-L6-v2"
    enable_caching: bool = True


class SemanticCache:
    """Semantic cache with embedding-based similarity detection."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache = {}
        self.timestamps = {}
        self.embedding_model = SentenceTransformer(config.embedding_model)
        self._lock = threading.Lock()
        self.embedding_cache = {}  # Add embedding cache
        print(f"[RAG] Semantic cache initialized: threshold={config.similarity_threshold}, ttl={config.cache_ttl_seconds}s")
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry has expired."""
        if key not in self.timestamps:
            return True
        elapsed = time.time() - self.timestamps[key]
        return elapsed > self.config.cache_ttl_seconds
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts."""
        embeddings = self.embedding_model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return similarity
    
    def get_cached_result(self, query: str) -> Optional[str]:
        """Retrieve cached result for semantically similar query."""
        if not self.config.enable_caching:
            return None
            
        with self._lock:
            # Check for exact match first
            if query in self.cache and not self._is_expired(query):
                print(f"[RAG] Exact cache hit for query")
                return self.cache[query]
            
            # Check for semantic similarity
            query_embedding = self.embedding_model.encode([query])
            
            for cached_query, cached_result in self.cache.items():
                if self._is_expired(cached_query):
                    continue
                
                # Compute similarity
                cached_embedding = self.embedding_model.encode([cached_query])
                similarity = cosine_similarity(query_embedding, cached_embedding)[0][0]
                
                if similarity >= self.config.similarity_threshold:
                    print(f"[RAG] Semantic cache hit: similarity={similarity:.3f}")
                    return cached_result
            
            return None
    
    def cache_result(self, query: str, result: str) -> None:
        """Cache result for query."""
        if not self.config.enable_caching:
            return
            
        with self._lock:
            # Remove expired entries
            expired_keys = [key for key in self.timestamps if self._is_expired(key)]
            for key in expired_keys:
                del self.cache[key]
                del self.timestamps[key]
            
            # Check cache size limit
            if len(self.cache) >= self.config.max_cache_size:
                # Remove oldest entry
                oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
                del self.cache[oldest_key]
                del self.timestamps[oldest_key]
            
            # Add new entry
            self.cache[query] = result
            self.timestamps[query] = time.time()
            print(f"[RAG] Result cached for query")
    
    # Add new methods for latency optimization
    def get_cached_embedding(self, query: str) -> Optional[np.ndarray]:
        """Retrieve cached embedding for query."""
        if not self.config.enable_caching:
            return None
            
        with self._lock:
            if query in self.embedding_cache and not self._is_expired(query):
                return self.embedding_cache[query]
            return None
    
    def cache_embedding(self, query: str, embedding: np.ndarray) -> None:
        """Cache embedding for query."""
        if not self.config.enable_caching:
            return
            
        with self._lock:
            # Remove expired entries
            expired_keys = [key for key in self.timestamps if self._is_expired(key)]
            for key in expired_keys:
                if key in self.embedding_cache:
                    del self.embedding_cache[key]
            
            # Check cache size limit
            if len(self.embedding_cache) >= self.config.max_cache_size:
                # Remove oldest entry
                oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
                if oldest_key in self.embedding_cache:
                    del self.embedding_cache[oldest_key]
            
            # Add new entry
            self.embedding_cache[query] = embedding
            self.timestamps[query] = time.time()
    
    def get_semantic_match(self, query: str, query_embedding: np.ndarray) -> Optional[Tuple[str, float]]:
        """Find semantically similar query in cache."""
        if not self.config.enable_caching:
            return None
            
        with self._lock:
            best_match = None
            best_similarity = 0.0
            
            for cached_query, cached_embedding in self.embedding_cache.items():
                if self._is_expired(cached_query):
                    continue
                
                # Compute similarity
                similarity = cosine_similarity([query_embedding], [cached_embedding])[0][0]
                
                if similarity >= self.config.similarity_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = (cached_query, similarity)
            
            return best_match


def implement_aggressive_caching() -> SemanticCache:
    """Deploy semantic cache with 0.95+ similarity threshold for aggressive deduplication.
    
    Returns:
        SemanticCache: Configured semantic caching system
    """
    print("[RAG] Initializing semantic caching system")
    
    config = CacheConfig(
        similarity_threshold=0.95,
        cache_ttl_seconds=3600,
        max_cache_size=1000,
        embedding_model="all-MiniLM-L6-v2",
        enable_caching=True
    )
    
    semantic_cache = SemanticCache(config)
    
    # Test with similar queries
    test_query1 = "What are the requirements for a senior Python developer position?"
    test_query2 = "What qualifications are needed for a senior Python developer role?"
    test_result = "Senior Python developers need 5+ years experience, ML knowledge, and cloud platform skills."
    
    # Cache result for first query
    semantic_cache.cache_result(test_query1, test_result)
    
    # Try to retrieve with similar query
    cached_result = semantic_cache.get_cached_result(test_query2)
    if cached_result:
        print(f"[RAG] Semantic cache test successful: retrieved result for similar query")
    else:
        print(f"[RAG] Semantic cache test: no match found (as expected for dissimilar queries)")
    
    print("[SUCCESS] Semantic caching system ready")
    return semantic_cache


# ================================
# Latency Optimization Components
# ================================

@dataclass
class LatencyOptimizationConfig:
    target_latency_ms: float = 100.0
    precompute_batch_size: int = 32
    parallel_retrieval: bool = True
    max_workers: int = 3
    enable_prefetching: bool = True
    cache_warmup_queries: int = 50
    index_optimization_interval: int = 1000


class LatencyOptimizer:
    def __init__(self, config: LatencyOptimizationConfig):
        self.config = config
        self.precomputed_embeddings = {}
        self.hot_queries = []
        self.query_frequencies = {}
        self.performance_metrics = []
        self.optimization_lock = threading.RLock()
        print(f"[RAG] Latency optimizer initialized: target={config.target_latency_ms}ms")
    
    def _measure_latency(self, func, *args, **kwargs) -> Tuple[Any, float]:
        """Measure function execution latency in milliseconds."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        latency_ms = (time.perf_counter() - start_time) * 1000
        return result, latency_ms
    
    def _optimize_index_for_speed(self, index: faiss.Index) -> faiss.Index:
        """Optimize FAISS index for speed over memory."""
        if hasattr(index, 'nprobe'):
            # Reduce nprobe for speed (trade-off with accuracy)
            original_nprobe = index.nprobe
            optimal_nprobe = min(5, original_nprobe)
            index.nprobe = optimal_nprobe
            print(f"[RAG] Index optimization: nprobe {original_nprobe} -> {optimal_nprobe}")
        
        return index
    
    def _ensure_embedding_compatibility(self, query_embedding: np.ndarray, coarse_retriever) -> np.ndarray:
        """Ensure query embedding is compatible with retriever index."""
        if hasattr(coarse_retriever, 'embedding_dim'):
            expected_dim = coarse_retriever.embedding_dim
        elif hasattr(coarse_retriever, 'index') and hasattr(coarse_retriever.index, 'd'):
            expected_dim = coarse_retriever.index.d
        else:
            # No dimension check needed
            return query_embedding
        
        current_dim = len(query_embedding)
        
        if current_dim == expected_dim:
            return query_embedding
        elif current_dim > expected_dim:
            # Truncate to match expected dimension
            print(f"[RAG] Truncating embedding: {current_dim} -> {expected_dim}")
            return query_embedding[:expected_dim]
        else:
            # Pad with zeros to match expected dimension
            print(f"[RAG] Padding embedding: {current_dim} -> {expected_dim}")
            padded = np.zeros(expected_dim, dtype=query_embedding.dtype)
            padded[:current_dim] = query_embedding
            return padded
    
    def _update_query_frequency(self, query: str) -> None:
        """Track query frequency for optimization."""
        with self.optimization_lock:
            self.query_frequencies[query] = self.query_frequencies.get(query, 0) + 1
            
            # Update hot queries list
            if len(self.query_frequencies) % 100 == 0:
                sorted_queries = sorted(
                    self.query_frequencies.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                self.hot_queries = [q for q, _ in sorted_queries[:self.config.cache_warmup_queries]]
    
    def optimize_retrieval_pipeline(self,
                                  query: str,
                                  adaptive_embedder,
                                  coarse_retriever,
                                  fine_grained_filter,
                                  semantic_cache) -> Tuple[List[str], List[float], float]:
        """Optimized retrieval pipeline with sub-100ms target latency.
        
        Args:
            query: Search query
            adaptive_embedder: Embedding system
            coarse_retriever: Coarse retrieval system
            fine_grained_filter: Fine-grained filtering system
            semantic_cache: Semantic caching system
            
        Returns:
            Tuple of documents, scores, and total latency in ms
        """
        pipeline_start = time.perf_counter()
        
        # Track query frequency
        self._update_query_frequency(query)
        
        # Step 1: Check semantic cache first
        cached_embedding = semantic_cache.get_cached_embedding(query)
        if cached_embedding is not None:
            query_embedding = cached_embedding
            embedding_latency = 0.0
            print(f"[RAG] Using cached embedding")
        else:
            # Check for semantic match
            temp_embedding = adaptive_embedder.embed_texts([query])[0]
            semantic_match = semantic_cache.get_semantic_match(query, temp_embedding)
            
            if semantic_match is not None:
                matched_query, similarity = semantic_match
                query_embedding = semantic_cache.get_cached_embedding(matched_query)
                embedding_latency = 0.0
                print(f"[RAG] Using semantic match: {similarity:.3f}")
            else:
                # Generate new embedding
                query_embedding, embedding_latency = self._measure_latency(
                    lambda: adaptive_embedder.embed_texts([query])[0]
                )
                semantic_cache.cache_embedding(query, query_embedding)
        
        # Step 2: Ensure embedding compatibility and perform coarse retrieval
        compatible_embedding = self._ensure_embedding_compatibility(query_embedding, coarse_retriever)
        
        try:
            (candidate_indices, candidate_scores, candidate_docs), coarse_latency = self._measure_latency(
                coarse_retriever.search, compatible_embedding, 20
            )
        except Exception as e:
            print(f"[RAG] Coarse retrieval failed: {e}")
            total_latency = (time.perf_counter() - pipeline_start) * 1000
            return [], [], total_latency
        
        if not candidate_docs:
            total_latency = (time.perf_counter() - pipeline_start) * 1000
            return [], [], total_latency
        
        # Step 3: Fine-grained filtering (if latency budget allows)
        elapsed_so_far = embedding_latency + coarse_latency
        remaining_budget = self.config.target_latency_ms - elapsed_so_far
        
        if remaining_budget > 20:  # Minimum 20ms for filtering
            try:
                (refined_indices, refined_scores, refined_docs), filter_latency = self._measure_latency(
                    fine_grained_filter.refine_candidates,
                    query, candidate_docs, candidate_indices, candidate_scores
                )
            except Exception as e:
                print(f"[RAG] Fine-grained filtering failed: {e}")
                refined_docs = candidate_docs[:5]
                refined_scores = candidate_scores[:5]
                filter_latency = 0.0
        else:
            # Skip fine-grained filtering to meet latency target
            refined_docs = candidate_docs[:5]
            refined_scores = candidate_scores[:5]
            filter_latency = 0.0
            print(f"[RAG] Skipped fine-grained filtering: budget={remaining_budget:.1f}ms")
        
        # Calculate total latency
        total_latency = (time.perf_counter() - pipeline_start) * 1000
        
        # Record performance metrics
        self.performance_metrics.append({
            "total_latency": total_latency,
            "embedding_latency": embedding_latency,
            "coarse_latency": coarse_latency,
            "filter_latency": filter_latency,
            "num_results": len(refined_docs),
            "cache_hit": cached_embedding is not None
        })
        
        # Keep recent metrics only
        if len(self.performance_metrics) > 1000:
            self.performance_metrics = self.performance_metrics[-1000:]
        
        print(f"[RAG] Pipeline latency: {total_latency:.1f}ms (target: {self.config.target_latency_ms}ms)")
        
        return refined_docs, refined_scores, total_latency
    
    def warmup_cache(self, warmup_queries: List[str], adaptive_embedder, semantic_cache) -> None:
        """Warm up cache with frequently accessed queries."""
        print(f"[RAG] Warming up cache with {len(warmup_queries)} queries")
        
        for query in warmup_queries:
            cached_embedding = semantic_cache.get_cached_embedding(query)
            if cached_embedding is None:  # Explicit None check instead of boolean evaluation
                embedding = adaptive_embedder.embed_texts([query])[0]
                semantic_cache.cache_embedding(query, embedding)
        
        print("[RAG] Cache warmup completed")
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self.performance_metrics:
            return {}
        
        recent_metrics = self.performance_metrics[-100:]  # Last 100 queries
        
        latencies = [m["total_latency"] for m in recent_metrics]
        cache_hits = [m["cache_hit"] for m in recent_metrics]
        
        return {
            "avg_latency_ms": np.mean(latencies),
            "p95_latency_ms": np.percentile(latencies, 95),
            "p99_latency_ms": np.percentile(latencies, 99),
            "min_latency_ms": np.min(latencies),
            "max_latency_ms": np.max(latencies),
            "cache_hit_rate": np.mean(cache_hits) * 100,
            "under_target_rate": np.mean([l <= self.config.target_latency_ms for l in latencies]) * 100,
            "total_queries": len(self.performance_metrics)
        }


def optimize_retrieval_latency(adaptive_embedder,
                             coarse_retriever,
                             fine_grained_filter,
                             semantic_cache) -> LatencyOptimizer:
    """Sub-100ms retrieval pipeline with precomputed embeddings and indexing.
    
    Args:
        adaptive_embedder: Embedding system
        coarse_retriever: Coarse retrieval system
        fine_grained_filter: Fine-grained filtering system
        semantic_cache: Semantic caching system
        
    Returns:
        LatencyOptimizer: Configured latency optimization system
    """
    print("[RAG] Initializing retrieval latency optimization")
    
    config = LatencyOptimizationConfig(
        target_latency_ms=100.0,
        precompute_batch_size=32,
        parallel_retrieval=True,
        max_workers=3,
        enable_prefetching=True,
        cache_warmup_queries=50,
        index_optimization_interval=1000
    )
    
    latency_optimizer = LatencyOptimizer(config)
    
    # Optimize index for speed
    if hasattr(coarse_retriever, 'index'):
        coarse_retriever.index = latency_optimizer._optimize_index_for_speed(coarse_retriever.index)
    
    # Warm up cache with common queries
    warmup_queries = [
        "Senior Python developer position",
        "Machine learning engineer job",
        "Data scientist role",
        "Frontend React developer"
    ]
    
    latency_optimizer.warmup_cache(warmup_queries, adaptive_embedder, semantic_cache)
    
    # Test latency optimization
    test_queries = [
        "Senior Python machine learning engineer position",
        "Data scientist with Python experience"
    ]
    
    print("[RAG] Testing latency optimization...")
    
    for query in test_queries:
        docs, scores, latency = latency_optimizer.optimize_retrieval_pipeline(
            query, adaptive_embedder, coarse_retriever, fine_grained_filter, semantic_cache
        )
        print(f"[RAG] Query latency: {latency:.1f}ms, results: {len(docs)}")
    
    # Performance summary
    stats = latency_optimizer.get_performance_stats()
    if stats:
        print(f"[RAG] Performance summary:")
        print(f"[RAG] Average latency: {stats.get('avg_latency_ms', 0):.1f}ms")
        print(f"[RAG] Cache hit rate: {stats.get('cache_hit_rate', 0):.1f}%")
    
    print("[SUCCESS] Retrieval latency optimization ready")
    return latency_optimizer


# ================================
# Fallback Strategies Components
# ================================

class FallbackMode(Enum):
    KEYWORD_SEARCH = "keyword"
    CACHED_RESULTS = "cached"
    SIMPLIFIED_RETRIEVAL = "simplified"
    EMERGENCY_RESPONSE = "emergency"


@dataclass
class FallbackConfig:
    max_wait_time_ms: float = 500.0
    enable_keyword_fallback: bool = True
    enable_cached_fallback: bool = True
    enable_simplified_fallback: bool = True
    emergency_response_enabled: bool = True
    keyword_similarity_threshold: float = 0.3
    cache_fallback_ttl_hours: int = 24
    health_check_interval_seconds: int = 30


class FallbackStrategies:
    def __init__(self, config: FallbackConfig):
        self.config = config
        self.system_health = {
            "vector_store": True,
            "embedding_service": True,
            "reranking_service": True
        }
        self.fallback_cache = {}
        self.keyword_index = {}
        self.emergency_responses = self._create_emergency_responses()
        self.health_lock = threading.RLock()
        print(f"[RAG] Fallback strategies initialized: max_wait={config.max_wait_time_ms}ms")
    
    def _create_emergency_responses(self) -> Dict[str, List[str]]:
        """Create emergency response templates for common query types."""
        return {
            "python": [
                "Python is a versatile programming language widely used in data science and machine learning.",
                "Senior Python developers typically need 3-5 years of experience with frameworks like Django or Flask.",
                "Python ML engineers often work with libraries like scikit-learn, TensorFlow, and PyTorch."
            ],
            "machine_learning": [
                "Machine learning engineers design and implement ML models for production systems.",
                "Key skills include Python, statistics, and experience with ML frameworks.",
                "Roles typically require understanding of both software engineering and data science."
            ],
            "data_scientist": [
                "Data scientists analyze data to extract insights and build predictive models.",
                "Common requirements include statistics, Python/R, and domain expertise.",
                "Experience with SQL, visualization tools, and cloud platforms is valuable."
            ],
            "default": [
                "This appears to be a technical role requiring specialized skills.",
                "Please check back later for more detailed information.",
                "Consider reviewing similar positions for relevant requirements."
            ]
        }
    
    def _simple_keyword_match(self, query: str, documents: List[str]) -> List[Tuple[str, float]]:
        """Simple keyword-based matching as fallback."""
        query_words = set(query.lower().split())
        results = []
        
        for doc in documents:
            doc_words = set(doc.lower().split())
            overlap = len(query_words & doc_words)
            total_words = len(query_words | doc_words)
            
            if total_words > 0:
                similarity = overlap / total_words
                if similarity >= self.config.keyword_similarity_threshold:
                    results.append((doc, similarity))
        
        # Sort by similarity score
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:5]  # Return top 5
    
    def _get_cached_results(self, query: str) -> Optional[List[str]]:
        """Retrieve cached results for similar queries."""
        if not self.config.enable_cached_fallback:
            return None
        
        # Simple cache lookup (in production, use more sophisticated similarity)
        query_normalized = query.lower().strip()
        
        if query_normalized in self.fallback_cache:
            cached_data, timestamp = self.fallback_cache[query_normalized]
            # Check if cache is still valid
            if time.time() - timestamp < (self.config.cache_fallback_ttl_hours * 3600):
                print("[RAG] Using cached fallback results")
                return cached_data
            else:
                # Remove expired cache
                del self.fallback_cache[query_normalized]
        
        return None
    
    def _cache_results(self, query: str, results: List[str]) -> None:
        """Cache results for future fallback use."""
        query_normalized = query.lower().strip()
        self.fallback_cache[query_normalized] = (results, time.time())
        
        # Limit cache size
        if len(self.fallback_cache) > 1000:
            # Remove oldest entries
            oldest_entries = sorted(
                self.fallback_cache.items(),
                key=lambda x: x[1][1]
            )[:100]
            for key, _ in oldest_entries:
                del self.fallback_cache[key]
    
    def _get_emergency_response(self, query: str) -> List[str]:
        """Generate emergency response based on query keywords."""
        query_lower = query.lower()
        
        # Check for keyword matches
        for keyword, responses in self.emergency_responses.items():
            if keyword in query_lower:
                return responses
        
        # Default response
        return self.emergency_responses["default"]
    
    def _check_system_health(self, system_name: str, check_func: Callable) -> bool:
        """Check health of a system component."""
        try:
            start_time = time.time()
            check_func()
            elapsed_ms = (time.time() - start_time) * 1000
            
            # Update health status
            with self.health_lock:
                is_healthy = elapsed_ms < self.config.max_wait_time_ms
                self.system_health[system_name] = is_healthy
                
            return is_healthy
            
        except Exception as e:
            print(f"[RAG] Health check failed for {system_name}: {e}")
            with self.health_lock:
                self.system_health[system_name] = False
            return False
    
    def execute_with_fallback(self,
                             query: str,
                             primary_retrieval_func: Callable,
                             available_documents: List[str] = None) -> Tuple[List[str], FallbackMode, float]:
        """Execute retrieval with automatic fallback strategies.
        
        Args:
            query: Search query
            primary_retrieval_func: Primary retrieval function to try first
            available_documents: Documents available for keyword fallback
            
        Returns:
            Tuple of results, fallback mode used, and execution time
        """
        start_time = time.time()
        
        # Try primary retrieval first
        try:
            print("[RAG] Attempting primary retrieval")
            
            # Set timeout for primary retrieval
            result = self._execute_with_timeout(
                primary_retrieval_func, 
                self.config.max_wait_time_ms / 1000
            )
            
            if result and len(result) > 0:
                elapsed_time = (time.time() - start_time) * 1000
                print(f"[RAG] Primary retrieval successful: {elapsed_time:.1f}ms")
                self._cache_results(query, result)
                return result, FallbackMode.CACHED_RESULTS, elapsed_time
                
        except Exception as e:
            print(f"[RAG] Primary retrieval failed: {e}")
        
        # Fallback 1: Check cached results
        if self.config.enable_cached_fallback:
            cached_results = self._get_cached_results(query)
            if cached_results:
                elapsed_time = (time.time() - start_time) * 1000
                print(f"[RAG] Using cached fallback: {elapsed_time:.1f}ms")
                return cached_results, FallbackMode.CACHED_RESULTS, elapsed_time
        
        # Fallback 2: Keyword-based search
        if self.config.enable_keyword_fallback and available_documents:
            print("[RAG] Attempting keyword-based fallback")
            keyword_results = self._simple_keyword_match(query, available_documents)
            
            if keyword_results:
                results = [doc for doc, _ in keyword_results]
                elapsed_time = (time.time() - start_time) * 1000
                print(f"[RAG] Keyword fallback successful: {len(results)} results, {elapsed_time:.1f}ms")
                self._cache_results(query, results)
                return results, FallbackMode.KEYWORD_SEARCH, elapsed_time
        
        # Fallback 3: Emergency response
        if self.config.emergency_response_enabled:
            print("[RAG] Using emergency response fallback")
            emergency_results = self._get_emergency_response(query)
            elapsed_time = (time.time() - start_time) * 1000
            return emergency_results, FallbackMode.EMERGENCY_RESPONSE, elapsed_time
        
        # No fallback available
        elapsed_time = (time.time() - start_time) * 1000
        return [], FallbackMode.EMERGENCY_RESPONSE, elapsed_time
    
    def _execute_with_timeout(self, func: Callable, timeout_seconds: float) -> Any:
        """Execute function with timeout."""
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(func)
            try:
                return future.result(timeout=timeout_seconds)
            except concurrent.futures.TimeoutError:
                print(f"[RAG] Function timed out after {timeout_seconds}s")
                return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system health status."""
        with self.health_lock:
            return {
                "system_health": self.system_health.copy(),
                "cache_size": len(self.fallback_cache),
                "emergency_responses_available": len(self.emergency_responses),
                "all_systems_healthy": all(self.system_health.values())
            }
    
    def simulate_system_failure(self, system_name: str, duration_seconds: float = 5.0) -> None:
        """Simulate system failure for testing (development only)."""
        print(f"[RAG] Simulating {system_name} failure for {duration_seconds}s")
        
        def restore_system():
            time.sleep(duration_seconds)
            with self.health_lock:
                self.system_health[system_name] = True
            print(f"[RAG] {system_name} restored")
        
        with self.health_lock:
            self.system_health[system_name] = False
        
        # Start restoration timer
        threading.Thread(target=restore_system, daemon=True).start()


def create_fallback_strategies() -> FallbackStrategies:
    """Graceful degradation when vector store unavailable or slow.
    
    Returns:
        FallbackStrategies: Configured fallback strategy system
    """
    print("[RAG] Initializing fallback strategies system")
    
    config = FallbackConfig(
        max_wait_time_ms=500.0,
        enable_keyword_fallback=True,
        enable_cached_fallback=True,
        enable_simplified_fallback=True,
        emergency_response_enabled=True,
        keyword_similarity_threshold=0.3,
        cache_fallback_ttl_hours=24,
        health_check_interval_seconds=30
    )
    
    fallback_system = FallbackStrategies(config)
    
    # Test fallback strategies
    test_documents = [
        "Senior Python Developer with machine learning experience required for AI startup",
        "Frontend React developer needed for e-commerce platform development",
        "Data Scientist position open for predictive analytics and model deployment",
        "DevOps Engineer with Kubernetes and AWS experience for cloud infrastructure"
    ]
    
    test_query = "Python machine learning engineer position"
    
    # Test normal operation
    def mock_primary_retrieval():
        return ["Senior Python Developer with machine learning experience required for AI startup"]
    
    results, mode, elapsed = fallback_system.execute_with_fallback(
        test_query, mock_primary_retrieval, test_documents
    )
    print(f"[RAG] Test 1 - Normal operation: {len(results)} results via {mode.value}, {elapsed:.1f}ms")
    
    # Test with simulated failure
    def mock_failing_retrieval():
        raise Exception("Vector store unavailable")
    
    results, mode, elapsed = fallback_system.execute_with_fallback(
        test_query, mock_failing_retrieval, test_documents
    )
    print(f"[RAG] Test 2 - With failure: {len(results)} results via {mode.value}, {elapsed:.1f}ms")
    
    # Test cached fallback (should use cache from first test)
    results, mode, elapsed = fallback_system.execute_with_fallback(
        test_query, mock_failing_retrieval, test_documents
    )
    print(f"[RAG] Test 3 - Cached fallback: {len(results)} results via {mode.value}, {elapsed:.1f}ms")
    
    # Test emergency response for unknown query
    unknown_query = "Completely unknown technical position"
    results, mode, elapsed = fallback_system.execute_with_fallback(
        unknown_query, mock_failing_retrieval, []
    )
    print(f"[RAG] Test 4 - Emergency response: {len(results)} results via {mode.value}, {elapsed:.1f}ms")
    
    # Show system status
    status = fallback_system.get_system_status()
    print(f"[RAG] System status: {status['cache_size']} cached entries, healthy: {status['all_systems_healthy']}")
    
    print("[SUCCESS] Fallback strategies system ready")
    return fallback_system