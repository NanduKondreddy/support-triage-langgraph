"""
Phase 2 - Hybrid Retrieval (Production-Grade)

Pipeline:
1. BM25 (lexical) → top 10 results
2. Embedding similarity (semantic) → rerank top 10
3. Hybrid score = 0.5 * norm(bm25) + 0.5 * cosine_sim
4. Return top 5 results

Optimization:
- Precompute all embeddings at initialization
- Single model load (singleton pattern)
- Runtime < 100ms per query
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import os
import re
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("sentence-transformers required. pip install sentence-transformers")

from corpus_loader import Document

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_+'-]*", re.I)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "for", "from", "have", "how",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "our", "please", "that", "the",
    "this", "to", "with", "you", "your",
}

# Global singleton model
_EMBEDDING_MODEL = None

def _get_embedding_model():
    """Load embedding model once globally."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDING_MODEL


@dataclass(frozen=True)
class Hit:
    """Search result with metadata."""
    doc: Document
    score: float  # Hybrid score (0.5 * bm25_norm + 0.5 * semantic)
    bm25_score: float  # Raw BM25 score
    semantic_score: float  # Cosine similarity
    snippet: str


def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25."""
    return [tok.lower().strip("-'") for tok in TOKEN_RE.findall(text) if tok.lower() not in STOPWORDS]


class HybridRetriever:
    """Production-grade hybrid retriever: BM25 + semantic."""

    def __init__(self, docs: list[Document]) -> None:
        self.docs = docs
        self.model = _get_embedding_model()
        
        # BM25 setup
        self.doc_tokens = [tokenize(f"{doc.title} {doc.path} {doc.text}") for doc in docs]
        self.doc_lengths = [len(tokens) or 1 for tokens in self.doc_tokens]
        self.avg_len = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        
        # IDF computation
        df: dict[str, int] = defaultdict(int)
        for tokens in self.doc_tokens:
            for token in set(tokens):
                df[token] += 1
        self.idf = {
            token: math.log(1 + (len(docs) - count + 0.5) / (count + 0.5))
            for token, count in df.items()
        }
        self.term_counts = [Counter(tokens) for tokens in self.doc_tokens]
        
        # Precompute embeddings for all documents (ONCE)
        print("[Retriever] Precomputing embeddings for corpus...")
        doc_texts = [f"{doc.title} {doc.path} {doc.text}" for doc in docs]
        self.embeddings = self.model.encode(doc_texts, convert_to_numpy=True)
        print(f"[Retriever] Precomputed {len(self.embeddings)} embeddings")

    def search(self, query: str, k: int = 5) -> list[Hit]:
        """
        Dual retrieval pipeline (NO CANDIDATE COLLAPSE):
        1. BM25 retrieval → top 10
        2. Embedding similarity (full corpus) → top 10
        3. Merge results, remove duplicates
        4. Score each with both metrics
        5. Hybrid score = 0.5 * bm25_norm + 0.5 * semantic
        6. Return top k
        """
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        # STEP 1: BM25 retrieval (full corpus)
        bm25_scores = []
        for idx in range(len(self.docs)):
            score = self._bm25(query_tokens, idx)
            bm25_scores.append(score)
        
        bm25_array = np.array(bm25_scores)
        top_10_bm25_indices = set(np.argsort(bm25_array)[-10:][::-1])
        
        # STEP 2: Semantic retrieval (full corpus)
        query_embedding = self.model.encode([query], convert_to_numpy=True)[0]
        
        semantic_scores = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-10
        )
        
        top_10_semantic_indices = set(np.argsort(semantic_scores)[-10:][::-1])
        
        # STEP 3: Merge both result sets, remove duplicates
        candidate_indices = top_10_bm25_indices | top_10_semantic_indices
        
        # STEP 4: Score each candidate (weighted differently based on source)
        candidates = []
        for idx in candidate_indices:
            doc = self.docs[idx]
            bm25_score = float(bm25_array[idx])
            semantic_score = float(semantic_scores[idx])
            in_bm25 = idx in top_10_bm25_indices
            
            candidates.append({
                "idx": idx,
                "doc": doc,
                "bm25_score": bm25_score,
                "semantic_score": semantic_score,
                "in_bm25": in_bm25,
            })
        
        if not candidates:
            return []
        
        # STEP 5A: Normalize semantic scores across ALL candidates (comparable units)
        # Currently semantic scores are in [-1, 1], normalize to [0, 1] based on candidate range
        semantic_scores_list = [c["semantic_score"] for c in candidates]
        semantic_min = min(semantic_scores_list)
        semantic_max = max(semantic_scores_list)
        semantic_range = semantic_max - semantic_min + 1e-6
        
        # BM25 normalization (only from documents in BM25 top 10)
        bm25_vals = [c["bm25_score"] for c in candidates if c["in_bm25"]]
        if bm25_vals:
            bm25_min = min(bm25_vals)
            bm25_max = max(bm25_vals)
            bm25_range = bm25_max - bm25_min + 1e-6
        else:
            bm25_range = 1e-6
        
        # Payment keywords for domain boosting
        payment_keywords = {
            "card", "transaction", "charge", "payment", "billing", "bill", 
            "charged", "paid", "dispute", "invoice", "refund", "fraud"
        }
        
        # STEP 5B: Compute final scores with proper normalization + domain boosting
        for c in candidates:
            # Normalize semantic score to [0, 1] based on candidate range
            semantic_norm = (c["semantic_score"] - semantic_min) / semantic_range
            
            if c["in_bm25"]:
                # Document found in BOTH BM25 and semantic: use hybrid (0.5/0.5)
                bm25_norm = (c["bm25_score"] - bm25_min) / bm25_range if bm25_vals else 0.5
                c["hybrid_score"] = 0.5 * bm25_norm + 0.5 * semantic_norm
            else:
                # Document found ONLY in semantic: use 0.7 weight (slight boost)
                c["hybrid_score"] = 0.7 * semantic_norm
            
            # Light domain boosting: if query has payment keywords and doc is Visa, boost slightly
            if c["doc"].domain == "visa" and any(kw in query.lower() for kw in payment_keywords):
                c["hybrid_score"] += 0.05
        
        # STEP 6: Sort by hybrid score and return top k
        candidates.sort(key=lambda c: c["hybrid_score"], reverse=True)
        
        hits = []
        for c in candidates[:k]:
            doc = c["doc"]
            snippet = self._snippet(doc.text, query_tokens)
            hit = Hit(
                doc=doc,
                score=c["hybrid_score"],
                bm25_score=c["bm25_score"],
                semantic_score=c["semantic_score"],
                snippet=snippet,
            )
            hits.append(hit)
        
        return hits


    def _bm25(self, query_tokens: list[str], idx: int) -> float:
        """Compute BM25 score for document at index."""
        k1 = 1.45
        b = 0.72
        counts = self.term_counts[idx]
        length = self.doc_lengths[idx]
        score = 0.0
        
        for token in query_tokens:
            freq = counts.get(token, 0)
            if not freq:
                continue
            idf = self.idf.get(token, 0.0)
            denom = freq + k1 * (1 - b + b * length / self.avg_len)
            score += idf * (freq * (k1 + 1)) / denom
        
        return score

    def _snippet(self, text: str, query_tokens: list[str]) -> str:
        """Extract snippet containing query tokens."""
        lowered = text.lower()
        positions = [lowered.find(token) for token in query_tokens if lowered.find(token) >= 0]
        
        if positions:
            start = max(min(positions) - 140, 0)
        else:
            start = 0
        
        snippet = " ".join(text[start : start + 520].split())
        return snippet

    def evidence_phrase(self, hit: Hit, query: str, max_words: int = 18) -> str:
        """Extract best evidence phrase from document."""
        tokens = set(tokenize(query))
        sentences = re.split(r"(?<=[.!?])\s+", hit.doc.text)
        
        best = ""
        best_score = -1
        
        for sentence in sentences[:80]:
            stripped = sentence.strip()
            if (
                stripped.startswith("#")
                or stripped.startswith("**Note")
                or "last updated" in stripped.lower()
                or "source_url" in stripped.lower()
            ):
                continue
            
            words = sentence.split()
            if len(words) < 4:
                continue
            
            overlap = len(tokens.intersection(tokenize(sentence)))
            if overlap > best_score:
                best = sentence
                best_score = overlap
        
        words = " ".join(best.split()).split()
        return " ".join(words[:max_words]) if words else hit.doc.title
