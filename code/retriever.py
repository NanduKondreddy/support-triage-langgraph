from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import os
import re

try:
    USE_EMBEDDINGS = os.getenv("USE_EMBEDDINGS", "false").lower() == "true"
    if USE_EMBEDDINGS:
        import numpy as np
        from sentence_transformers import SentenceTransformer
        SEMANTIC_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    else:
        SEMANTIC_MODEL = None
except ImportError:
    SEMANTIC_MODEL = None

from corpus_loader import Document

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_+'-]*", re.I)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "for", "from", "have", "how",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "our", "please", "that", "the",
    "this", "to", "with", "you", "your",
}

# ---------------------------------------------------------------------------
# Query expansion: bridges paraphrased user language → corpus vocabulary.
# Each key is a phrase the user might write; values are corpus-aligned synonyms
# injected into the query so BM25 + TF-IDF scores improve on semantic mismatches.
# Keep this list conservative — only add pairs that are genuinely equivalent.
# ---------------------------------------------------------------------------
SYNONYM_MAP: dict[str, list[str]] = {
    # Session / access
    "kicked out":           ["session timeout", "inactivity", "lobby", "session expired"],
    "sent back":            ["timeout", "inactivity", "session expired"],
    "logged out":           ["session timeout", "inactivity"],
    "not working":          ["error", "failed", "issue", "unavailable"],
    "stopped working":      ["down", "failing", "unavailable", "outage"],
    "can't access":         ["access denied", "restricted", "blocked", "unable"],
    "cannot access":        ["access denied", "restricted", "blocked"],
    "not loading":          ["error", "failed", "unavailable"],
    # Card / payment
    "card isn't going through": ["transaction declined", "card declined", "payment failed"],
    "card not working":     ["card declined", "transaction declined"],
    "blocked card":         ["card restricted", "card declined", "issuer"],
    "money deducted":       ["charged", "payment deducted", "transaction"],
    "abroad":               ["international", "travel", "overseas", "foreign transaction"],
    "overseas":             ["international", "travel", "foreign transaction"],
    # Assessment / candidate
    "interview kicked":     ["session timeout", "inactivity", "lobby"],
    "test not opening":     ["assessment failed", "test error", "unable to start"],
    "link expired":         ["test expiration", "expired link", "reinvite candidate"],
    "reinvite":             ["test expiration", "resend invitation", "candidate attempt"],
    # User management
    "left the company":     ["deactivate user", "remove user", "roles management"],
    "employee left":        ["deactivate user", "remove user", "roles management"],
    "remove them":          ["deactivate user", "remove user", "roles management"],
    # Infrastructure
    "all requests failing": ["outage", "service down", "api failing", "sitewide"],
    "requests failing":     ["api error", "outage", "service unavailable"],
    "timing out":           ["timeout", "request timeout", "latency"],
    # Visa fraud
    "scam call":            ["suspicious call", "fraud", "otp scam"],
    "fake item":            ["wrong product", "merchant dispute", "chargeback"],
    "wrong item":           ["wrong product", "merchant dispute", "chargeback"],
    # Claude
    "crawl my site":        ["web crawler", "anthropic crawler", "block crawler"],
    "training data":        ["model improvement", "data retention", "privacy"],
}


def expand_query(text: str) -> str:
    """Expand user phrasing to corpus-aligned vocabulary for better retrieval.

    Appends synonym tokens to the query without removing the original text,
    so the original signal is preserved and new signal is additive only.
    """
    lower = text.lower()
    extras: list[str] = []
    for phrase, synonyms in SYNONYM_MAP.items():
        if phrase in lower:
            extras.extend(synonyms)
    if extras:
        return text + " " + " ".join(extras)
    return text


@dataclass(frozen=True)
class Hit:
    doc: Document
    score: float
    bm25_score: float
    vector_score: float
    overlap: int
    snippet: str


def tokenize(text: str) -> list[str]:
    return [tok.lower().strip("-'") for tok in TOKEN_RE.findall(text) if tok.lower() not in STOPWORDS]


class Retriever:
    def __init__(self, docs: list[Document]) -> None:
        self.docs = docs
        self.doc_tokens = [tokenize(f"{doc.title} {doc.path} {doc.text}") for doc in docs]
        self.doc_lengths = [len(tokens) or 1 for tokens in self.doc_tokens]
        self.avg_len = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        df: dict[str, int] = defaultdict(int)
        for tokens in self.doc_tokens:
            for token in set(tokens):
                df[token] += 1
        self.idf = {token: math.log(1 + (len(docs) - count + 0.5) / (count + 0.5)) for token, count in df.items()}
        self.term_counts = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_vectors = [self._tfidf_vector(counts) for counts in self.term_counts]
        self.doc_norms = [self._norm(vector) for vector in self.doc_vectors]

        self.embeddings = None
        if SEMANTIC_MODEL is not None:
            import numpy as np
            from pathlib import Path
            emb_path = Path("data/embeddings.npy")
            if emb_path.exists():
                self.embeddings = np.load(emb_path)
            else:
                self.embeddings = None
                global USE_EMBEDDINGS
                USE_EMBEDDINGS = False

    def search(self, query: str, domain: str | None = None, k: int = 5) -> list[Hit]:
        # Preserve the original query tokens for post-retrieval re-ranking.
        original_tokens = tokenize(query)
        # Expand query before tokenization to bridge paraphrase gaps
        query = expand_query(query)
        
        import os
        if os.getenv("USE_EMBEDDINGS", "false").lower() == "true" and self.embeddings is not None and SEMANTIC_MODEL is not None:
            try:
                import numpy as np
                q_emb = SEMANTIC_MODEL.encode([query])[0]
                norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(q_emb)
                norms[norms == 0] = 1e-10
                sims = np.dot(self.embeddings, q_emb) / norms
                top_idx = sims.argsort()[-2:][::-1]
                expansions = [self.docs[i].text[:50] for i in top_idx if sims[i] > 0.4]
                if expansions:
                    exp_query = query + " " + " ".join(expansions)
                    exp_emb = SEMANTIC_MODEL.encode([exp_query])[0]
                    drift_sim = np.dot(q_emb, exp_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(exp_emb) + 1e-10)
                    if drift_sim > 0.8:
                        query = exp_query
            except Exception:
                pass
                
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        query_vector = self._tfidf_vector(Counter(query_tokens))
        query_norm = self._norm(query_vector)
        
        semantic_scores = None
        if self.embeddings is not None and SEMANTIC_MODEL is not None:
            import numpy as np
            q_emb = SEMANTIC_MODEL.encode([query])[0]
            norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(q_emb)
            norms[norms == 0] = 1e-10
            semantic_scores = np.dot(self.embeddings, q_emb) / norms
            
        scored: list[Hit] = []
        raw_scores = []
        sem_scores_filtered = []
        doc_indices = []
        
        for idx, doc in enumerate(self.docs):
            if domain and doc.domain != domain:
                continue
            bm25_score = self._bm25(query_tokens, idx)
            vector_score = self._cosine(query_vector, query_norm, idx)
            title_overlap = sum(1 for token in set(query_tokens) if token in tokenize(doc.title + " " + doc.path))
            body_overlap = len(set(query_tokens).intersection(set(self.doc_tokens[idx])))
            title_bonus = 0.45 * title_overlap
            overlap_bonus = min(body_overlap, 8) * 0.08
            
            raw_score = bm25_score + 0.3 * vector_score + title_bonus + overlap_bonus
            raw_scores.append(raw_score)
            if USE_EMBEDDINGS and semantic_scores is not None:
                sem_scores_filtered.append(float(semantic_scores[idx]))
            doc_indices.append((doc, bm25_score, vector_score, title_overlap))
            
        def normalize(scores):
            if not scores: return []
            min_s, max_s = min(scores), max(scores)
            if max_s - min_s < 1e-6:
                return [0.5] * len(scores)
            return [(s - min_s) / (max_s - min_s) for s in scores]

        if USE_EMBEDDINGS and semantic_scores is not None:
            norm_raw = normalize(raw_scores)
            norm_sem = normalize(sem_scores_filtered)
            w = 0.7 if len(query_tokens) <= 5 else 0.5
            final_scores = [w * norm_raw[i] + (1 - w) * norm_sem[i] for i in range(len(norm_raw))]
        else:
            final_scores = raw_scores

        for i, (doc, bm25_score, vector_score, title_overlap) in enumerate(doc_indices):
            if final_scores[i] > 0.0:
                scored.append(
                    Hit(
                        doc=doc,
                        score=final_scores[i],
                        bm25_score=bm25_score,
                        vector_score=vector_score,
                        overlap=title_overlap,
                        snippet=self._snippet(doc.text, query_tokens),
                    )
                )
        scored.sort(key=lambda hit: hit.score, reverse=True)
        # Post-retrieval re-rank: score top candidates against the ORIGINAL
        # (un-expanded) query to strip expansion noise and restore precision.
        candidates = scored[: k * 3]  # generous candidate pool
        if original_tokens and candidates:
            original_set = set(original_tokens)
            for hit in candidates:
                doc_idx = self.docs.index(hit.doc)
                orig_bm25 = self._bm25(original_tokens, doc_idx)
                orig_overlap = len(original_set.intersection(set(self.doc_tokens[doc_idx])))
                # Blend: original signal gets 30% weight in final ordering.
                hit_reranked = hit.score + 0.30 * orig_bm25 + 0.05 * min(orig_overlap, 6)
                # Store as a new score via replacement (frozen dataclass workaround)
                scored[scored.index(hit)] = Hit(
                    doc=hit.doc,
                    score=hit_reranked,
                    bm25_score=hit.bm25_score,
                    vector_score=hit.vector_score,
                    overlap=hit.overlap,
                    snippet=hit.snippet,
                )
            scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:k]

    def _bm25(self, query_tokens: list[str], idx: int) -> float:
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
        lowered = text.lower()
        positions = [lowered.find(token) for token in query_tokens if lowered.find(token) >= 0]
        start = max(min(positions) - 140, 0) if positions else 0
        snippet = " ".join(text[start : start + 520].split())
        return snippet

    def evidence_phrase(self, hit: Hit, query: str, max_words: int = 18) -> str:
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

    def _tfidf_vector(self, counts: Counter[str]) -> dict[str, float]:
        return {token: (1 + math.log(freq)) * self.idf.get(token, 0.0) for token, freq in counts.items() if freq > 0}

    def _norm(self, vector: dict[str, float]) -> float:
        return math.sqrt(sum(value * value for value in vector.values())) or 1.0

    def _cosine(self, query_vector: dict[str, float], query_norm: float, idx: int) -> float:
        doc_vector = self.doc_vectors[idx]
        overlap = set(query_vector).intersection(doc_vector)
        if not overlap:
            return 0.0
        dot = sum(query_vector[token] * doc_vector[token] for token in overlap)
        return dot / (query_norm * self.doc_norms[idx])
