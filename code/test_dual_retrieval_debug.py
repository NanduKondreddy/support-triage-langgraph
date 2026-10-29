#!/usr/bin/env python
"""
Debug: Verify dual retrieval is finding candidates from BOTH BM25 and semantic searches
"""

import sys
import time
from pathlib import Path
import numpy as np
from retriever_v2 import HybridRetriever, tokenize
from corpus_loader import load_corpus

# Fix encoding
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 80)
print("DEBUG: DUAL RETRIEVAL MECHANISM")
print("=" * 80)
print()

# Load corpus
print("[Loading corpus...]")
data_dir = Path("../data")
corpus = load_corpus(data_dir)
print(f"Loaded {len(corpus)} documents")
print()

# Initialize retriever
print("[Initializing HybridRetriever...]")
retriever = HybridRetriever(corpus)
print()

# Test query
query = "charged incorrectly"
query_tokens = tokenize(query)

print(f"Query: {query}")
print(f"Tokens: {query_tokens}")
print()

# Manually compute what the retriever does
print("[STEP 1: BM25 retrieval (full corpus)]")
bm25_scores = []
for idx in range(len(corpus)):
    score = retriever._bm25(query_tokens, idx)
    bm25_scores.append(score)

bm25_array = np.array(bm25_scores)
top_10_bm25_indices = np.argsort(bm25_array)[-10:][::-1]

print(f"Top 10 BM25 indices: {top_10_bm25_indices}")
for rank, idx in enumerate(top_10_bm25_indices, 1):
    doc = corpus[idx]
    print(f"  [{rank}] {doc.domain:12} | Score: {bm25_array[idx]:7.2f} | {doc.title[:50]}")
print()

print("[STEP 2: Semantic retrieval (full corpus)]")
query_embedding = retriever.model.encode([query], convert_to_numpy=True)[0]
semantic_scores = np.dot(retriever.embeddings, query_embedding) / (
    np.linalg.norm(retriever.embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-10
)
top_10_semantic_indices = np.argsort(semantic_scores)[-10:][::-1]

print(f"Top 10 Semantic indices: {top_10_semantic_indices}")
for rank, idx in enumerate(top_10_semantic_indices, 1):
    doc = corpus[idx]
    print(f"  [{rank}] {doc.domain:12} | Score: {semantic_scores[idx]:7.3f} | {doc.title[:50]}")
print()

print("[STEP 3: Merged candidate set]")
candidate_indices = set(top_10_bm25_indices) | set(top_10_semantic_indices)
print(f"Total unique candidates: {len(candidate_indices)}")

# Count by domain
domain_counts = {}
for idx in candidate_indices:
    doc = corpus[idx]
    domain_counts[doc.domain] = domain_counts.get(doc.domain, 0) + 1

print(f"By domain: {domain_counts}")
print()

# Check if ANY Visa docs are in candidate set
visa_candidates = [idx for idx in candidate_indices if corpus[idx].domain == "visa"]
print(f"Visa candidates in merged set: {len(visa_candidates)}")
if visa_candidates:
    for idx in visa_candidates[:3]:
        doc = corpus[idx]
        print(f"  - {doc.title[:60]}")
print()

print("[STEP 4: Ranking ALL candidates]")
results = retriever.search(query, k=10)
print(f"Top 10 results:")
for rank, hit in enumerate(results, 1):
    print(f"  [{rank}] {hit.doc.domain:12} | Score: {hit.score:.3f} (BM25: {hit.bm25_score:7.2f}, Semantic: {hit.semantic_score:7.3f}) | {hit.doc.title[:50]}")

print()
print("=" * 80)
