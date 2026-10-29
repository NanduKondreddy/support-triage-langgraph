"""
Decision Engine — Controlled Decision Intelligence Layer

This module provides structured reasoning for the support triage agent while maintaining
safety and determinism. It analyzes query intent, retrieved chunks, and risk signals
to make intelligent decisions about whether to answer or escalate.

RULES:
- LLM must NOT decide safety
- LLM must NOT override escalation
- LLM must NOT invent knowledge
- LLM can ONLY help interpret intent and structure response
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Decision:
    """Structured decision output from the decision engine."""
    intent: str
    confidence: float
    safe_to_answer: bool
    reasoning: str


# ============================================================================
# Static Decision Rules (No LLM needed)
# ============================================================================

# Queries that are CLEARLY answerable without any AI assistance
CLEARLY_ANSWERABLE = {
    "password reset": "password_reset",
    "reset password": "password_reset",
    "forgot password": "password_reset",
    "change password": "password_reset",
    "update password": "password_reset",
    "how do i": "how_to",
    "how to": "how_to",
    "where do i": "how_to",
    "where to": "how_to",
    "can i": "permission",
    "is it possible": "permission",
    "what is": "factual",
    "what are": "factual",
    "tell me about": "factual",
}

# Queries that are CLEARLY NOT safe to answer without human review
CLEARLY_UNSAFE = {
    "refund": "billing_refund",
    "give me my money": "billing_refund",
    "increase my score": "account_authority",
    "change my score": "account_authority",
    "regrade": "account_authority",
    "review my answers": "account_authority",
    "answer key": "privacy_data",
    "solution": "privacy_data",
    "export all": "privacy_data",
    "don't remember": "fraud",
    "don't recognize": "fraud",
    "not me": "fraud",
    "unauthorized": "fraud",
    "someone else": "fraud",
    "suspicious": "fraud",
    "compromised": "account_compromise",
    "hacked": "account_compromise",
    "stolen": "fraud",
    "identity theft": "account_compromise",
    "legal": "legal",
    "lawsuit": "legal",
    "attorney": "legal",
    "lawyer": "legal",
}


def analyze_vague(text: str) -> bool:
    """Check if query is too vague to answer safely."""
    vague_patterns = [
        r"^it'?s (not working|broken|wrong)$",
        r"^help( me)?$",
        r"^something(')?s wrong$",
        r"^things? (are |is )?(broken|wrong)$",
        r"^acting (strange|weird|funny)$",
        r"^issue$",
        r"^problem$",
    ]
    import re
    text_clean = text.strip().lower()
    if len(text_clean.split()) > 15:
        return False
    for pattern in vague_patterns:
        if re.search(pattern, text_clean):
            return True
    return False


def evaluate_context_match(query: str, retrieved_chunks: list) -> dict:
    """
    Evaluate how well retrieved chunks match the query.
    
    Returns:
        {
            "match_score": float,  # 0-1 score
            "direct_answer": bool,  # chunks directly answer query
            "partial_match": bool,  # chunks partially related
            "reasoning": str
        }
    """
    if not retrieved_chunks:
        return {
            "match_score": 0.0,
            "direct_answer": False,
            "partial_match": False,
            "reasoning": "No chunks retrieved"
        }
    
    query_lower = query.lower()
    query_tokens = set(query_lower.split())
    
    # Stopwords to ignore
    stopwords = {"my", "has", "been", "since", "the", "a", "an", "is", "for", "to", "in", 
               "on", "with", "of", "and", "or", "but", "it", "that", "this", "be", "are", 
               "from", "as", "at", "what", "how", "why", "who", "when", "i", "you", "your"}
    query_key_tokens = query_tokens - stopwords
    
    # Check top chunks for relevance
    matched_chunks = 0
    partial_chunks = 0
    
    for chunk in retrieved_chunks[:5]:
        chunk_text = chunk.doc.text.lower() if chunk.doc else ""
        chunk_tokens = set(chunk_text.split()) - stopwords
        
        # Calculate token overlap
        overlap = len(query_key_tokens & chunk_tokens)
        overlap_ratio = overlap / max(len(query_key_tokens), 1)
        
        if overlap_ratio >= 0.3:
            matched_chunks += 1
        elif overlap_ratio >= 0.1:
            partial_chunks += 1
    
    # Compute match score
    if matched_chunks >= 3:
        match_score = 0.85
        direct_answer = True
        partial_match = False
        reasoning = f"Strong context match: {matched_chunks}/5 chunks match query"
    elif matched_chunks >= 1 and matched_chunks + partial_chunks >= 2:
        match_score = 0.65
        direct_answer = False
        partial_match = True
        reasoning = f"Partial context match: {matched_chunks} direct, {partial_chunks} partial"
    elif partial_chunks >= 2:
        match_score = 0.45
        direct_answer = False
        partial_match = True
        reasoning = f"Weak context match: {partial_chunks} partial only"
    else:
        match_score = 0.25
        direct_answer = False
        partial_match = False
        reasoning = "Insufficient context match"
    
    return {
        "match_score": match_score,
        "direct_answer": direct_answer,
        "partial_match": partial_match,
        "reasoning": reasoning
    }


def check_retrieval_quality(retrieved_chunks: list, query: str) -> dict:
    """Check the quality and relevance of retrieved chunks."""
    if not retrieved_chunks:
        return {
            "quality": "none",
            "score": 0.0,
            "issue": "No chunks retrieved",
        }
    
    # Check if top chunk has meaningful overlap with query
    top_chunk = retrieved_chunks[0]
    top_score = top_chunk.score if hasattr(top_chunk, 'score') else 0.0
    
    if top_score < 0.30:
        return {
            "quality": "weak",
            "score": top_score,
            "issue": "Low retrieval confidence",
        }
    
    # Check lexical overlap
    query_tokens = set(query.lower().split())
    chunk_text = top_chunk.doc.text.lower() if top_chunk.doc else ""
    overlap = len(query_tokens & set(chunk_text.split()))
    
    if overlap == 0 and len(query_tokens) >= 3:
        return {
            "quality": "weak",
            "score": top_score,
            "issue": "Zero lexical overlap",
        }
    
    return {
        "quality": "good" if top_score >= 0.45 else "acceptable",
        "score": top_score,
        "issue": None,
    }


def static_decision(
    query: str,
    retrieved_chunks: list,
    semantic_risk: tuple | None,
    retrieval_confidence: float,
) -> Decision:
    """
    Make a decision using static rules + context-based reasoning.
    
    This is the deterministic fallback when LLM is unavailable.
    Uses evaluate_context_match to determine if chunks directly answer the query.
    """
    query_lower = query.lower()
    
# 1. Check for clearly unsafe patterns first
    for pattern, intent in CLEARLY_UNSAFE.items():
        if pattern in query_lower:
            return Decision(
                intent=intent,
                confidence=0.95,
                safe_to_answer=False,
                reasoning=f"Unsafe pattern detected: '{pattern}'",
            )
    
    # 2. Check for vague queries
    if analyze_vague(query):
        return Decision(
            intent="vague",
            confidence=0.99,
            safe_to_answer=False,
            reasoning="Query too vague to answer safely",
        )
    
    # 3. Check retrieval quality
    retrieval_quality = check_retrieval_quality(retrieved_chunks, query)
    if retrieval_quality["quality"] == "none":
        return Decision(
            intent="unknown",
            confidence=0.0,
            safe_to_answer=False,
            reasoning="No relevant documentation found",
        )
    
    if retrieval_quality["quality"] == "weak":
        return Decision(
            intent="unclear",
            confidence=retrieval_quality["score"],
            safe_to_answer=False,
            reasoning=f"Weak retrieval: {retrieval_quality['issue']}",
        )
    
    # 4. Check semantic risk
    if semantic_risk is not None:
        label, score, is_ambiguous = semantic_risk
        if label is not None or is_ambiguous:
            return Decision(
                intent=label or "risk_detected",
                confidence=1.0 - score,
                safe_to_answer=False,
                reasoning=f"Semantic risk: {label} ({score:.2f})",
            )
    
    # 5. CONTEXT-BASED REASONING: Evaluate if chunks directly answer query
    context_eval = evaluate_context_match(query, retrieved_chunks)

    # Use match_score > 0.6 threshold for safe_to_answer
    if context_eval["match_score"] > 0.6:
        # Good context match - safe to answer
        confidence_from_context = context_eval["match_score"] * retrieval_confidence
        return Decision(
            intent="support_query",
            confidence=max(confidence_from_context, 0.65),
            safe_to_answer=True,
            reasoning=context_eval["reasoning"],
        )
    else:
        # Weak context match - escalate
        return Decision(
            intent="unclear",
            confidence=context_eval["match_score"],
            safe_to_answer=False,
            reasoning=context_eval["reasoning"] + " - Insufficient context match",
        )
    
    # Should not reach here, but safety net
    return Decision(
        intent="unclear",
        confidence=retrieval_confidence,
        safe_to_answer=False,
        reasoning="Safety fallback - insufficient confidence",
    )


# ============================================================================
# LLM-Assisted Decision (Optional Enhancement)
# ============================================================================

def llm_assisted_decision(
    query: str,
    retrieved_chunks: list,
    semantic_risk: tuple | None,
    retrieval_confidence: float,
) -> Decision:
    """
    Make a decision with optional LLM assistance.
    
    The LLM helps interpret intent and structure the response,
    but does NOT override safety decisions.
    """
    # First run static decision
    static_result = static_decision(query, retrieved_chunks, semantic_risk, retrieval_confidence)
    
    # If static decision is already unsafe, use it
    if not static_result.safe_to_answer:
        return static_result
    
    # If static decision is confident, use it
    if static_result.confidence >= 0.7:
        return static_result
    
    # Only use LLM to help interpret unclear cases
    # where we already think it's safe to answer
    try:
        import os
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return static_result
        
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        
        # Build context from chunks
        chunk_summaries = []
        for chunk in retrieved_chunks[:3]:
            summary = f"- {chunk.doc.title}: {chunk.doc.text[:150]}..."
            chunk_summaries.append(summary)
        
        context = "\n".join(chunk_summaries)
        
        prompt = f"""You are a support decision assistant helping classify user intent.

Query: {query}

Retrieved context:
{context}

Semantic risk: {semantic_risk[0] if semantic_risk and semantic_risk[0] else 'None'}

Return ONLY a JSON object with:
{{
    "intent": "password_reset" | "how_to" | "billing" | "account" | "technical" | "other",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}

Do NOT override safety. Do NOT invent information."""
        
        message = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}]
        )
        
        import json
        try:
            response = json.loads(message.content[0].text)
            return Decision(
                intent=response.get("intent", static_result.intent),
                confidence=response.get("confidence", static_result.confidence),
                safe_to_answer=True,
                reasoning=response.get("reasoning", static_result.reasoning),
            )
        except (json.JSONDecodeError, AttributeError):
            return static_result
            
    except Exception:
        return static_result


def make_decision(
    query: str,
    retrieved_chunks: list,
    semantic_risk: tuple | None,
    retrieval_confidence: float,
    use_llm: bool = False,
) -> Decision:
    """
    Main decision entry point.
    
    Args:
        query: User issue text
        retrieved_chunks: Top chunks from retriever
        semantic_risk: (label, score, ambiguity) tuple
        retrieval_confidence: BM25/retrieval confidence score
        use_llm: Whether to use LLM for ambiguous cases
    
    Returns:
        Decision with intent, confidence, safe_to_answer, reasoning
    """
    if use_llm:
        return llm_assisted_decision(
            query, retrieved_chunks, semantic_risk, retrieval_confidence
        )
    return static_decision(
        query, retrieved_chunks, semantic_risk, retrieval_confidence
    )


# ============================================================================
# Test Functions
# ============================================================================

def run_tests():
    """Run basic decision tests."""
    test_cases = [
        # (query, expected_safe, expected_intent)
        ("How do I reset my password?", True, "password_reset"),
        ("I don't recognize this charge", False, "fraud"),
        ("Something is wrong", False, "vague"),
        ("My card was declined abroad", False, "unclear"),
        ("How do I use the API?", True, "how_to"),
        ("I want a refund", False, "billing_refund"),
    ]
    
    print("Running decision engine tests...\n")
    
    for query, expected_safe, expected_intent in test_cases:
        decision = make_decision(
            query=query,
            retrieved_chunks=[],
            semantic_risk=None,
            retrieval_confidence=0.5,
        )
        
        status = "✓" if decision.safe_to_answer == expected_safe else "✗"
        intent_match = "✓" if decision.intent == expected_intent else "✗"
        
        print(f"{status} Query: {query}")
        print(f"   Intent: {decision.intent} ({intent_match})")
        print(f"   Safe: {decision.safe_to_answer} (expected: {expected_safe})")
        print(f"   Reasoning: {decision.reasoning}")
        print()


if __name__ == "__main__":
    run_tests()
