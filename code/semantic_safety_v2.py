"""
Semantic Safety Detection Module (Hardened)

Fixes 4 critical gaps:
1. Multi-label detection (not single-label)
2. False positive reduction (self-reference check)
3. Negation handling (not fraud = safe)
4. Confidence bands (HIGH/MEDIUM/LOW, not binary)
"""

import numpy as np

# Global singleton model (load once)
_MODEL = None

def _get_model():
    """Lazy load sentence transformer model once."""
    global _MODEL
    if _MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            print(f"Warning: Failed to load semantic model: {e}")
            _MODEL = False
    return _MODEL if _MODEL is not False else None


# Semantic risk categories with DIVERSE seed sentences
RISK_CATEGORIES = {
    "fraud": [
        "I don't remember making this payment",
        "I never authorized this transaction",
        "This charge is not legitimate",
        "Someone fraudulently used my account",
        "I want to dispute this charge",
        "That purchase wasn't me",
        "Unknown charge on my account",
        "This transaction looks suspicious",
    ],
    "account_compromise": [
        "I think someone else is using my account",
        "My account feels compromised",
        "Something unusual is happening with my account",
        "Someone might have hacked my account",
        "There's suspicious activity on my account",
        "I didn't make these changes",
        "My account was accessed without permission",
        "I lost control of my account",
    ],
    "self_harm": [
        "I want to end everything",
        "I'm having suicidal thoughts",
        "I can't take this anymore",
        "I want to hurt myself",
        "Life is not worth living",
        "I'm thinking of ending it",
        "I feel like giving up",
        "I don't want to exist",
    ],
    "privacy": [
        "Show me my private data",
        "I need to access sensitive user information",
        "Display confidential information",
        "Can you show me API logs with credentials",
        "I want to see other users' data",
        "Extract sensitive records",
        "Bypass data protection",
        "I need someone else's personal info",
    ],
    "legal": [
        "My lawyer is contacting you",
        "I'm filing a lawsuit",
        "This is a legal matter",
        "I'm taking legal action",
        "My attorney will be in touch",
        "I'm getting a court order",
        "You'll be hearing from my legal team",
        "I'm pursing this legally",
    ],
}

# Precompute category embeddings
_CATEGORY_EMBEDDINGS = {}

# Configuration (calibrated for production)
SEMANTIC_THRESHOLD_HIGH = 0.65  # Clear risk
SEMANTIC_THRESHOLD_MEDIUM = 0.47  # Borderline (escalate soft)
AMBIGUITY_THRESHOLD = 0.04

# Negations to check
NEGATIONS = {"not", "no", "never", "don't", "doesn't", "didn't", "won't", "can't", "isn't", "aren't"}

# Self-reference words (first person only)
SELF_REFERENCES = {"my", "i", "me", "mine", "myself"}

# Third-person pronouns (NOT nouns - strictly pronouns/determiners)
THIRD_PERSON = {"he", "she", "their", "theirs", "them", "his", "hers"}

def _has_negation_near_risk(text: str, risk_keywords: list[str]) -> bool:
    """Check if negation appears near risk keywords."""
    text_lower = text.lower()
    words = text_lower.split()
    
    for keyword in risk_keywords:
        if keyword in text_lower:
            # Find keyword in words (handle multi-word keywords like "account compromise")
            keyword_words = keyword.split()
            for i, word in enumerate(words):
                if word.strip('.,!?;:') == keyword_words[0] or keyword_words[0] in word:
                    # Check 5 words before for negations
                    context_start = max(0, i - 5)
                    context_words = words[context_start:i+1]
                    context_text = ' '.join(context_words)
                    if any(neg in context_text for neg in NEGATIONS):
                        return True
    
    return False

def _has_self_reference(text: str) -> bool:
    """Check if text contains self-references (my, I, me). Return False if third-person."""
    text_lower = text.lower()
    words = text_lower.split()
    
    # Check for first-person pronouns
    has_self_ref = any(word in SELF_REFERENCES for word in words) or \
                   any(word.strip('.,!?;:') in SELF_REFERENCES for word in words)
    
    # Check for third-person pronouns
    has_third_person = any(word in THIRD_PERSON for word in words) or \
                       any(word.strip('.,!?;:') in THIRD_PERSON for word in words)
    
    # If third-person is present, it's NOT self-reference
    if has_third_person:
        return False
    
    return has_self_ref

def _precompute_embeddings():
    """Precompute MEAN embeddings for each category."""
    global _CATEGORY_EMBEDDINGS
    if _CATEGORY_EMBEDDINGS:
        return
    
    model = _get_model()
    if model is None:
        return
    
    for category, sentences in RISK_CATEGORIES.items():
        embeddings = model.encode(sentences, convert_to_numpy=True)
        mean_embedding = np.mean(embeddings, axis=0)
        norm = mean_embedding / (np.linalg.norm(mean_embedding) + 1e-10)
        _CATEGORY_EMBEDDINGS[category] = norm


def semantic_risk_detect(text: str) -> dict:
    """
    Detect semantic risk in text using embeddings (HARDENED).
    
    Args:
        text: Input text to analyze
    
    Returns:
        {
            "labels": list[str],  # Multiple labels if needed
            "scores": dict,  # Score per category
            "best_score": float,  # Highest score
            "is_ambiguous": bool,  # Top 2 scores close
            "confidence": str,  # "HIGH", "MEDIUM", "LOW"
            "negation_detected": bool,  # Negation flag
            "self_reference": bool,  # Self-reference flag
        }
    """
    if not text or len(text.strip()) == 0:
        return {
            "labels": [],
            "scores": {},
            "best_score": 0.0,
            "is_ambiguous": False,
            "confidence": "LOW",
            "negation_detected": False,
            "self_reference": False,
        }
    
    model = _get_model()
    if model is None:
        return {
            "labels": [],
            "scores": {},
            "best_score": 0.0,
            "is_ambiguous": False,
            "confidence": "LOW",
            "negation_detected": False,
            "self_reference": False,
        }
    
    # Precompute on first call
    if not _CATEGORY_EMBEDDINGS:
        _precompute_embeddings()
    
    if not _CATEGORY_EMBEDDINGS:
        return {
            "labels": [],
            "scores": {},
            "best_score": 0.0,
            "is_ambiguous": False,
            "confidence": "LOW",
            "negation_detected": False,
            "self_reference": False,
        }
    
    # Encode text and normalize
    try:
        text_emb = model.encode([text], convert_to_numpy=True)[0]
        text_norm = text_emb / (np.linalg.norm(text_emb) + 1e-10)
    except Exception:
        return {
            "labels": [],
            "scores": {},
            "best_score": 0.0,
            "is_ambiguous": False,
            "confidence": "LOW",
            "negation_detected": False,
            "self_reference": False,
        }
    
    # Check for negations and self-references
    has_negation = _has_negation_near_risk(text, list(RISK_CATEGORIES.keys()))
    has_self_ref = _has_self_reference(text)
    
    # Compute scores for all categories
    scores = {}
    for category, category_emb in _CATEGORY_EMBEDDINGS.items():
        sim = float(np.dot(text_norm, category_emb))
        
        # GAP 3: Reduce score if negation detected
        if has_negation:
            sim = sim * 0.4  # Significant penalty for negation
        
        # GAP 2: Reduce score if no self-reference (third-person report)
        # But NOT for legal threats (those can come from third parties)
        if not has_self_ref and category != "legal":
            sim = sim * 0.65  # Strong penalty for third-person reports
        
        scores[category] = sim
    
    # Sort by score
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_category, best_score = sorted_scores[0]
    second_category, second_score = sorted_scores[1] if len(sorted_scores) > 1 else (None, 0.0)
    
    # Check ambiguity
    is_ambiguous = False
    if second_category and (best_score - second_score) < AMBIGUITY_THRESHOLD:
        is_ambiguous = True
    
    # GAP 4: Determine confidence band
    if best_score >= SEMANTIC_THRESHOLD_HIGH:
        confidence = "HIGH"
    elif best_score >= SEMANTIC_THRESHOLD_MEDIUM:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"
    
    # GAP 1: Multi-label detection
    # Return multiple labels if they exceed MEDIUM threshold
    labels = []
    if best_score >= SEMANTIC_THRESHOLD_MEDIUM:
        labels.append(best_category)
    if second_category and second_score >= SEMANTIC_THRESHOLD_MEDIUM:
        labels.append(second_category)
    
    return {
        "labels": labels,
        "scores": scores,
        "best_score": best_score,
        "is_ambiguous": is_ambiguous,
        "confidence": confidence,
        "negation_detected": has_negation,
        "self_reference": has_self_ref,
    }


# Initialize on import
_precompute_embeddings()
