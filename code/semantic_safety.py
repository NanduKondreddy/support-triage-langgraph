"""
Semantic Safety Detection Module

Uses sentence embeddings (all-MiniLM-L6-v2) to detect high-risk categories:
- fraud
- account_compromise
- self_harm
- privacy
- legal

Catches soft signals that bypass keyword-based detection.

Architecture:
- Diverse seed sentences per category (avoid overfitting)
- Mean embeddings per category (robust representation)
- Ambiguity detection (when top 2 scores are close)
- Configurable threshold (calibrated for production)
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
            _MODEL = False  # Mark as failed
    return _MODEL if _MODEL is not False else None


# Semantic risk categories with DIVERSE seed sentences
# Each category has 7-8 diverse phrasings to avoid overfitting
RISK_CATEGORIES = {
    "fraud": [
        "I don't remember making this payment",
        "I never authorized this transaction",
        "This charge is not legitimate",
        "Someone fraudulently used my account",
        "I want to dispute this charge",
        "That purchase wasn't me",
        "Unknown charge on my account",
        "I'm claiming this as fraud",
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
# NOTE: Threshold calibrated for mean-embedding approach
# - Lower than 0.65 because using category means (not individual seeds)
# - Validated on diverse seed set of 8 sentences per category
SEMANTIC_THRESHOLD = 0.60  # Calibrated for mean embeddings
AMBIGUITY_THRESHOLD = 0.04  # Stricter for ambiguity (mean approach has tighter clustering)

def _precompute_embeddings():
    """Precompute MEAN embeddings for each category."""
    global _CATEGORY_EMBEDDINGS
    if _CATEGORY_EMBEDDINGS:
        return
    
    model = _get_model()
    if model is None:
        return
    
    for category, sentences in RISK_CATEGORIES.items():
        # Encode all seed sentences
        embeddings = model.encode(sentences, convert_to_numpy=True)
        
        # Compute mean embedding (robust representation)
        mean_embedding = np.mean(embeddings, axis=0)
        
        # Normalize the mean
        norm = mean_embedding / (np.linalg.norm(mean_embedding) + 1e-10)
        
        _CATEGORY_EMBEDDINGS[category] = norm


def semantic_risk_detect(text: str, threshold: float = None) -> tuple[str | None, float, bool]:
    """
    Detect semantic risk in text using embeddings.
    
    Args:
        text: Input text to analyze
        threshold: Minimum similarity score to flag risk (uses SEMANTIC_THRESHOLD if None)
    
    Returns:
        (category_label, best_score, is_ambiguous)
        - category_label: Risk category or None if below threshold
        - best_score: Highest similarity score
        - is_ambiguous: True if top 2 scores are within AMBIGUITY_THRESHOLD
    """
    if not text or len(text.strip()) == 0:
        return None, 0.0, False
    
    if threshold is None:
        threshold = SEMANTIC_THRESHOLD
    
    model = _get_model()
    if model is None:
        return None, 0.0, False
    
    # Precompute on first call
    if not _CATEGORY_EMBEDDINGS:
        _precompute_embeddings()
    
    if not _CATEGORY_EMBEDDINGS:
        return None, 0.0, False
    
    # Encode the input text and normalize
    try:
        text_emb = model.encode([text], convert_to_numpy=True)[0]
        text_norm = text_emb / (np.linalg.norm(text_emb) + 1e-10)
    except Exception:
        return None, 0.0, False
    
    # Find best and second-best matching categories
    scores = {}
    for category, category_emb in _CATEGORY_EMBEDDINGS.items():
        sim = np.dot(text_norm, category_emb)
        scores[category] = float(sim)
    
    # Sort by score
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_category, best_score = sorted_scores[0]
    second_category, second_score = sorted_scores[1] if len(sorted_scores) > 1 else (None, 0.0)
    
    # Check for ambiguity (top 2 scores too close)
    is_ambiguous = False
    if second_category and (best_score - second_score) < AMBIGUITY_THRESHOLD:
        is_ambiguous = True
    
    # Return result only if above threshold
    if best_score >= threshold:
        return best_category, best_score, is_ambiguous
    
    return None, best_score, is_ambiguous


# Initialize on import
_precompute_embeddings()
