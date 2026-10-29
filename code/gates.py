import re

# FIX 1 — Expanded Safety Detection
ACCOUNT_COMPROMISE_PATTERNS = [
    "logged into my account",
    "logged in to my account",
    "someone accessed",
    "someone has access",
    "account compromised",
    "account was hacked",
    "account hacked",
    "someone using my account",
    "unauthorized login",
    "unauthorized access",
    "ex-husband", "ex-wife", "ex-partner",  # domestic/relational threat
    "stalker", "stalking",
    "didn't authorize", "did not authorize",
    "not me who logged",
    "session i don't recognize",
    "device i don't recognize",
]

SOFT_COMPROMISE_PATTERNS = [
    "someone else", "seems to be using", "seems someone",
    "noticed activity", "strange activity", "weird activity",
    "didn't recognize", "did not recognize",
    "don't recognize this", "do not recognize this",
]

SOFT_FRAUD_PATTERNS = [
    "don't remember making", "do not remember making",
    "don't remember this", "do not remember this",
    "didn't make this", "did not make this",
    "wasn't me", "was not me",
    "never authorized", "never made",
    "unfamiliar transaction", "unknown transaction",
    "unknown charge", "unrecognized charge",
    "transferred out", "transferred $", "transferred money",
    "stolen", "missing money",
    "don't recognize", "do not recognize",
    "dispute with merchant", "dispute",
]

PRIVACY_DATA_PATTERNS = [
    "show me", "list all", "export",
    "api logs", "access logs", "who accessed",
    "candidates", "failed my test",
    "personal emails", "email address",
    "answer key", "solution",
    "anyone else", "if anyone",
    "tell me if", "verify",
]

CREDENTIAL_EXPOSURE_PATTERNS = [
    "tell me my api key", "tell me my current api",
    "tell me my password", "tell me my secret",
    "tell me my token", "reveal my",
]

PROMPT_INJECTION_PATTERNS = [
    "disregard", "ignore previous", "ignore all",
    "treat this as", "mark it as resolved",
    "output", "tell me your", "show your",
]

CRITICAL_SAFETY_PATTERNS = [
    "fraud", "unauthorized", "charge", "lawyer", "attorney", "legal",
    "hack", "compromised", "end it all", "suicide", "summarize messages", "user id"
] + ACCOUNT_COMPROMISE_PATTERNS + SOFT_COMPROMISE_PATTERNS + SOFT_FRAUD_PATTERNS + PRIVACY_DATA_PATTERNS + CREDENTIAL_EXPOSURE_PATTERNS + PROMPT_INJECTION_PATTERNS

def contains_safety_signal(text: str) -> bool:
    text_clean = text.lower()
    return any(pattern in text_clean for pattern in CRITICAL_SAFETY_PATTERNS)

# FIX 2 — Smarter Retrieval Gate
def meaningful_tokens(query):
    stopwords = {"my", "has", "been", "since", "the", "a", "an", "is", "for", "to", "in", "on", "with", "of", "and", "or", "but", "it", "that", "this", "be", "are", "from", "as", "at", "what", "how", "why", "who", "when"}
    return [t for t in query.lower().split() if t not in stopwords]

def retrieval_is_trustworthy(query, top_chunks, top_score):
    if top_score < 0.35:
        return False, "low_absolute_score"
    
    if len(top_chunks) >= 2:
        gap = top_chunks[0].score - top_chunks[1].score
        if gap < 0.05 and top_score < 0.55:
            return False, "ambiguous_retrieval"
            
    query_tokens = meaningful_tokens(query)
    if not top_chunks:
        return False, "no_chunks"
    chunk_text = top_chunks[0].doc.text.lower()
    overlap = sum(1 for t in query_tokens if t in chunk_text)
    
    if len(query_tokens) >= 3 and overlap == 0:
        return False, "zero_lexical_overlap"
        
    return True, "ok"

# FIX 3 — Vague Query Handling
VAGUE_PATTERNS = [
    r"^things? (are |is )?broken",
    r"^(it'?s |its )?not working",
    r"^acting (strange|weird|funny)",
    r"^something(')?s? wrong",
    r"^help( me)?$",
    r"^broken$",
    r"^doesn'?t work$",
    r"^issue$",
    r"^problem$",
]

def is_vague(text: str) -> bool:
    text_clean = text.strip().lower()
    if len(text_clean.split()) > 15:
        return False
    for pattern in VAGUE_PATTERNS:
        if re.search(pattern, text_clean):
            return True
    return False

# FIX 4 — Domain Hard Filter
def enforce_domain_match(detected_domain, top_chunks):
    if detected_domain in (None, "unknown", "out_of_scope", ""):
        return True
    
    matching = sum(1 for c in top_chunks[:3] if c.doc.domain == detected_domain)
    return matching >= 2

# FIX 6 — Out-of-Scope Handling
def handle_out_of_scope(query, max_score_across_domains):
    if max_score_across_domains < 0.30:
        if contains_safety_signal(query):
            return {
                "status": "escalated",
                "request_type": "product_issue",
                "product_area": "fraud_protection",
                "response": "Escalate to a human support specialist. This case needs account-specific review or carries risk that should not be resolved by an automated agent.",
                "justification": "Out-of-scope but contains safety signal."
            }
        
        return {
            "status": "replied",
            "request_type": "invalid",
            "product_area": "out_of_scope",
            "response": (
                "This request appears to be outside the scope of our "
                "supported domains (HackerRank, Claude, Visa). "
                "Please contact the appropriate provider directly."
            ),
            "justification": f"Out-of-scope: max retrieval score {max_score_across_domains:.2f} across all domains."
        }
    return None
