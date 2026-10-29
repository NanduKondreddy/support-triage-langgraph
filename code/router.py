from __future__ import annotations


DOMAIN_TERMS = {
    "hackerrank": ("hackerrank", "assessment", "test", "candidate", "interview", "mock interview", "certificate", "submissions", "apply tab", "recruiter"),
    "claude": ("claude", "anthropic", "workspace", "team plan", "bedrock", "lti", "crawl", "model", "conversation"),
    "visa": ("visa", "card", "merchant", "charge", "traveller", "traveler", "cheque", "cash", "blocked", "minimum spend", "identity theft"),
}


def route_company(company: str | None, text: str) -> tuple[str | None, float, str]:
    raw = (company or "").strip().lower()
    if raw in ("hackerrank", "claude", "visa"):
        return raw, 1.0, "explicit company column"
    lower = text.lower()
    scores = {domain: sum(1 for term in terms if term in lower) for domain, terms in DOMAIN_TERMS.items()}
    best, score = max(scores.items(), key=lambda item: item[1])
    if not score:
        return None, 0.0, "no domain keywords"
    sorted_scores = sorted(scores.values(), reverse=True)
    margin = score - (sorted_scores[1] if len(sorted_scores) > 1 else 0)
    confidence = min(0.95, 0.45 + (0.2 * score) + (0.15 * margin))
    if confidence < 0.58:
        return None, confidence, "low-confidence domain keywords; searched all corpora"
    return best, confidence, f"keyword route score={score}, margin={margin}"


def normalize_company(company: str | None, text: str) -> str | None:
    return route_company(company, text)[0]


def area_override(text: str, domain: str | None, retrieved_area: str) -> str:
    lower = text.lower()
    if not retrieved_area:
        if "api" in lower: return "api_usage"
        if "limit" in lower: return "usage_limits"
        if "password" in lower: return "account_security"
        if "payment" in lower: return "payment_processing"
        if "loading" in lower: return "performance"
        if "merchant" in lower: return "payments_acceptance"
        if "test" in lower: return "screen"
        return "general_support"
    if not domain:
        return retrieved_area or "conversation_management"
    if domain == "hackerrank":
        if any(term in lower for term in ("infosec", "security process", "security forms", "filling in the forms")):
            return "general_support"
        if any(term in lower for term in ("mock interview", "resume", "apply tab", "certificate", "google login", "delete my account")):
            return "community"
        if any(term in lower for term in ("subscription", "payment", "refund", "order id", "billing")):
            return "billing"
        if any(term in lower for term in ("remove an interviewer", "employee has left", "remove them", "user")):
            return "settings"
        if any(term in lower for term in ("interview", "lobby", "inactivity")):
            return "interviews"
        return "screen"
    if domain == "claude":
        if any(term in lower for term in ("crawl", "privacy", "private info", "delete", "data")):
            return "privacy"
        if any(term in lower for term in ("team workspace", "seat", "admin", "remove", "organization")):
            return "team_enterprise"
        if any(term in lower for term in ("bedrock", "api", "requests are failing", "console")):
            return "api_console"
        if any(term in lower for term in ("lti", "professor", "students", "education")):
            return "claude_for_education"
        if "security vulnerability" in lower or "bug bounty" in lower:
            return "safeguards"
        return retrieved_area or "claude"
    if domain == "visa":
        if any(term in lower for term in ("travel", "traveller", "traveler", "blocked", "voyage", "unblock", "ne marche")):
            return "travel_support"
        if "lost or stolen visa card" in lower or "stolen visa card" in lower:
            return "general_support"
        if any(term in lower for term in ("identity", "fraud", "stolen", "otp", "suspicious call", "scam")):
            return "fraud_protection"
        if "contactless" in lower:
            return "card_features"
        if any(term in lower for term in ("merchant", "charge", "minimum", "dispute")):
            return "dispute_resolution"
        if "cash" in lower:
            return "consumer"
        return "general_support"
    return retrieved_area or "general_support"
