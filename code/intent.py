from __future__ import annotations

import re


INTENT_MARKERS = (
    " and ",
    " also ",
    " plus ",
    " as well ",
    " another issue",
    " multiple issues",
    " first ",
    " second ",
)
INTENT_PRIORITY = {
    "security": 0,
    "fraud": 1,
    "account_compromise": 2,
    "payment_issue": 3,
    "blocking_issue": 4,
    "account_change": 5,
    "feature_request": 6,
    "out_of_scope": 7,
    "general_support": 8,
}


def detect_multi_intent(text: str) -> tuple[bool, str]:
    lower = " ".join(text.lower().split())
    marker_count = sum(1 for marker in INTENT_MARKERS if marker in lower)
    question_count = lower.count("?")
    action_verbs = len(re.findall(r"\b(refund|restore|delete|remove|change|update|reschedule|dispute|fix|help|confirm)\b", lower))
    multi = question_count > 1 or marker_count >= 2 or action_verbs >= 3
    if not multi:
        return False, ""
    primary = primary_intent(lower)
    return True, f"Multiple intents detected; primary intent prioritized as {primary.replace('_', ' ')}."


def primary_intent(text: str) -> str:
    lower = " ".join(text.lower().split())
    intents: set[str] = set()
    if any(term in lower for term in ("security vulnerability", "bug bounty", "data breach", "leaks files")):
        intents.add("security")
    if any(term in lower for term in ("unauthorized", "fraud", "scam", "otp", "identity theft", "fake item")):
        intents.add("fraud")
    if any(term in lower for term in ("hacked", "account was hacked", "password changed", "restore access")):
        intents.add("account_compromise")
    if any(term in lower for term in ("payment", "refund", "charge", "money left", "billing")):
        intents.add("payment_issue")
    if any(term in lower for term in ("down", "not working", "failing", "blocked", "unable", "error")):
        intents.add("blocking_issue")
    if any(term in lower for term in ("change my email", "delete my account", "delete my old account", "remove", "export")):
        intents.add("account_change")
    if any(term in lower for term in ("feature request", "can you add", "extend it")):
        intents.add("feature_request")
    if any(term in lower for term in ("movie", "recipe", "weather", "photosynthesis", "resignation email")):
        intents.add("out_of_scope")
    if not intents:
        return "general_support"
    return min(intents, key=lambda intent: INTENT_PRIORITY[intent])
