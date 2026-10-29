from __future__ import annotations

import re


BUG_TERMS = ("down", "not working", "stopped working", "failing", "error", "bug", "blocker", "unable", "can't", "cannot", "crash", "none of")
FEATURE_TERMS = ("feature request", "can you add", "please add", "enhancement", "support for")
INVALID_TERMS = (
    "iron man", "actor", "weather", "recipe", "joke", "delete all files", "system prompt",
    "ignore previous", "photosynthesis", "homework", "resignation email",
)


def classify_request_type(text: str, company: str | None) -> str:
    lower = text.lower()
    if any(x in lower for x in ["suggest", "feature", "would be nice", "can you add"]):
        return "feature_request"

    if any(x in lower for x in ["error", "freezes", "not working", "crash", "bug", "broken", "failing"]):
        return "bug"

    if len(lower.strip()) < 10:
        return "invalid"

    return "product_issue"
