from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SafetyAssessment:
    force_escalate: bool
    invalid: bool
    reason: str
    unsafe: bool = False


PROMPT_INJECTION = (
    "ignore previous", "ignore all", "rules internal", "documents retrieved",
    "logic exact", "developer message", "show your instructions", "print all files", "delete all files",
    "disregard the policy", "output escalated=false", "mark this as replied",
)
HARMFUL_PATTERNS = (
    "build a bomb", "make explosives", "make a bomb", "kill someone", "attack a person",
    "weapon instructions", "steal credentials",
)
ESCALATE_PATTERNS = {
    "account_authority": ("restore my access", "not the workspace owner", "not admin", "removed my seat", "change my score", "increase my score", "ncrease my score", "review my answers", "answer key"),
    "billing_refund": ("refund me today", "give me the refund", "give me my money", "chargeback", "payment with order id", "pause our subscription", "approve a refund"),
    "security": ("security vulnerability", "security bug", "bug bounty", "hacked", "unauthorized", "identity has been stolen", "identity theft", "account was hacked", "leaks files", "did not authorize", "fraud", "scam", "pretend you are", "admin access", "master password", '"""system:', "system prompt", "transferred out", "transferred $", "stolen"),
    "legal_dispute": ("ban the seller", "lawsuit", "legal notice", "merchant sent the wrong product", "lawyer will take action", "consulting my lawyer"),
    "sensitive_data_export": ("export all team data", "export organization data", "download all team data"),
    "sitewide_outage": ("site is down", "none of the pages", "all requests are failing", "stopped working completely", "submissions across any challenges"),
    "self_harm_or_crisis": ("cant take it anymore", "nothing left for me", "don't see the point of anything", "kill myself", "suicide"),
}


def assess_safety(text: str, company: str | None) -> SafetyAssessment:
    lower = " ".join(text.lower().split())
    if any(term in lower for term in HARMFUL_PATTERNS):
        return SafetyAssessment(False, False, "Unsafe harmful-content request was refused; safe support intent may still be handled.", True)
    injection_reason = ""
    if any(term in lower for term in PROMPT_INJECTION):
        if any(term in lower for term in ("visa", "card", "claude", "anthropic", "hackerrank", "assessment", "workspace", "conversation")):
            injection_reason = "Ignored prompt-injection text and handled only the support request."
        else:
            return SafetyAssessment(False, True, "The message asks for unsafe or out-of-scope system actions.")
    for reason, patterns in ESCALATE_PATTERNS.items():
        if any(pattern in lower for pattern in patterns):
            return SafetyAssessment(True, False, reason.replace("_", " "))
    if not company and any(term in lower for term in ("down", "not working", "failing", "stopped working")):
        return SafetyAssessment(True, False, "Ambiguous outage report without enough product context.")
    return SafetyAssessment(False, False, injection_reason)


def detect_harmful(text: str) -> bool:
    lower = " ".join(text.lower().split())
    return any(term in lower for term in HARMFUL_PATTERNS)
