from __future__ import annotations

from dataclasses import dataclass

from retriever import Hit
from safety import SafetyAssessment, assess_safety


MIN_RETRIEVAL_SCORE = 2.0
MIN_COMBINED_CONFIDENCE = 0.46


@dataclass(frozen=True)
class EscalationDecision:
    escalate: bool
    invalid: bool
    reason: str
    confidence: float


def confidence_score(route_confidence: float, request_type: str, top_hit: Hit | None, multi_intent: bool) -> float:
    retrieval = 0.0
    if top_hit:
        retrieval = min(1.0, (top_hit.score / 8.0) + min(top_hit.overlap, 8) * 0.025)
    type_confidence = 0.62 if request_type == "product_issue" else 0.72
    confidence = (0.50 * retrieval) + (0.30 * route_confidence) + (0.20 * type_confidence)
    if multi_intent:
        confidence -= 0.08
    return max(0.0, min(1.0, confidence))


def escalation_guard(
    text: str,
    domain: str | None,
    request_type: str,
    top_hit: Hit | None,
    route_confidence: float = 1.0,
    multi_intent: bool = False,
) -> EscalationDecision:
    combined = confidence_score(route_confidence, request_type, top_hit, multi_intent)
    safety: SafetyAssessment = assess_safety(text, domain)
    if safety.unsafe:
        return EscalationDecision(False, False, safety.reason, combined)
    if safety.invalid or request_type == "invalid":
        return EscalationDecision(False, True, safety.reason or "Ticket is outside supported scope.", combined)
    if safety.force_escalate:
        return EscalationDecision(True, False, safety.reason, combined)
    if not domain:
        return EscalationDecision(False, True, "No supported product domain could be inferred from the ticket.", combined)
    if not top_hit or top_hit.score < MIN_RETRIEVAL_SCORE or combined < MIN_COMBINED_CONFIDENCE:
        return EscalationDecision(True, False, f"Combined confidence was too low to answer safely ({combined:.2f}).", combined)
    return EscalationDecision(False, False, "", combined)
