"""State definition for the LangGraph-based Support Triage Agent."""
from __future__ import annotations

from typing import Any, TypedDict


class TriageState(TypedDict, total=False):
    """Unified state object passed between nodes in the LangGraph support triage graph."""
    # Input ticket fields
    ticket_text: str
    subject: str
    company: str
    
    # Classification & Routing
    domain: str
    intent: str
    all_intents: list[str]
    
    # Safety & Resilience
    safety_flag: bool
    safety_reason: str
    
    # Retrieval & Evidence
    retrieved_chunks: list[Any]
    confidence: float
    confidence_breakdown: dict[str, float]
    
    # Validation & Decisions
    grounded: bool
    decision: str  # "answer" or "escalate"
    
    # Final Output Fields
    status: str  # "replied" or "escalated"
    product_area: str
    request_type: str
    response: str
    justification: str
