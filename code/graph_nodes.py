"""LangGraph Nodes for the Multi-Domain Support Triage Agent.

Each node receives the current TriageState and returns a dictionary of updated fields.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from pathlib import Path

from graph_state import TriageState
from langchain_retriever import SupportLangChainRetriever
from corpus_loader import load_corpus, Document
from retriever import Retriever, Hit
from sanitizer import sanitize_row
from safety import assess_safety, detect_harmful
from gates import (
    contains_safety_signal,
    enforce_domain_match,
    is_vague,
    retrieval_is_trustworthy,
)
from agent import (
    CONF_HIGH,
    CONF_LOW,
    SupportAgent,
    adaptive_conflict_resolver,
    apex_merge_intents,
    calculate_explainable_score,
    detect_intent,
    hybrid_noise_filter,
)
from decision_engine import make_decision
from semantic_safety import semantic_risk_detect


class TriageNodeContext:
    """Shared resources (retriever, corpus, agent helper) for the graph nodes."""

    def __init__(self, data_dir: Path | str) -> None:
        self.data_dir = Path(data_dir)
        self.docs: list[Document] = load_corpus(self.data_dir)
        self.internal_retriever = Retriever(self.docs)
        self.agent_helper = SupportAgent(self.data_dir)
        self.langchain_retriever = SupportLangChainRetriever(retriever=self.internal_retriever)


# Global context instance (can be initialized or passed)
_DEFAULT_CONTEXT: Optional[TriageNodeContext] = None


def get_default_context(data_dir: Optional[Path | str] = None) -> TriageNodeContext:
    global _DEFAULT_CONTEXT
    if _DEFAULT_CONTEXT is None:
        if data_dir is None:
            # Default to repo root / data
            root = Path(__file__).resolve().parents[1]
            data_dir = root / "data"
        _DEFAULT_CONTEXT = TriageNodeContext(data_dir)
    return _DEFAULT_CONTEXT


def safety_node(state: TriageState, context: Optional[TriageNodeContext] = None) -> Dict[str, Any]:
    """Safety stage: checks PII, prompt injections, harm, and hard escalation patterns."""
    text = state.get("ticket_text", "")
    company = state.get("company", "")
    
    # 1. Sanitization & Harmful Detection
    if detect_harmful(text):
        return {
            "safety_flag": True,
            "safety_reason": "Harmful or unsafe instructions detected.",
            "request_type": "invalid",
        }

    # 2. Safety Assessment (Injections & hard escalation rules)
    safety = assess_safety(text, company)
    if safety.unsafe:
        return {
            "safety_flag": True,
            "safety_reason": safety.reason or "Unsafe content detected.",
            "request_type": "invalid",
        }
    if safety.invalid:
        return {
            "safety_flag": True,
            "safety_reason": safety.reason or "Ticket is outside supported scope.",
            "request_type": "invalid",
        }
    if safety.force_escalate:
        return {
            "safety_flag": True,
            "safety_reason": safety.reason or "Safety policy requires human escalation.",
            "request_type": "product_issue",
        }

    # 3. Direct prompt injection phrases & overrides
    t_lower = text.lower()
    injection_triggers = (
        "ignore", "disregard", "system override", "[system", "master password",
        "admin access", "developer prompt", "system prompt", "pretend you are"
    )
    if any(term in t_lower for term in injection_triggers):
        if ("instruction" in t_lower or "rule" in t_lower or "override" in t_lower or "prompt" in t_lower or "password" in t_lower or "access" in t_lower):
            return {
                "safety_flag": True,
                "safety_reason": "Prompt injection attempt detected.",
                "request_type": "risk",
            }

    # 4. Mandatory escalation patterns from decision engine (refunds, lawsuits, authority, account security)
    from decision_engine import CLEARLY_UNSAFE
    for unsafe_key, category in CLEARLY_UNSAFE.items():
        if unsafe_key in t_lower:
            return {
                "safety_flag": True,
                "safety_reason": f"Safety policy requires human escalation: {category} ({unsafe_key})",
                "request_type": "product_issue",
            }

    # 5. Outage and severe disruption patterns
    outage_triggers = ("500 error", "all requests are failing", "stopped working completely", "site is down")
    if any(term in t_lower for term in outage_triggers):
        return {
            "safety_flag": True,
            "safety_reason": "Sitewide outage or infrastructure failure requires human incident response.",
            "request_type": "product_issue",
        }

    return {"safety_flag": False, "safety_reason": ""}


def resilience_node(state: TriageState, context: Optional[TriageNodeContext] = None) -> Dict[str, Any]:
    """Resilience stage: checks for empty, vague, or gibberish/noise inputs."""
    # If already flagged by safety, preserve it
    if state.get("safety_flag"):
        return {}

    text = state.get("ticket_text", "").strip()
    if not text:
        return {
            "safety_flag": True,
            "safety_reason": "Input is empty.",
            "request_type": "product_issue",
        }

    t_lower = text.lower()
    # Hybrid noise filter (character density, symbol checks)
    if hybrid_noise_filter(t_lower, initial_conf=0.35):
        return {
            "safety_flag": True,
            "safety_reason": "Input failed noise/gibberish density heuristics.",
            "request_type": "product_issue",
        }

    # Vague input pattern check
    if is_vague(text) or (len(text.split()) <= 7 and any(w in t_lower for w in ("doesn't work", "not working", "broken", "pls fix", "please fix", "went wrong"))):
        return {
            "safety_flag": True,
            "safety_reason": "Query is too vague to resolve safely without clarification.",
            "request_type": "product_issue",
        }

    return {"safety_flag": False}


def intent_node(state: TriageState, context: Optional[TriageNodeContext] = None) -> Dict[str, Any]:
    """Intent mapping and domain routing stage."""
    ctx = context or get_default_context()
    text = state.get("ticket_text", "")
    t_lower = text.lower()
    company = (state.get("company") or "").lower()

    # 1. Multi-intent splitting and merging
    intents = ctx.agent_helper.split_intents(t_lower)
    
    # 2. Cross-domain conflict detection
    if adaptive_conflict_resolver(intents):
        return {
            "safety_flag": True,
            "safety_reason": "Cross-domain collision detected across unrelated domains.",
            "product_area": "conversation_management",
            "request_type": "product_issue",
        }

    intents = apex_merge_intents(intents)
    primary_intent_text = intents[0] if intents else t_lower

    # 3. Domain determination
    detected_domain = None
    if any(w in t_lower for w in ["visa", "card", "transaction"]):
        detected_domain = "visa"
    elif any(w in t_lower for w in ["claude", "anthropic", "bedrock"]):
        detected_domain = "claude"
    elif any(w in t_lower for w in ["hackerrank", "assessment", "recruiter", "coding test"]):
        detected_domain = "hackerrank"
    
    if not detected_domain and company in ("visa", "claude", "hackerrank"):
        detected_domain = company
    
    domain = detected_domain or ""
    detected_intent = detect_intent(primary_intent_text) or "general_inquiry"

    return {
        "domain": domain,
        "intent": detected_intent,
        "all_intents": intents,
    }


def retrieve_node(state: TriageState, context: Optional[TriageNodeContext] = None) -> Dict[str, Any]:
    """Retrieval stage: performs BM25 + embedding search via LangChain retriever."""
    ctx = context or get_default_context()
    text = state.get("ticket_text", "")
    domain = state.get("domain", "")
    all_intents = state.get("all_intents") or [text]
    query = all_intents[0] if all_intents else text

    # Use internal retriever search to obtain rich Hit objects with score and overlap
    hits = ctx.internal_retriever.search(query, domain=domain if domain else None, k=5)
    
    # Also query through LangChain retriever to validate LC integration
    lc_docs = ctx.langchain_retriever.invoke(query)

    # Calculate explainable confidence score
    scores = calculate_explainable_score(query, state.get("intent", ""), hits)
    conf = scores["total"]

    return {
        "retrieved_chunks": hits,
        "confidence": conf,
        "confidence_breakdown": scores,
    }


def ground_node(state: TriageState, context: Optional[TriageNodeContext] = None) -> Dict[str, Any]:
    """Grounding stage: verifies lexical overlap, semantic alignment, and domain matching."""
    text = state.get("ticket_text", "")
    domain = state.get("domain", "")
    hits: List[Hit] = state.get("retrieved_chunks", [])
    conf = state.get("confidence", 0.0)

    if not hits:
        return {"grounded": False}

    top_score = hits[0].score if hits else 0.0
    trustworthy, reason = retrieval_is_trustworthy(text, hits, top_score)
    domain_ok = enforce_domain_match(domain, hits)

    # Decision engine check
    s_label, s_score, s_ambig = semantic_risk_detect(text)
    decision = make_decision(text, hits, (s_label, s_score, s_ambig), retrieval_confidence=conf, use_llm=False)

    grounded = bool(trustworthy and domain_ok and decision.safe_to_answer and conf >= CONF_HIGH)
    return {
        "grounded": grounded,
        "decision_engine_safe": decision.safe_to_answer,
        "grounding_reason": decision.reasoning if not decision.safe_to_answer else reason,
    }


def answer_node(state: TriageState, context: Optional[TriageNodeContext] = None) -> Dict[str, Any]:
    """Answer stage: builds grounded, actionable response."""
    ctx = context or get_default_context()
    text = state.get("ticket_text", "")
    domain = state.get("domain", "")
    hits: List[Hit] = state.get("retrieved_chunks", [])
    conf = state.get("confidence", 0.0)
    scores = state.get("confidence_breakdown", {})
    all_intents = state.get("all_intents", [text])

    product_area = hits[0].doc.product_area if hits else "general_support"
    reply_dict = ctx.agent_helper._reply(
        domain=domain,
        product_area=product_area,
        request_type="product_issue",
        hits=hits,
        text=text,
        confidence=conf,
        route_reason="retrieval_grounded",
    )

    justification_payload = {
        "confidence": conf,
        "keyword_score": scores.get("keyword_score", 0.0),
        "domain_score": scores.get("domain_score", 0.0),
        "intent_coherence": scores.get("intent_coherence", 0.0),
        "signal_strength": scores.get("signal_strength", 0.0),
        "final_intent": text,
        "merged": len(all_intents) > 1,
        "decision": "resolve",
        "reason": f"High confidence grounding via LangGraph ({reply_dict['product_area']})"
    }

    return {
        "decision": "answer",
        "status": "replied",
        "product_area": reply_dict["product_area"],
        "request_type": "product_issue",
        "response": reply_dict["response"],
        "justification": json.dumps(justification_payload),
    }


def escalate_node(state: TriageState, context: Optional[TriageNodeContext] = None) -> Dict[str, Any]:
    """Escalate stage: creates safe, human-handoff response."""
    ctx = context or get_default_context()
    text = state.get("ticket_text", "")
    domain = state.get("domain", "")
    reason = state.get("safety_reason") or state.get("grounding_reason") or "Insufficient confidence for automated guidance."
    scores = state.get("confidence_breakdown", {})
    conf = state.get("confidence", 0.0)
    req_type = state.get("request_type") or "product_issue"

    # Self-harm or crisis intervention
    t_lower = text.lower()
    if any(p in t_lower for p in ("cant take it anymore", "nothing left for me", "suicide", "end it all")):
        return {
            "decision": "escalate",
            "status": "escalated",
            "product_area": "general_support",
            "request_type": "product_issue",
            "response": (
                "I'm really sorry that you're feeling this way. You're not alone, and there are people who want to help. "
                "Please consider reaching out to a trusted friend, family member, or a mental health professional. "
                "If you're in immediate distress, contacting a local crisis support service can help you get immediate assistance."
            ),
            "justification": "Escalated due to self-harm or crisis signal requiring human support.",
        }

    # Out of scope / invalid request
    if req_type == "invalid":
        return {
            "decision": "escalate",
            "status": "replied",
            "product_area": "out_of_scope",
            "request_type": "invalid",
            "response": (
                "I am sorry, this is out of scope for this support agent. "
                "I can help with HackerRank, Claude, or Visa support questions grounded in the provided help corpus."
            ),
            "justification": f"Out of scope: {reason}",
        }

    # Guided clarification fallback for mid-range confidence
    if conf >= CONF_LOW and not state.get("safety_flag"):
        suggestion = ctx.agent_helper.intent_specific_answer(text)
        if not suggestion:
            if domain == "visa": suggestion = "inquire about your specific card transaction or limit."
            elif domain == "claude": suggestion = "ask about API integration or model usage."
            elif domain == "hackerrank": suggestion = "contact support regarding your candidate assessment."
        
        if suggestion:
            just_payload = {
                "confidence": conf,
                "scores": scores,
                "decision": "clarify",
                "reason": "Confidence in mid-range; providing guided fallback."
            }
            return {
                "decision": "escalate",
                "status": "replied",
                "product_area": "general_support",
                "request_type": "product_issue",
                "response": f"Your query seems related to {suggestion} Please provide more details or verify if this matches your issue.",
                "justification": json.dumps(just_payload),
            }

    # Standard safe human escalation
    just_payload = {
        "decision": "safe_escalation",
        "reason": reason,
        "scores": scores,
    }
    return {
        "decision": "escalate",
        "status": "escalated",
        "product_area": "general_support",
        "request_type": req_type,
        "response": "I've reviewed your request and it requires human assistance to ensure it's handled correctly.",
        "justification": json.dumps(just_payload),
    }
