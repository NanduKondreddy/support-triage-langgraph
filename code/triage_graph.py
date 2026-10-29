"""LangGraph Multi-Domain Support Triage Graph Definition and Pipeline.

Wires discrete nodes into an executable StateGraph with conditional routing:
- Safety & Resilience checks (routes straight to escalate if unsafe/invalid/noise)
- Intent mapping & domain classification
- Hybrid BM25 / vector retrieval via LangChain
- Lexical and semantic grounding check
- High-confidence grounded answering vs safe human escalation
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from langgraph.graph import StateGraph, START, END

from graph_state import TriageState
from graph_nodes import (
    TriageNodeContext,
    get_default_context,
    safety_node,
    resilience_node,
    intent_node,
    retrieve_node,
    ground_node,
    answer_node,
    escalate_node,
)
from sanitizer import sanitize_row
from validator import validate_output


def route_after_resilience(state: TriageState) -> str:
    """Conditional routing after resilience check."""
    if state.get("safety_flag", False):
        return "escalate"
    return "intent"


def route_after_retrieve(state: TriageState) -> str:
    """Conditional routing after retrieval stage."""
    conf = state.get("confidence", 0.0)
    # Below safe threshold, route directly to safe escalation
    if conf < 0.40:
        return "escalate"
    return "ground"


def route_after_ground(state: TriageState) -> str:
    """Conditional routing after grounding verification."""
    if state.get("grounded", False):
        return "answer"
    return "escalate"


def build_triage_graph(context: Optional[TriageNodeContext] = None):
    """Builds and compiles the LangGraph StateGraph for support triage."""
    ctx = context or get_default_context()

    # Bind node functions with context
    def _safety(state: TriageState) -> Dict[str, Any]:
        return safety_node(state, ctx)

    def _resilience(state: TriageState) -> Dict[str, Any]:
        return resilience_node(state, ctx)

    def _intent(state: TriageState) -> Dict[str, Any]:
        return intent_node(state, ctx)

    def _retrieve(state: TriageState) -> Dict[str, Any]:
        return retrieve_node(state, ctx)

    def _ground(state: TriageState) -> Dict[str, Any]:
        return ground_node(state, ctx)

    def _answer(state: TriageState) -> Dict[str, Any]:
        return answer_node(state, ctx)

    def _escalate(state: TriageState) -> Dict[str, Any]:
        return escalate_node(state, ctx)

    builder = StateGraph(TriageState)

    # Register nodes
    builder.add_node("safety", _safety)
    builder.add_node("resilience", _resilience)
    builder.add_node("intent", _intent)
    builder.add_node("retrieve", _retrieve)
    builder.add_node("ground", _ground)
    builder.add_node("answer", _answer)
    builder.add_node("escalate", _escalate)

    # Wire edges: START -> safety -> resilience
    builder.add_edge(START, "safety")
    builder.add_edge("safety", "resilience")

    # Conditional edge: resilience -> (escalate | intent)
    builder.add_conditional_edges(
        "resilience",
        route_after_resilience,
        {"escalate": "escalate", "intent": "intent"},
    )

    # Edge: intent -> retrieve
    builder.add_edge("intent", "retrieve")

    # Conditional edge: retrieve -> (escalate | ground)
    builder.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {"escalate": "escalate", "ground": "ground"},
    )

    # Conditional edge: ground -> (answer | escalate)
    builder.add_conditional_edges(
        "ground",
        route_after_ground,
        {"answer": "answer", "escalate": "escalate"},
    )

    # Terminal edges
    builder.add_edge("answer", END)
    builder.add_edge("escalate", END)

    return builder.compile()


class LangGraphSupportAgent:
    """Production support triage agent powered by LangGraph."""

    def __init__(self, data_dir: Path | str) -> None:
        self.context = TriageNodeContext(data_dir)
        self.graph = build_triage_graph(self.context)

    def triage(self, row: dict[str, str]) -> dict[str, str]:
        """Triage an incoming support ticket row through the LangGraph StateGraph."""
        ticket = sanitize_row(row)
        initial_state: TriageState = {
            "ticket_text": ticket.text,
            "subject": row.get("Subject") or row.get("subject") or "",
            "company": ticket.company or "",
            "domain": (ticket.company or "").lower(),
            "safety_flag": False,
            "safety_reason": "",
            "intent": "",
            "all_intents": [],
            "retrieved_chunks": [],
            "confidence": 0.0,
            "confidence_breakdown": {},
            "grounded": False,
            "decision": "escalate",
            "status": "escalated",
            "product_area": "general_support",
            "request_type": "product_issue",
            "response": "",
            "justification": "",
        }

        final_state = self.graph.invoke(initial_state)

        result = {
            "status": final_state.get("status", "escalated"),
            "product_area": final_state.get("product_area", "general_support"),
            "response": final_state.get("response", "Escalate to a human support specialist."),
            "justification": final_state.get("justification", "Escalated through LangGraph workflow."),
            "request_type": final_state.get("request_type", "product_issue"),
        }
        return validate_output(result)

    def get_mermaid_diagram(self) -> str:
        """Returns the Mermaid diagram representation of the compiled state graph."""
        return self.graph.get_graph().draw_mermaid()
