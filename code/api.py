"""FastAPI service for the Multi-Domain Support Triage Agent powered by LangGraph."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

from triage_graph import LangGraphSupportAgent

app = FastAPI(
    title="Multi-Domain Support Triage API (LangGraph)",
    description="Intelligent support ticket triage service using LangGraph state graphs and LangChain retrieval.",
    version="1.0.0",
)

# Initialize agent singleton
DATA_DIR = ROOT / "data"
agent: Optional[LangGraphSupportAgent] = None


@app.on_event("startup")
def startup_event():
    global agent
    agent = LangGraphSupportAgent(DATA_DIR)


class TicketRequest(BaseModel):
    issue: str = Field(..., description="Customer support issue description")
    subject: Optional[str] = Field(default="", description="Subject line of ticket")
    company: Optional[str] = Field(default="", description="Target company or service (HackerRank, Claude, Visa)")


class TicketResponse(BaseModel):
    status: str = Field(..., description="'replied' or 'escalated'")
    decision: str = Field(..., description="'answer' or 'escalate'")
    product_area: str = Field(..., description="Identified product domain/subsystem")
    request_type: str = Field(..., description="Request category (product_issue, risk, invalid)")
    response: str = Field(..., description="Generated answer or escalation guidance")
    justification: str = Field(..., description="Explainable reasoning and confidence metrics")


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "support-triage-langgraph",
        "version": "1.0.0",
    }


@app.get("/graph/mermaid")
def get_mermaid_diagram():
    global agent
    if agent is None:
        agent = LangGraphSupportAgent(DATA_DIR)
    return {"mermaid": agent.get_mermaid_diagram()}


@app.post("/triage", response_model=TicketResponse)
def triage_ticket(ticket: TicketRequest):
    global agent
    if agent is None:
        agent = LangGraphSupportAgent(DATA_DIR)
    
    row = {
        "issue": ticket.issue,
        "subject": ticket.subject or "",
        "company": ticket.company or "",
    }
    try:
        result = agent.triage(row)
        decision = "answer" if result["status"] == "replied" and result.get("request_type") != "invalid" else "escalate"
        return TicketResponse(
            status=result["status"],
            decision=decision,
            product_area=result["product_area"],
            request_type=result["request_type"],
            response=result["response"],
            justification=result["justification"],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Triage error: {exc}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
