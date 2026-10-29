# code/ - Core Implementation

This directory contains the primary logic for the Multi-Domain Support Triage Agent, supporting both the compiled **LangGraph StateGraph** pipeline and the legacy baseline pipeline.

## 🚀 Execution
```bash
# 1. Install dependencies from repo root
pip install -r ../requirements.txt

# 2. Run the LangGraph agent
python main.py --input ../support_issues/support_issues.csv --output ../support_issues/output.csv

# 3. Run legacy pipeline
python main.py --pipeline legacy --input ../support_issues/support_issues.csv --output ../support_issues/legacy_output.csv
```

## 📂 Module Map

### LangGraph Engine
- **`graph_state.py`**: Typed state dictionary (`TriageState`) tracking ticket context, routing flags, retrieval chunks, and decisions.
- **`langchain_retriever.py`**: Custom LangChain `BaseRetriever` wrapping the hybrid BM25 / vector search engine.
- **`graph_nodes.py`**: Discrete node implementations (`safety`, `resilience`, `intent`, `retrieve`, `ground`, `answer`, `escalate`).
- **`triage_graph.py`**: StateGraph builder, conditional edge router, and `LangGraphSupportAgent` interface.
- **`api.py`**: FastAPI service exposing `/health`, `/triage`, and `/graph/mermaid` endpoints.

### Core Utilities & Legacy Pipeline
- **`main.py`**: Entry point conforming to the project submission contract.
- **`agent.py`**: Legacy `SupportAgent` intelligence controller.
- **`corpus_loader.py`**: Markdown corpus parsing and metadata extraction.
- **`retriever.py`**: Hybrid retrieval engine (lexical overlap, BM25, and semantic search).
- **`safety.py`**: Prompt injection and high-risk safety assessment.
- **`sanitizer.py`**: PII scrubbing and ticket normalization.
- **`gates.py`**: Retrieval quality, domain hard filters, and vagueness heuristics.
- **`decision_engine.py`**: Controlled decision reasoning layer.
- **`validator.py`**: Output label validation ensuring compliance with challenge constraints.

## ⚙️ Configuration
Tuning parameters in `agent.py` & `graph_nodes.py`:
- `CONF_HIGH` (0.65): Confidence threshold required for automated grounded resolution.
- `CONF_LOW` (0.40): Threshold below which tickets are routed directly to safe human escalation.
