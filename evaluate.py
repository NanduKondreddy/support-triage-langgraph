"""Evaluation script for the LangGraph Support Triage Agent.

Evaluates performance on a benchmark set of tickets:
- Retrieval hit rate
- Escalation precision, recall, and F1-score
- Answer grounding rate
- Overall decision accuracy
"""
from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))

from triage_graph import LangGraphSupportAgent
from corpus_loader import Document


def run_evaluation(dataset_path: Path, data_dir: Path) -> dict[str, float]:
    print(f"Loading evaluation dataset from: {dataset_path}")
    with dataset_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    print(f"Initializing LangGraphSupportAgent with corpus at: {data_dir}")
    agent = LangGraphSupportAgent(data_dir)

    total_tickets = len(rows)
    retrieval_hits = 0
    grounded_answers = 0
    total_answers = 0

    # Confusion matrix for Escalation
    # Positive class = "escalate"
    tp = 0  # Expected escalate, got escalated
    fp = 0  # Expected answer, got escalated
    fn = 0  # Expected escalate, got answered
    tn = 0  # Expected answer, got answered

    correct_decisions = 0

    print(f"\nEvaluating {total_tickets} benchmark tickets...")
    for idx, row in enumerate(rows, start=1):
        actual = agent.triage(row)
        expected_decision = row.get("expected_decision", "escalate").strip().lower()
        expected_status = row.get("expected_status", "escalated").strip().lower()

        actual_status = actual["status"].strip().lower()
        actual_decision = "answer" if actual_status == "replied" and actual.get("request_type") != "invalid" else "escalate"

        # Check retrieval hit: if domain matches or relevant doc retrieved
        company = (row.get("company") or "").lower()
        if company and company != "none":
            # Search context for relevant domain docs
            hits = agent.context.internal_retriever.search(row["issue"], domain=company, k=3)
            if hits and any(h.doc.domain == company for h in hits):
                retrieval_hits += 1
        else:
            # Query without domain constraint
            hits = agent.context.internal_retriever.search(row["issue"], k=3)
            if hits:
                retrieval_hits += 1

        # Check grounding for answered cases
        if actual_decision == "answer":
            total_answers += 1
            just = actual.get("justification", "")
            if "High confidence grounding" in just or "grounded" in just.lower():
                grounded_answers += 1

        # Classification metrics for escalation
        is_expected_escalate = (expected_decision == "escalate")
        is_actual_escalate = (actual_decision == "escalate")

        if is_expected_escalate and is_actual_escalate:
            tp += 1
            correct_decisions += 1
        elif not is_expected_escalate and not is_actual_escalate:
            tn += 1
            correct_decisions += 1
        elif not is_expected_escalate and is_actual_escalate:
            fp += 1
        elif is_expected_escalate and not is_actual_escalate:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = correct_decisions / total_tickets if total_tickets > 0 else 0.0
    hit_rate = retrieval_hits / total_tickets if total_tickets > 0 else 0.0
    grounding_rate = grounded_answers / total_answers if total_answers > 0 else 0.0

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    print("\n" + "=" * 55)
    print(f"SUPPORT TRIAGE LANGGRAPH BENCHMARK RESULTS ({timestamp})")
    print("=" * 55)
    print(f"Total Evaluated Tickets   : {total_tickets}")
    print(f"Overall Decision Accuracy : {accuracy * 100:.2f}% ({correct_decisions}/{total_tickets})")
    print(f"Retrieval Hit Rate        : {hit_rate * 100:.2f}% ({retrieval_hits}/{total_tickets})")
    print(f"Escalation Precision      : {precision * 100:.2f}% (TP={tp}, FP={fp})")
    print(f"Escalation Recall         : {recall * 100:.2f}% (TP={tp}, FN={fn})")
    print(f"Escalation F1-Score       : {f1 * 100:.2f}%")
    print(f"Answer Grounding Pass Rate: {grounding_rate * 100:.2f}% ({grounded_answers}/{total_answers})")
    print("=" * 55 + "\n")

    return {
        "timestamp": timestamp,
        "total": total_tickets,
        "accuracy": accuracy,
        "retrieval_hit_rate": hit_rate,
        "escalation_precision": precision,
        "escalation_recall": recall,
        "escalation_f1": f1,
        "grounding_pass_rate": grounding_rate,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


if __name__ == "__main__":
    data_directory = ROOT / "data"
    tickets_path = ROOT / "tests" / "evaluation_tickets.csv"
    run_evaluation(tickets_path, data_directory)
