"""Main entry point for the HackerRank Orchestrate Support Triage Agent.

This script parses command-line arguments, loads the support ticket CSV,
initializes the SupportAgent with the documentation corpus, and processes
each ticket sequentially. The results are written to an output CSV.

Zero external dependencies required.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from agent import SupportAgent
from validator import ALLOWED_REQUEST_TYPE, ALLOWED_STATUS

try:
    from triage_graph import LangGraphSupportAgent
    HAS_GRAPH = True
except ImportError:
    HAS_GRAPH = False


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "support_issues" / "support_issues.csv" if (ROOT / "support_issues" / "support_issues.csv").exists() else ROOT / "support_tickets" / "support_tickets.csv"
DEFAULT_OUTPUT = ROOT / "support_issues" / "output.csv"
DEFAULT_DATA = ROOT / "data"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments including input/output paths and flags.
    """
    parser = argparse.ArgumentParser(description="Run the HackerRank Orchestrate support triage agent.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input CSV path.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output CSV path.")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Support corpus directory.")
    parser.add_argument("--pipeline", choices=["graph", "legacy"], default="graph" if HAS_GRAPH else "legacy", help="Execution pipeline: 'graph' (LangGraph) or 'legacy' (original).")
    parser.add_argument("--debug", action="store_true", help="Print detailed per-ticket routing and retrieval decisions.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if args.pipeline == "graph" and HAS_GRAPH:
        print("[Engine] Initializing LangGraph Support Agent StateGraph...")
        agent = LangGraphSupportAgent(Path(args.data))
    else:
        print("[Engine] Initializing Legacy Support Agent...")
        agent = SupportAgent(Path(args.data))

    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    fieldnames = ["issue", "subject", "company", "response", "product_area", "status", "request_type", "justification"]
    results = []
    for index, row in enumerate(rows, start=1):
        result = agent.triage(row)
        output_row = {
            "issue": row.get("Issue") or row.get("issue") or "",
            "subject": row.get("Subject") or row.get("subject") or "",
            "company": row.get("Company") or row.get("company") or "",
            **result,
        }
        results.append(output_row)
        if args.debug:
            issue_snip = (output_row['issue'][:60] + "...") if len(output_row['issue']) > 60 else output_row['issue']
            print(f"\n[{index:03d}] Issue: {issue_snip.strip()}")
            print(f"      [Router] Company: {output_row['company'] or 'None'}")
            print(f"      [Agent]  Type: {result['request_type']} | Area: {result['product_area']}")
            print(f"      [Output] Status: {result['status'].upper()}")
            print(f"      [Reason] {result['justification']}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Wrote {len(results)} predictions to {output_path}")
    statuses = {row["status"] for row in results}
    request_types = {row["request_type"] for row in results}
    if not statuses.issubset(ALLOWED_STATUS) or not request_types.issubset(ALLOWED_REQUEST_TYPE):
        raise ValueError("Output validator failed to enforce allowed labels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
