from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize output.csv for final submission sanity checks.")
    parser.add_argument("--output", default=str(ROOT / "support_tickets" / "output.csv"))
    args = parser.parse_args()

    with Path(args.output).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    total = len(rows)
    escalated = sum(1 for row in rows if row.get("status") == "escalated")
    replied = sum(1 for row in rows if row.get("status") == "replied")
    invalid_labels = [
        index
        for index, row in enumerate(rows, start=1)
        if row.get("status") not in {"replied", "escalated"}
        or row.get("request_type") not in {"product_issue", "feature_request", "bug", "invalid"}
    ]
    missing_grounding = [index for index, row in enumerate(rows, start=1) if "Confidence=" not in row.get("justification", "")]
    rate = escalated / total if total else 0.0

    print(f"rows={total}")
    print(f"replied={replied}")
    print(f"escalated={escalated}")
    print(f"escalation_rate={rate:.1%}")
    print(f"invalid_label_rows={invalid_labels}")
    print(f"missing_confidence_rows={missing_grounding}")
    if total and not 0.20 <= rate <= 0.40:
        print("warning=escalation rate outside target review band")
    return 0 if not invalid_labels and not missing_grounding else 1


if __name__ == "__main__":
    raise SystemExit(main())
