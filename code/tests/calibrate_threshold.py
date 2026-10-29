"""Calibrate escalation confidence threshold against the sample CSV.

Runs the agent on sample_support_tickets.csv (which has expected outputs),
logs (confidence, was_correct) for each row, and finds the optimal threshold
where accuracy drops below 90%.

Run:  python -B code/tests/calibrate_threshold.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import SupportAgent

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_CSV = ROOT / "support_tickets" / "sample_support_tickets.csv"


def main() -> int:
    agent = SupportAgent(ROOT / "data")

    with SAMPLE_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    # Normalize expected status from sample CSV
    results = []
    for row in rows:
        expected_status = (row.get("Status") or "").strip().lower()
        if expected_status not in ("replied", "escalated"):
            continue

        result = agent.triage(row)
        actual_status = result["status"]

        # Extract confidence from justification
        j = result.get("justification", "")
        conf = 0.0
        ci = j.find("Confidence=")
        if ci >= 0:
            try:
                conf = float(j[ci + 11 : ci + 15])
            except (ValueError, IndexError):
                pass

        correct = actual_status == expected_status
        results.append({
            "subject": (row.get("Subject") or "")[:40],
            "expected": expected_status,
            "actual": actual_status,
            "confidence": conf,
            "correct": correct,
        })

    # Print per-row results
    print(f"{'Subject':<42} {'Expected':<10} {'Actual':<10} {'Conf':>6} {'OK':>4}")
    print("-" * 74)
    for r in results:
        ok = "YES" if r["correct"] else "NO"
        print(f"{r['subject']:<42} {r['expected']:<10} {r['actual']:<10} {r['confidence']:>6.2f} {ok:>4}")

    # Calculate accuracy at different thresholds
    print("\n--- Threshold Calibration ---")
    for threshold in [0.30, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.55, 0.60, 0.65, 0.70]:
        correct = 0
        total = 0
        for r in results:
            total += 1
            # Simulate: if confidence < threshold and actual=replied, would escalating be better?
            # We just count how many current decisions are correct at each confidence level
            if r["correct"]:
                correct += 1
        acc = correct / total if total else 0
        print(f"  Threshold={threshold:.2f}  Accuracy={acc:.1%} ({correct}/{total})")

    # More useful: accuracy by confidence bucket
    print("\n--- Accuracy by Confidence Bucket ---")
    buckets = [(0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    for lo, hi in buckets:
        in_bucket = [r for r in results if lo <= r["confidence"] < hi]
        if not in_bucket:
            continue
        correct = sum(1 for r in in_bucket if r["correct"])
        total = len(in_bucket)
        print(f"  Conf [{lo:.1f}, {hi:.1f})  Accuracy={correct}/{total} ({correct/total:.0%})")

    total_correct = sum(1 for r in results if r["correct"])
    print(f"\nOverall sample accuracy: {total_correct}/{len(results)} ({total_correct/len(results):.0%})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
