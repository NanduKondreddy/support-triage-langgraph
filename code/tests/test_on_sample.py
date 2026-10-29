from __future__ import annotations

import csv
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from agent import SupportAgent  # noqa: E402


def normalize(value: str) -> str:
    return (value or "").strip().lower()


def test_sample_labels() -> None:
    agent = SupportAgent(ROOT / "data")
    sample = ROOT / "support_tickets" / "sample_support_tickets.csv"
    with sample.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    status_matches = 0
    request_type_matches = 0
    for row in rows:
        result = agent.triage(row)
        status_matches += normalize(result["status"]) == normalize(row["Status"])
        request_type_matches += normalize(result["request_type"]) == normalize(row["Request Type"])

    assert status_matches == len(rows)
    assert request_type_matches == len(rows)


if __name__ == "__main__":
    test_sample_labels()
    print("sample label checks passed")
