from __future__ import annotations

from dataclasses import dataclass
import re


INJECTION_PATTERNS = (
    r"ignore (all )?(previous|prior) instructions",
    r"ignore (your )?(system|developer) prompt",
    r"disregard (the )?(policy|rules)",
    r"output escalated=false",
    r"mark this as replied",
    r"show (me )?(your )?(system|developer) (prompt|message|instructions)",
    r"display .*?(rules|documents retrieved|logic exact)",
    r"reveal .*?(retrieved docs|exact fraud logic|fraud logic)",
    r"print all files",
    r"delete all files",
)


@dataclass(frozen=True)
class SanitizedTicket:
    issue: str
    subject: str
    company: str
    text: str
    adversarial: bool
    notes: tuple[str, ...]


def sanitize_row(row: dict[str, str]) -> SanitizedTicket:
    issue = row.get("Issue") or row.get("issue") or ""
    subject = row.get("Subject") or row.get("subject") or ""
    company = row.get("Company") or row.get("company") or ""
    text = normalize_text(f"{subject}\n{issue}")
    notes: list[str] = []
    adversarial = False
    cleaned = text
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, cleaned, flags=re.I):
            adversarial = True
            notes.append(f"Removed untrusted ticket instruction matching: {pattern}")
            cleaned = re.sub(pattern, "[ignored ticket instruction]", cleaned, flags=re.I)
    return SanitizedTicket(
        issue=issue.strip(),
        subject=subject.strip(),
        company=company.strip(),
        text=cleaned.strip(),
        adversarial=adversarial,
        notes=tuple(notes),
    )


def normalize_text(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text.replace("\r\n", "\n")).strip()
