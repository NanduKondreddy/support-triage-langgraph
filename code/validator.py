from __future__ import annotations


ALLOWED_STATUS = {"replied", "escalated"}
ALLOWED_REQUEST_TYPE = {"product_issue", "feature_request", "bug", "invalid"}
OUTPUT_FIELDS = ["response", "product_area", "status", "request_type", "justification"]


def validate_output(result: dict[str, str]) -> dict[str, str]:
    """Ensure the agent's output exactly matches the judge's required schema.
    
    Validates presence and correctness of all 5 required columns. If the agent
    failed to produce a valid status or request_type, falls back to safe defaults
    (escalate/invalid) rather than failing the evaluation pipeline.
    """
    cleaned = {field: str(result.get(field, "") or "").strip() for field in OUTPUT_FIELDS}
    if cleaned["status"] not in ALLOWED_STATUS:
        cleaned["status"] = "escalated"
    if cleaned["request_type"] not in ALLOWED_REQUEST_TYPE:
        cleaned["request_type"] = "invalid"
    if not cleaned["response"]:
        cleaned["response"] = "Escalate to a human support specialist."
        cleaned["status"] = "escalated"
    if not cleaned["justification"]:
        cleaned["justification"] = "Validator supplied a fallback because the agent returned an incomplete result."
    return cleaned
