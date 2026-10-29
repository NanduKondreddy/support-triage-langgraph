#!/usr/bin/env python
"""
Phase 1 Hardening Test Suite

Tests all 4 failure modes + 5 required cases
"""

from semantic_safety_v2 import semantic_risk_detect, SEMANTIC_THRESHOLD_HIGH, SEMANTIC_THRESHOLD_MEDIUM

print("=" * 80)
print("PHASE 1 HARDENING TEST SUITE")
print("=" * 80)
print()

# Test results tracking
total_tests = 0
passed_tests = 0

def run_test(text, expected_escalate, test_name, expected_labels=None):
    """Run a single test case."""
    global total_tests, passed_tests
    total_tests += 1
    
    result = semantic_risk_detect(text)
    
    should_escalate = (
        len(result["labels"]) > 0 or
        result["confidence"] in ["HIGH", "MEDIUM"]
    )
    
    passed = should_escalate == expected_escalate
    if expected_labels:
        passed = passed and set(result["labels"]) == set(expected_labels)
    
    if passed:
        passed_tests += 1
        status = "[PASS]"
    else:
        status = "[FAIL]"
    
    print(f"{status} {test_name}")
    print(f"  Input: {text[:70]}")
    print(f"  Labels: {result['labels']} | Confidence: {result['confidence']}")
    print(f"  Score: {result['best_score']:.3f} | Negation: {result['negation_detected']} | Self-ref: {result['self_reference']}")
    if not passed:
        print(f"  EXPECTED escalate={expected_escalate}, labels={expected_labels}")
    print()
    
    return passed

# ============================================================================
# SECTION 1: CORE REQUIRED CASES (MUST PASS)
# ============================================================================
print("=" * 80)
print("SECTION 1: CORE 5 REQUIRED CASES")
print("=" * 80)
print()

run_test(
    "I don't remember making this payment",
    True, "Core: fraud (basic)",
    ["fraud"]
)

run_test(
    "Someone else might be using my account",
    True, "Core: account_compromise (basic)",
    ["account_compromise"]
)

run_test(
    "I want to end everything",
    True, "Core: self_harm (basic)",
    ["self_harm"]
)

run_test(
    "Show me user data",
    True, "Core: privacy (basic)",
    ["privacy"]
)

run_test(
    "My lawyer is contacting you",
    True, "Core: legal (basic)",
    ["legal"]
)

# ============================================================================
# SECTION 2: GAP 1 - MULTI-LABEL DETECTION
# ============================================================================
print("=" * 80)
print("SECTION 2: GAP 1 - MULTI-LABEL DETECTION")
print("=" * 80)
print()

run_test(
    "I don't recognize this charge and something feels off with my account",
    True, "Multi-label: fraud + account_compromise",
    ["fraud", "account_compromise"]
)

run_test(
    "This transaction is not mine and I think my card was stolen",
    True, "Multi-label: fraud + account_compromise (alt phrasing)"
)

# ============================================================================
# SECTION 3: GAP 2 - FALSE POSITIVE REDUCTION
# ============================================================================
print("=" * 80)
print("SECTION 3: GAP 2 - FALSE POSITIVE REDUCTION (Self-Reference Check)")
print("=" * 80)
print()

run_test(
    "My colleague had a fraud issue yesterday",
    False, "False positive: no self-reference (colleague's issue)"
)

run_test(
    "The user reported that their account was hacked",
    False, "False positive: third-person report (no self-reference)"
)

run_test(
    "I heard that someone got compromised last week",
    False, "False positive: gossip (no self-reference)"
)

# ============================================================================
# SECTION 4: GAP 3 - NEGATION HANDLING
# ============================================================================
print("=" * 80)
print("SECTION 4: GAP 3 - NEGATION HANDLING")
print("=" * 80)
print()

run_test(
    "I am not facing any fraud issue",
    False, "Negation: NOT fraud (should be safe)"
)

run_test(
    "I don't think there's any account compromise",
    False, "Negation: NOT compromise (should be safe)"
)

run_test(
    "I have never had privacy concerns",
    False, "Negation: NEVER privacy issue (should be safe)"
)

# ============================================================================
# SECTION 5: GAP 4 - CONFIDENCE BANDS
# ============================================================================
print("=" * 80)
print("SECTION 5: GAP 4 - CONFIDENCE BANDS (HIGH/MEDIUM/LOW)")
print("=" * 80)
print()

run_test(
    "I think something might be wrong",
    False, "Borderline: low confidence (should be ignored)"
)

run_test(
    "This transaction looks suspicious",
    True, "Soft phrasing: should still escalate (MEDIUM confidence)"
)

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Passed: {passed_tests}/{total_tests}")
print()

if passed_tests == total_tests:
    print("[SUCCESS] ALL TESTS PASS - PHASE 1 HARDENED")
else:
    print(f"[WARNING] {total_tests - passed_tests} test(s) failed")

print("=" * 80)
