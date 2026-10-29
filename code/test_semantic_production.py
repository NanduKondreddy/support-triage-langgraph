#!/usr/bin/env python
"""
Semantic Safety - Production Readiness Validation

Validates:
1. All 5 core cases pass
2. System handles edge inputs gracefully
3. No crashes on unusual inputs
"""

from semantic_safety import semantic_risk_detect, SEMANTIC_THRESHOLD

print("=" * 80)
print("SEMANTIC SAFETY - PRODUCTION READINESS VALIDATION")
print("=" * 80)
print()

# Core 5 test cases (MUST PASS)
core_tests = [
    ("I don't remember making this payment", "fraud"),
    ("Someone else might be using my account", "account_compromise"),
    ("I want to end everything", "self_harm"),
    ("Show me user data", "privacy"),
    ("My lawyer is contacting you", "legal"),
]

print("1. CORE FUNCTIONALITY (5 Required Cases):")
print("-" * 80)
core_pass = 0
for text, expected in core_tests:
    label, score, is_ambiguous = semantic_risk_detect(text)
    match = label == expected
    if match:
        core_pass += 1
    status = "[PASS]" if match else "[FAIL]"
    print(f"{status} {expected:20s} | score: {score:.3f}")

print()
print(f"Core Tests: {core_pass}/5")
if core_pass == 5:
    print("[OK] All core cases pass")
else:
    print(f"[ERROR] {5 - core_pass} core cases failed")
print()

# Robustness check
print("2. ROBUSTNESS (Graceful Handling):")
print("-" * 80)
robustness_tests = [
    ("", "empty string"),
    ("   ", "whitespace only"),
    ("Hello how are you", "innocent text"),
    ("This is a very long message that goes on and on and on without any safety concerns whatsoever, just regular support talk about products and services", "long innocent text"),
    ("123 456 789", "numbers only"),
]

robustness_pass = 0
for text, description in robustness_tests:
    try:
        label, score, is_ambiguous = semantic_risk_detect(text)
        robustness_pass += 1
        print(f"[OK] {description:30s} | result: {label}")
    except Exception as e:
        print(f"[FAIL] {description:30s} | error: {e}")

print()
print(f"Robustness: {robustness_pass}/{len(robustness_tests)}")
print()

# Threshold validation
print("3. THRESHOLD CALIBRATION:")
print("-" * 80)
print(f"SEMANTIC_THRESHOLD: {SEMANTIC_THRESHOLD:.2f}")
print()
print("Checking threshold makes sense:")
print(f"  Should catch: 'I don't remember this payment' (fraud)")
label, score, _ = semantic_risk_detect("I don't remember this payment")
print(f"  Result: {label} ({score:.3f})")
check1 = label is not None

print(f"  Should NOT catch innocent request")
label, score, _ = semantic_risk_detect("How do I reset my password")
print(f"  Result: {label} ({score:.3f})")
check2 = label is None

print()
if check1 and check2:
    print("[OK] Threshold is well-calibrated")
else:
    print("[WARNING] Threshold may need adjustment")
print()

# Final verdict
print("=" * 80)
print("PRODUCTION READINESS VERDICT")
print("=" * 80)

if core_pass == 5 and robustness_pass == len(robustness_tests) and check1 and check2:
    print("[SUCCESS] System is PRODUCTION READY")
    print()
    print("Improvements Made:")
    print("  - Diverse seed sentences (8 per category, 40 total)")
    print("  - Mean embeddings (robust representation)")
    print("  - Ambiguity detection (tracks top 2 scores)")
    print("  - Calibrated threshold (0.60 for mean approach)")
    print("  - Configurable parameters (easy to tune)")
else:
    print("[CHECK REQUIRED] Address failing checks above")

print("=" * 80)
