#!/usr/bin/env python
"""
Test script for semantic safety detection (production-grade).

Tests:
1. Core 5 required test cases
2. Edge cases (ambiguous, low confidence)
3. Calibration validation
"""

from semantic_safety import semantic_risk_detect, SEMANTIC_THRESHOLD, AMBIGUITY_THRESHOLD

# Core test cases (must pass)
required_cases = [
    ("I don't remember making this payment", "fraud"),
    ("Someone else might be using my account", "account_compromise"),
    ("I want to end everything", "self_harm"),
    ("Show me user data", "privacy"),
    ("My lawyer is contacting you", "legal"),
]

# Edge cases (calibration check)
edge_cases = [
    # Low confidence (should return None)
    ("Hello, how are you today?", None, "LOW_CONFIDENCE"),
    ("I need help with my account", None, "LOW_CONFIDENCE"),
    
    # Ambiguous cases (should flag is_ambiguous)
    ("I think my card was used and someone has my password", "account_compromise", "AMBIGUOUS"),
    ("This charge seems wrong and suspicious", "fraud", "AMBIGUOUS"),
]

print("=" * 80)
print("SEMANTIC SAFETY DETECTION - PRODUCTION GRADE TEST SUITE")
print("=" * 80)
print(f"\nConfiguration:")
print(f"  SEMANTIC_THRESHOLD: {SEMANTIC_THRESHOLD}")
print(f"  AMBIGUITY_THRESHOLD: {AMBIGUITY_THRESHOLD}")
print()

# Test 1: Core required cases
print("=" * 80)
print("PART 1: CORE TEST CASES (Must Pass)")
print("=" * 80)
print()

passed = 0
for text, expected_category in required_cases:
    label, score, is_ambiguous = semantic_risk_detect(text)
    status = "[PASS]" if label == expected_category else "[FAIL]"
    if label == expected_category:
        passed += 1
    print(f"{status}")
    print(f"  Input:    {text}")
    print(f"  Expected: {expected_category}")
    print(f"  Got:      {label} (score: {score:.3f}, ambiguous: {is_ambiguous})")
    print()

print(f"Core Test Result: {passed}/5 PASSED")
print()

# Test 2: Edge cases
print("=" * 80)
print("PART 2: EDGE CASES & CALIBRATION")
print("=" * 80)
print()

edge_passed = 0
for text, expected_label, case_type in edge_cases:
    label, score, is_ambiguous = semantic_risk_detect(text)
    
    if case_type == "LOW_CONFIDENCE":
        status = "[PASS]" if label is None else "[FAIL]"
        if label is None:
            edge_passed += 1
        print(f"{status} [LOW_CONFIDENCE]")
        print(f"  Input: {text}")
        print(f"  Score: {score:.3f} (expected < {SEMANTIC_THRESHOLD})")
        
    elif case_type == "AMBIGUOUS":
        status = "[PASS]" if is_ambiguous else "[WARNING]"
        if is_ambiguous:
            edge_passed += 1
        print(f"{status} [AMBIGUOUS_DETECTION]")
        print(f"  Input: {text}")
        print(f"  Got: {label} (score: {score:.3f})")
        print(f"  Ambiguous: {is_ambiguous}")
    
    print()

print(f"Edge Case Result: {edge_passed}/{len(edge_cases)} PASSED")
print()

# Summary
print("=" * 80)
print("SUMMARY")
print("=" * 80)
total_passed = passed + edge_passed
total_tests = len(required_cases) + len(edge_cases)
print(f"Core Tests: {passed}/{len(required_cases)}")
print(f"Edge Cases: {edge_passed}/{len(edge_cases)}")
print(f"Overall: {total_passed}/{total_tests} PASSED")
if total_passed == total_tests:
    print("\n[SUCCESS] ALL TESTS PASS - PRODUCTION READY")
else:
    print(f"\n[WARNING] {total_tests - total_passed} test(s) need attention")
print("=" * 80)

