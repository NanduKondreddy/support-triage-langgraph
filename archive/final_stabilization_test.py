import os
import sys
import json
from pathlib import Path

# Add code directory to path
sys.path.append(str(Path.cwd() / "code"))

try:
    from agent import SupportAgent
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def run_stabilization_tests():
    # Initialize Agent
    agent = SupportAgent(Path("data"))

    test_cases = [
        "refund",
        "login",
        "login?",
        "payment failed",
        "###$$$ random text",
        "payment failed and account locked",
        "refund not processed after payment failed",
        "my visa got rejected for canada",
        "payment??? failed!!",
        "account issue maybe login problem not sure",
        "Claude subscription payment failed",
        "visa rejected and claude not working",
        "idk something wrong"
    ]

    print(f"{'#'*20} FINAL STABILIZATION TESTS {'#'*20}\n")
    
    results = []
    
    for i, test_input in enumerate(test_cases, 1):
        row = {"Issue": test_input, "Subject": "", "Company": "unknown"}
        output = agent.triage(row)
        
        # Parse justification
        just = output.get("justification", "{}")
        if isinstance(just, str):
            try: just = json.loads(just)
            except: pass
        
        print(f"TEST {i}: '{test_input}'")
        print(f"DECISION: {output.get('status')} | AREA: {output.get('product_area')}")
        
        if isinstance(just, dict) and "confidence" in just:
            print(f"CONFIDENCE: {just.get('confidence', 'N/A')}")
            print(f"SCORES: KW={just.get('keyword_score')} DOM={just.get('domain_score')} COH={just.get('intent_coherence')} SIG={just.get('signal_strength')}")
            print(f"FLAGS: Merged={just.get('merged')} Intent='{just.get('final_intent')}'")
        
        print(f"REASON: {just.get('reason', 'N/A')}")
        print("-" * 50)
        
        results.append({
            "input": test_input,
            "output": output,
            "justification": just
        })
        
    return results

if __name__ == "__main__":
    run_stabilization_tests()
