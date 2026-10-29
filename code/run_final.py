import pandas as pd
from agent import SupportAgent
import traceback

try:
    from pathlib import Path
    agent = SupportAgent(Path("../data"))
    df = pd.read_csv("../final_tests.csv")
    df.fillna('', inplace=True)
    results = []
    
    with open("../final_tests_output.md", "w", encoding="utf-8") as f:
        f.write("# Final 30 Test Results\n\n")
        
        for idx, row in df.iterrows():
            r = row.to_dict()
            res = agent.triage(r)
            results.append({"issue": r["issue"], "status": res["status"], "request_type": res["request_type"], "product_area": res.get("product_area", ""), "response": res["response"], "justification": res["justification"]})
            
            f.write(f"**Issue {idx+1}:** {r['issue']}\n")
            f.write(f"**Status:** {res['status']} | **Type:** {res['request_type']} | **Area:** {res.get('product_area', '')}\n")
            f.write(f"**Justification:** {res['justification']}\n")
            f.write(f"**Response:** {res['response']}\n")
            f.write("---\n")

    pd.DataFrame(results).to_csv("../final_results.csv", index=False)
    print("SUCCESS")
except Exception as e:
    print("ERROR:")
    traceback.print_exc()
