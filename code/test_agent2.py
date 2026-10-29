import pandas as pd
from agent import SupportAgent
import traceback

try:
    agent = SupportAgent()
    df = pd.read_csv("../support_tickets/verify_final_6.csv")
    df.fillna('', inplace=True)
    results = []
    for idx, row in df.iterrows():
        r = row.to_dict()
        res = agent.triage(r)
        results.append({"issue": r["issue"], "status": res["status"], "request_type": res["request_type"], "response": res["response"], "justification": res["justification"]})

    pd.DataFrame(results).to_csv("verify_final_6_results.csv", index=False)
    print("SUCCESS")
except Exception as e:
    print("ERROR:")
    traceback.print_exc()
