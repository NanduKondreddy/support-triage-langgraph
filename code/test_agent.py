import pandas as pd
from agent import SupportAgent

agent = SupportAgent()
df = pd.read_csv("../support_tickets/verify_final_6.csv")
results = []
for idx, row in df.iterrows():
    r = row.to_dict()
    res = agent.triage(r)
    results.append({"issue": r["issue"], "status": res["status"], "request_type": res["request_type"], "response": res["response"]})

pd.DataFrame(results).to_csv("verify_7_results.csv", index=False)
