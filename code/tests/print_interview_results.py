import csv
with open('support_tickets/interview_results.csv', 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

tiers = [
    (1, 'TIER 1 (Easy FAQs)', 0, 3),
    (2, 'TIER 2 (Moderate)', 3, 6),
    (3, 'TIER 3 (Hard)', 6, 9),
    (4, 'TIER 4 (Adversarial)', 9, 12),
    (5, 'TIER 5 (Safety-Critical)', 12, 17)
]

for t_num, t_name, start, end in tiers:
    print(f'\n## {t_name}')
    for i in range(start, end):
        r = rows[i]
        print(f'\n### Test {t_num}.{i-start+1}')
        print(f'Company: {r["company"]} | Subject: {r["subject"].encode("ascii", "replace").decode()}')
        print(f'Issue: {r["issue"].encode("ascii", "replace").decode()}')
        print(f'=> status: {r["status"]}')
        print(f'=> request_type: {r["request_type"]}')
        print(f'=> product_area: {r["product_area"]}')
        print(f'=> response: {r["response"].encode("ascii", "replace").decode()}')
        print(f'=> justification: {r["justification"].encode("ascii", "replace").decode()}')
