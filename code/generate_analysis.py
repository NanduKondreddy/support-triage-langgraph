import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT_CSV = ROOT.parent / "support_tickets" / "output.csv"
ANALYSIS_MD = ROOT / "analysis.md"

def main():
    if not OUTPUT_CSV.exists():
        print(f"Output file not found: {OUTPUT_CSV}")
        return

    with OUTPUT_CSV.open('r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    if total == 0:
        return

    status_counts = Counter(r['status'] for r in rows)
    type_counts = Counter(r['request_type'] for r in rows)
    
    comp_breakdown = {}
    for r in rows:
        comp = r['company'] or 'None'
        if comp not in comp_breakdown:
            comp_breakdown[comp] = {'replied': 0, 'escalated': 0}
        comp_breakdown[comp][r['status']] += 1

    escalation_triggers = Counter()
    for r in rows:
        if r['status'] == 'escalated':
            j = r['justification'].lower()
            if 'security' in j or 'compromise' in j:
                escalation_triggers['security/compromise'] += 1
            elif 'refund' in j or 'billing' in j:
                escalation_triggers['billing/refund'] += 1
            elif 'legal' in j:
                escalation_triggers['legal_dispute'] += 1
            elif 'account authority' in j or 'access changes' in j:
                escalation_triggers['account_authority'] += 1
            elif 'sitewide outage' in j:
                escalation_triggers['sitewide_outage'] += 1
            elif 'ambiguous' in j:
                escalation_triggers['ambiguous_context'] += 1
            elif 'confidence was too low' in j:
                escalation_triggers['low_confidence'] += 1
            elif 'unrelated' in j or 'outside supported scope' in j or 'invalid' in r['request_type']:
                 escalation_triggers['out_of_scope'] += 1
            else:
                escalation_triggers['other'] += 1

    out = []
    out.append('# Output Analysis\n')
    out.append(f'- Total tickets: {total}')
    out.append(f'- Replied: {status_counts["replied"]} ({status_counts["replied"]/total:.1%})')
    out.append(f'- Escalated: {status_counts["escalated"]} ({status_counts["escalated"]/total:.1%})\n')

    out.append('## By Company')
    for comp, counts in sorted(comp_breakdown.items()):
        c_total = counts['replied'] + counts['escalated']
        rep_pct = counts['replied'] / c_total if c_total else 0
        esc_pct = counts['escalated'] / c_total if c_total else 0
        out.append(f'- **{comp}**: {rep_pct:.0%} replied / {esc_pct:.0%} escalated ({c_total} total)')

    out.append('\n## Request Type Distribution')
    for req_type, count in type_counts.most_common():
        out.append(f'- {req_type}: {count}')

    out.append('\n## Escalation Triggers Fired')
    for trigger, count in escalation_triggers.most_common():
        out.append(f'- {trigger}: {count}')

    with ANALYSIS_MD.open('w', encoding='utf-8') as f:
        f.write('\n'.join(out) + '\n')

    print(f"Wrote analysis to {ANALYSIS_MD}")

if __name__ == '__main__':
    main()
