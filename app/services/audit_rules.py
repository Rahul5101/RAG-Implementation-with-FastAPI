import re

RULES = [
    {
        'id': 'auto_renewal_short_notice',
        'description': 'Auto-renewal with short notice period (<30 days)',
        'severity': 'high',
        'matcher': lambda text: bool(re.search(r"auto-?renew(?:al)?[\s\S]{0,120}(notice|days|written notice).*?(?:30|thirty)", text, re.I) == False and bool(re.search(r"auto-?renew|renew automatically", text, re.I)))
    },
    {
        'id': 'unlimited_liability',
        'description': 'Unlimited liability or no liability cap',
        'severity': 'critical',
        'matcher': lambda text: bool(re.search(r"liabilit(y|ies).*?(unlimited|no cap|no limit)", text, re.I))
    },
    {
        'id': 'broad_indemnity',
        'description': 'Broad indemnity clause (very broad/reciprocal wording)',
        'severity': 'medium',
        'matcher': lambda text: bool(re.search(r"indemnif(y|ies|ication).*?(hold harmless|defend|indemnify).*?\b(against|from)\b", text, re.I))
    }
]


def run_audit_on_text(text: str):
    findings = []
    for r in RULES:
        try:
            triggered = r['matcher'](text)
        except Exception:
            triggered = False
        if triggered:
            # find evidence span
            m = re.search(r"(.{0,80}?(auto-?renew|liabilit|indemnif)[\s\S]{0,160}?)", text, re.I)
            evidence = m.group(0) if m else text[:200]
            findings.append({
                'id': r['id'],
                'description': r['description'],
                'severity': r['severity'],
                'evidence': evidence
            })
    return findings