"""Summarise a saved results .jsonl (valid JSON rate + exact match per document type)."""
import json
import sys
from collections import defaultdict

rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
stats = defaultdict(lambda: [0, 0, 0])
for r in rows:
    s = stats[r.get("doc_type", "all")]
    s[0] += 1
    s[1] += bool(r.get("valid_json"))
    s[2] += bool(r.get("exact_match"))
for k, (n, v, e) in sorted(stats.items()):
    print(f"{k:15s} n={n:4d}  valid JSON={v/n:6.1%}  exact match={e/n:6.1%}")
