"""One-off helper: replace the embedded plotly.js bundle in existing HTML reports by a CDN link (~4.7 MB smaller each).

    python tools_slim_reports.py
"""
import re

from data import REPORTS

pat = re.compile(r'<script[^>]*>/\*\*\s*\* plotly\.js v([\d.]+).*?</script>', re.S)
for f in REPORTS.rglob("*.html"):
    s = f.read_text(encoding="utf-8")
    new, n = pat.subn(lambda m: f'<script charset="utf-8" src="https://cdn.plot.ly/plotly-{m.group(1)}.min.js"></script>', s, count=1)
    if n:
        f.write_text(new, encoding="utf-8")
        print(f"{f.relative_to(REPORTS)}: {len(s) // 1024} KB -> {len(new) // 1024} KB")
