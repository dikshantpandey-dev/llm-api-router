from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_api_router.providers import LocalDeterministicProvider
from llm_api_router.router import LLMRouter, RouteRequest

providers = [
    LocalDeterministicProvider("local-fast", "deterministic-mini", 80, 0.02, 0.06),
    LocalDeterministicProvider("local-balanced", "deterministic-balanced", 180, 0.05, 0.20),
    LocalDeterministicProvider("local-reasoning", "deterministic-o3-mini", 350, 0.30, 1.20),
]
router = LLMRouter(providers)
requests = [
    RouteRequest("Summarize a Nepali RAG benchmark result in two bullet points.", priority="cost"),
    RouteRequest("Classify whether a retrieval answer is grounded in citations.", priority="latency"),
    RouteRequest("Plan a research experiment comparing embedding models.", needs_reasoning=True),
]
traces = []
for item in requests:
    response, trace = router.complete(item)
    traces.append({**trace, "prompt": item.prompt, "text": response.text, **asdict(response)})
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "docs").mkdir(exist_ok=True)
(ROOT / "results" / "routing_trace.json").write_text(json.dumps(traces, indent=2), encoding="utf-8")
with (ROOT / "results" / "routing_trace.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(traces[0].keys()))
    writer.writeheader()
    writer.writerows(traces)
rows = "".join(
    f"<tr><td>{row['priority']}</td><td>{row['selected_model']}</td><td>{row['latency_ms']}</td><td>${row['estimated_cost_usd']:.6f}</td><td>{row['prompt']}</td></tr>"
    for row in traces
)
html = f'''<!doctype html><html><head><meta charset="utf-8"><title>LLM API Router</title>
<style>body{{font-family:Inter,Arial,sans-serif;background:#f7f7fb;color:#1d2433;margin:0}}main{{max-width:1120px;margin:0 auto;padding:42px}}h1{{font-size:40px;margin:0}}.strip{{display:flex;gap:14px;margin:24px 0}}.pill{{background:#111827;color:white;border-radius:999px;padding:10px 16px}}table{{width:100%;border-collapse:collapse;background:white;border:1px solid #dde1ea}}td,th{{padding:13px;border-bottom:1px solid #e6e9f0;text-align:left}}th{{background:#4c1d95;color:white}}</style></head><body><main>
<h1>LLM API Router</h1><p>Policy-based routing across cost, latency, and reasoning needs.</p><section class="strip"><span class="pill">{len(providers)} providers</span><span class="pill">{len(traces)} routed requests</span><span class="pill">Azure-ready</span></section>
<table><tr><th>Priority</th><th>Selected model</th><th>Latency ms</th><th>Cost</th><th>Prompt</th></tr>{rows}</table></main></body></html>'''
(ROOT / "docs" / "dashboard.html").write_text(html, encoding="utf-8")
print(json.dumps(traces, indent=2))
