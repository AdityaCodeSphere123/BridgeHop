"""Plan quality from recorded server plans (no LLM): how often is Q1 just the whole question?"""
import json, os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from a4.evaluate import answer_scores, normalize_answer as N
G = {json.loads(l)["id"]: json.loads(l) for l in open("data/dev.jsonl")}
P = {json.loads(l)["id"]: json.loads(l) for l in open("runs/v1b/predictions.jsonl")}
STOP = set("a an the of in on at to for by with from and or is was were are did do does what which who whom whose when where why how".split())
def content(s): return [w for w in N(s).split() if w not in STOP]
rows = []
for line in open("runs/v1b/traces.jsonl"):
    t = json.loads(line); g = G[t["id"]]; out = t["calls"][0]["output"]
    q1 = re.search(r"Q1\s*:\s*(.+)", out); q2 = re.search(r"Q2\s*:\s*(.+)", out)
    parsed = bool(q1 and q2 and "#1" in q2.group(1))
    q = set(content(g["question"]))
    cov = len(set(content(q1.group(1))) & q) / max(1, len(q)) if q1 else 1.0
    f1 = answer_scores(P[t["id"]]["answer"], [g["answer"]] + g["answer_aliases"])[1]
    rows.append((parsed, cov, f1, g["question"], q1.group(1) if q1 else ""))
import statistics
for lo, hi in ((0, .5), (.5, .7), (.7, .8), (.8, .9), (.9, 1.01)):
    sel = [r for r in rows if r[0] and lo <= r[1] < hi]
    if sel: print(f"parsed, Q1 covers {lo:.1f}-{hi:.1f} of question words: n={len(sel):3d} meanF1={statistics.mean(r[2] for r in sel):.3f}")
sel = [r for r in rows if not r[0]]; print(f"unparsed: n={len(sel)} meanF1={statistics.mean(r[2] for r in sel):.3f}")
if len(sys.argv) > 1:
    for r in sorted(rows, key=lambda r: -r[1])[:40]:
        if r[0]: print(f"{r[1]:.2f} F1={r[2]:.2f} | {r[3]}\n      Q1: {r[4]}")
