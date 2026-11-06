"""How does the real model read? Uses recorded server reads (traces) - no LLM calls.
For every read prompt that contains the gold paragraph of its hop: was the answer right, 'unknown', or
wrong, as a function of the gold paragraph's position and the number of passages shown."""
import json, os, re, sys
from collections import defaultdict
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from a4.evaluate import answer_scores, normalize_answer as N
G = {json.loads(l)["id"]: json.loads(l) for l in open("data/dev.jsonl")}
want = {d for g in G.values() for d in g["supporting_doc_ids"]}
C = {}
for line in open("data/corpus.jsonl"):
    r = json.loads(line)
    if r["doc_id"] in want: C[r["doc_id"]] = (r["title"], r["text"])
stats = defaultdict(lambda: [0, 0, 0])  # right, unknown, wrong
pos_stats = defaultdict(lambda: [0, 0, 0]); n_stats = defaultdict(lambda: [0, 0, 0]); mode_stats = defaultdict(lambda: [0, 0, 0])
for line in open(sys.argv[1]):
    t = json.loads(line); g = G[t["id"]]
    golds = [g["answer"]] + g["answer_aliases"]; b = g["decomposition"][0]["answer"]
    has = lambda d, a: f" {N(a)} " in f" {N(C[d][0] + ' ' + C[d][1])} "
    gids = g["supporting_doc_ids"]
    h2 = next((d for d in gids if any(has(d, a) for a in golds)), gids[1]); h1 = next(d for d in gids if d != h2)
    for c in t["calls"]:
        if c["type"] != "llm": continue
        pr = c["prompt"]
        if "one step towards" in pr: hop, gd, target = 1, h1, [b]
        elif "We already know that #1" in pr: hop, gd, target = 2, h2, golds
        else: continue
        lines = [l for l in pr.splitlines() if re.match(r"^\[\d+\] ", l)]
        pos = next((i + 1 for i, l in enumerate(lines) if C[gd][1][:60] in l), None)
        if pos is None: continue
        if hop == 2 and N(b) not in N(pr.split("We already know that #1 =")[1][:150]): continue  # wrong bridge
        m = re.search(r"Answer\s*:\s*(.+)", c["output"]); a = (m.group(1) if m else c["output"]).strip()
        k = 0 if answer_scores(a, target)[1] >= 0.5 else (1 if "unknown" in a.lower() or "none" in a.lower()[:6] else 2)
        mode = "guess" if "best guess" in pr else "strict"
        for d, key in ((stats, hop), (pos_stats, (hop, min(pos, 8))), (n_stats, (hop, len(lines))), (mode_stats, (hop, mode))):
            d[key][k] += 1
fmt = lambda v: f"n={sum(v):4d} right={v[0]/sum(v):.2f} unknown={v[1]/sum(v):.2f} wrong={v[2]/sum(v):.2f}"
for k in sorted(stats): print("hop", k, fmt(stats[k]))
print("-- by position of gold passage (8 = 8+)")
for k in sorted(pos_stats): print(" ", k, fmt(pos_stats[k]))
print("-- by number of passages shown")
for k in sorted(n_stats): print(" ", k, fmt(n_stats[k]))
print("-- by instruction")
for k in sorted(mode_stats): print(" ", k, fmt(mode_stats[k]))
