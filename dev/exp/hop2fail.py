"""Recorded hop-2 reads that had the gold passage AND the right bridge, but answered wrong (no LLM calls)."""
import json, os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from a4.evaluate import answer_scores, normalize_answer as N
G = {json.loads(l)["id"]: json.loads(l) for l in open("data/dev.jsonl")}
want = {d for g in G.values() for d in g["supporting_doc_ids"]}
C = {}
for line in open("data/corpus.jsonl"):
    r = json.loads(line)
    if r["doc_id"] in want: C[r["doc_id"]] = (r["title"], r["text"])
for line in open("runs/v1b/traces.jsonl"):
    t = json.loads(line); g = G[t["id"]]
    golds = [g["answer"]] + g["answer_aliases"]; b = g["decomposition"][0]["answer"]
    has = lambda d, a: f" {N(a)} " in f" {N(C[d][0] + ' ' + C[d][1])} "
    gids = g["supporting_doc_ids"]; h2 = next((d for d in gids if any(has(d, a) for a in golds)), gids[1])
    for c in t["calls"]:
        pr = c.get("prompt", "")
        if c["type"] != "llm" or "We already know that #1" not in pr or "best guess" in pr: continue
        if N(b) not in N(pr.split("We already know that #1 =")[1][:150]): continue
        lines = [l for l in pr.splitlines() if re.match(r"^\[\d+\] ", l)]
        pos = next((i + 1 for i, l in enumerate(lines) if C[h2][1][:60] in l), None)
        m = re.search(r"Answer\s*:\s*(.+)", c["output"]); a = (m.group(1) if m else c["output"]).strip()
        n = re.search(r"Passage\s*:\s*\[?(\d+)", c["output"])
        if pos and answer_scores(a, golds)[1] < 0.5 and "unknown" not in a.lower():
            subq = re.search(r"Question: (.+)", pr.split("Passages:")[1]).group(1)
            cited = lines[int(n.group(1)) - 1][:110] if n and int(n.group(1)) <= len(lines) else None
            print(f"Q2: {subq}\n  gold {g['answer']!r} (pos {pos})  model {a!r} from passage {n.group(1) if n else '?'}\n  cited: {cited}\n  gold : {C[h2][0]}: {C[h2][1][:110]}\n")
            break
