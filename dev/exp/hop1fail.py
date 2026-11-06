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
n = 0
for line in open("runs/v1b/traces.jsonl"):
    t = json.loads(line); g = G[t["id"]]
    golds = [g["answer"]] + g["answer_aliases"]; b = g["decomposition"][0]["answer"]
    has = lambda d, a: f" {N(a)} " in f" {N(C[d][0] + ' ' + C[d][1])} "
    gids = g["supporting_doc_ids"]
    h2 = next((d for d in gids if any(has(d, a) for a in golds)), gids[1]); h1 = next(d for d in gids if d != h2)
    for c in t["calls"]:
        if c["type"] == "llm" and "one step towards" in c["prompt"] and "best guess" not in c["prompt"]:
            lines = [l for l in c["prompt"].splitlines() if re.match(r"^\[\d+\] ", l)]
            pos = next((i + 1 for i, l in enumerate(lines) if C[h1][1][:60] in l), None)
            m = re.search(r"Answer\s*:\s*(.+)", c["output"]); a = (m.group(1) if m else c["output"]).strip()
            if pos and answer_scores(a, [b])[1] < 0.5:
                subq = re.search(r"Question: (.+)", c["prompt"].split("Passages:")[1]).group(1)
                print(f"Q: {g['question']}\n  Q1: {subq}\n  gold bridge: {b!r}  model: {a!r}  gold pos {pos}/{len(lines)}\n  gold passage: {C[h1][0]}: {C[h1][1][:180]}\n")
                n += 1
                break
    if n >= 25: break
