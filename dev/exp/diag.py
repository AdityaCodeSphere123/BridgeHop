"""Chain diagnostics for a run: hop-1 retrieval -> bridge -> hop-2 retrieval -> answer."""
import json, os, re, sys
from collections import Counter
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from a4.data import read_jsonl
from a4.evaluate import answer_scores, normalize_answer as N, evidence_f1

run = sys.argv[1]
gold = {g["id"]: g for g in read_jsonl(os.path.join(ROOT, "data/dev.jsonl"))}
preds = {r["id"]: r for r in read_jsonl(os.path.join(run, "predictions.jsonl"))}
traces = {t["id"]: t for t in read_jsonl(os.path.join(run, "traces.jsonl"))}
texts = {}
want = {d for g in gold.values() for d in g["supporting_doc_ids"]}
for line in open(os.path.join(ROOT, "data/corpus.jsonl")):
    r = json.loads(line)
    if r["doc_id"] in want: texts[r["doc_id"]] = r["title"] + " " + r["text"]
def has(d, a): a = N(a); return bool(a) and f" {a} " in f" {N(texts.get(d,''))} "
C = Counter(); f1s = Counter(); n = Counter()
for qid, r in preds.items():
    g = gold[qid]; golds = [g["answer"]] + g["answer_aliases"]
    gids = g["supporting_doc_ids"]
    h2 = next((d for d in gids if any(has(d, a) for a in golds)), gids[1]); h1 = next(d for d in gids if d != h2)
    ret = {d for c in traces[qid]["calls"] if c["type"] == "search" for d in c["doc_ids"]}
    _, f1 = answer_scores(r["answer"], golds)
    key = ("h1" if h1 in ret else "--") + "/" + ("h2" if h2 in ret else "--")
    n[key] += 1; f1s[key] += f1
    cits = r["citations"]
    C["cite_h1"] += h1 in cits; C["cite_h2"] += h2 in cits; C["ncit_%d" % len(cits)] += 1
for k in sorted(n): print(f"{k}: n={n[k]:3d}  meanF1={f1s[k]/n[k]:.3f}  F1-mass-lost={n[k]-f1s[k]:.1f}")
print(dict(C))

def bridge(t):
    best = ""
    for c in t["calls"]:
        if c["type"] == "llm" and "This is one step towards answering" in c.get("prompt", ""):
            m = re.search(r"Answer\s*:\s*(.+)", c.get("output") or "")
            if m and "unknown" not in m.group(1).lower(): best = m.group(1).strip(" .")
    return best
B = Counter(); BF = Counter()
ex = []
for qid, r in preds.items():
    g = gold[qid]; golds = [g["answer"]] + g["answer_aliases"]
    gids = g["supporting_doc_ids"]
    h2 = next((d for d in gids if any(has(d, a) for a in golds)), gids[1]); h1 = next(d for d in gids if d != h2)
    ret = {d for c in traces[qid]["calls"] if c["type"] == "search" for d in c["doc_ids"]}
    b = bridge(traces[qid]); gb = g["decomposition"][0]["answer"]
    ok = "none" if not b else ("ok" if answer_scores(b, [gb])[1] >= 0.5 else "wrong")
    _, f1 = answer_scores(r["answer"], golds)
    key = ok + "/" + ("h2" if h2 in ret else "--")
    B[key] += 1; BF[key] += f1
    if key == "ok/--" and len(ex) < 30: ex.append((g["question"], b, [c["query"] for c in traces[qid]["calls"] if c["type"]=="search"], texts[h2][:160]))
for k in sorted(B): print(f"bridge {k}: n={B[k]} meanF1={BF[k]/B[k]:.3f}")
if len(sys.argv) > 2:
    for e in ex: print("\nQ:", e[0], "\nB:", e[1], "\nqueries:", e[2], "\nH2:", e[3])
