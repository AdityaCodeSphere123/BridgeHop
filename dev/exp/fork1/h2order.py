import os, sys, json, re
ROOT = "/Users/aditya/Desktop/classes/COL764-Assignment4"
sys.path.insert(0, ROOT + "/dev/exp/r10"); sys.path.insert(0, ROOT); os.chdir(ROOT)
import ablate
from a4.data import SearchEngine, read_jsonl
from a4.evaluate import answer_scores
from a4.tools import Tools
gold = list(read_jsonl("data/dev.jsonl")); engine = SearchEngine.load("data")
os.environ["AGENT"] = sys.argv[1]; A = ablate.load_agent()
snap = {}; orig_read = A.read
def read(q, notes, tools):
    if notes.hop == 1 and q not in snap:   # first hop-2 read: snapshot what it sees
        snap[q] = (notes.answers[0], [p for p in A.retry_order(notes.ranked(1), notes.shown)][:12])
    return orig_read(q, notes, tools)
A.read = read
def pos(lst, ids):
    for i, p in enumerate(lst):
        if p.doc_id in ids: return i
    return None
stats = {"cur": [], "bridge_first": []}
moved = []
for g in gold:
    A.run(g["question"], Tools(engine, ablate.CacheLLM()))
    if g["question"] not in snap: continue
    bridge, shown = snap[g["question"]]
    ids = set(g["supporting_doc_ids"][1:])  # last supporting doc = hop-2 paragraph
    if answer_scores(bridge, [g["decomposition"][0]["answer"]])[1] < 0.5: continue
    clean = lambda t: A.norm(re.sub(r"\s*\([^)]*\)\s*$", "", t))
    re_ = [p for p in shown if clean(p.title) == A.norm(bridge)] + [p for p in shown if clean(p.title) != A.norm(bridge)]
    a, b = pos(shown, ids), pos(re_, ids)
    stats["cur"].append(a); stats["bridge_first"].append(b)
    if a != b: moved.append((a, b))
for k, v in stats.items():
    got = [x for x in v if x is not None]
    print(f"{k:13s} n={len(v)} gold shown={len(got)} at pos0={sum(x==0 for x in got)} pos<=2={sum(x<=2 for x in got)} mean pos={sum(got)/len(got):.2f}")
up = sum(1 for a, b in moved if b < a); down = sum(1 for a, b in moved if b > a)
print("moved:", len(moved), "up", up, "down", down)
