import os, sys, json
ROOT="/Users/aditya/Desktop/classes/COL764-Assignment4"
sys.path.insert(0, ROOT+"/dev/exp/r10"); sys.path.insert(0, ROOT); os.chdir(ROOT)
import ablate
from a4.data import SearchEngine, read_jsonl
gold = {g["id"]: g for g in read_jsonl("data/dev.jsonl")}; engine = SearchEngine.load("data")
def go(p):
    os.environ["AGENT"] = p; return ablate.run("full", list(gold.values()), engine)
a, _ = go(sys.argv[1]); b, mb = go(sys.argv[2])
print(f"A={sum(v[1] for v in a.values())/300:.4f} B={sum(v[1] for v in b.values())/300:.4f} B misses={len(mb)}")
ch = sorted([(round(b[i][1]-a[i][1], 2), gold[i]["question"][:70], gold[i]["answer"], a[i][0], b[i][0]) for i in a if abs(a[i][1]-b[i][1]) > 1e-9])
for c in ch: print(c)
