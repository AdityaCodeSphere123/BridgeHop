import os, sys, collections
ROOT = "/Users/aditya/Desktop/classes/COL764-Assignment4"
sys.path.insert(0, ROOT + "/dev/exp/r10"); sys.path.insert(0, ROOT); os.chdir(ROOT)
import ablate
from a4.data import SearchEngine, read_jsonl
from a4.tools import Tools
gold = list(read_jsonl("data/dev.jsonl")); engine = SearchEngine.load("data")
os.environ["AGENT"] = "agent.py"; A = ablate.load_agent()
cap = {}; orig = A.answer
def answer(q, notes, tools):
    r = orig(q, notes, tools); cap[q] = (notes, r); return r
A.answer = answer
c = collections.Counter(); slot = collections.Counter()
for g in gold:
    A.run(g["question"], Tools(engine, ablate.CacheLLM())); n, r = cap[g["question"]]
    gs = g["supporting_doc_ids"]; cited = r["citations"]
    for d in gs:
        if d in cited: c["gold cited"] += 1
        elif d in n.passages: c["gold RETRIEVED but not cited"] += 1
        else: c["gold never retrieved"] += 1
    for i, d in enumerate(cited[:2]):
        slot[(i, d in gs)] += 1
print(dict(c)); print("citation slot (0=bridge,1=answer) correct?:", dict(slot))
