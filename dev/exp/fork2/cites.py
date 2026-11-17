import os, sys, json
ROOT = "/Users/aditya/Desktop/classes/COL764-Assignment4"
sys.path.insert(0, ROOT + "/dev/exp/r10"); sys.path.insert(0, ROOT); os.chdir(ROOT)
import ablate
from a4.data import SearchEngine, read_jsonl
from a4.evaluate import evidence_f1
from a4.tools import Tools
gold = list(read_jsonl("data/dev.jsonl")); engine = SearchEngine.load("data")
os.environ["AGENT"] = "agent.py"; A = ablate.load_agent()
cap = {}; orig = A.answer
def answer(q, notes, tools):
    r = orig(q, notes, tools); cap[q] = (notes, r); return r
A.answer = answer
rows = []
for g in gold:
    A.run(g["question"], Tools(engine, ablate.CacheLLM()))
    notes, r = cap[g["question"]]
    rows.append((g, notes, r))
def strat(name, fn):
    ev = [evidence_f1(fn(g, n, r), g["supporting_doc_ids"]) for g, n, r in rows]
    base = [evidence_f1(r["citations"], g["supporting_doc_ids"]) for g, n, r in rows]
    d = sum(ev) / 300 - sum(base) / 300
    better = sum(a > b for a, b in zip(ev, base)); worse = sum(a < b for a, b in zip(ev, base))
    print(f"{name:58s} EvF1={sum(ev)/300:.4f} (d={d:+.4f}, dS={0.2*d:+.4f}) better={better} worse={worse}")
P = lambda n, d: n.passages.get(d)
def has(n, d, phrase): 
    p = P(n, d); return p is not None and bool(phrase) and A.mentions(p, phrase)
strat("current", lambda g, n, r: r["citations"])
def keep_supported(g, n, r):
    c = r["citations"]; fin = r["answer"]; b = n.answers[0]
    ok = [d for d in c if has(n, d, b) or has(n, d, fin) or A.named_in(P(n, d), g["question"])]
    return ok or c[:1]
strat("drop citations with neither bridge, answer nor named title", keep_supported)
def keep_supported2(g, n, r):
    c = r["citations"]; fin = r["answer"]; b = n.answers[0]
    ok = [d for d in c if has(n, d, b) or has(n, d, fin)]
    return ok or c[:1]
strat("drop citations with neither bridge nor answer", keep_supported2)
def drop_answer_unsupported(g, n, r):
    c = r["citations"]
    if len(c) == 2 and not has(n, c[1], r["answer"]): return c[:1]
    return c
strat("drop 2nd citation if it lacks the answer", drop_answer_unsupported)
def third(g, n, r, need_answer=True, need_bridge=True):
    c = list(r["citations"]); fin = r["answer"]; b = n.answers[0]
    if len(c) < 2: return c
    extra = None
    if need_answer and not has(n, c[1], fin):
        extra = next((p.doc_id for p in n.ranked(1) if p.doc_id not in c and A.mentions(p, fin)), None)
    if extra is None and need_bridge and b and not has(n, c[0], b):
        extra = next((p.doc_id for p in n.ranked(0) if p.doc_id not in c and A.mentions(p, b)), None)
    return c + [extra] if extra else c
strat("add 3rd: passage with the answer/bridge if a citation lacks it", third)
strat("add 3rd: passage with the answer if 2nd lacks it", lambda g, n, r: third(g, n, r, True, False))
strat("add 3rd: passage with the bridge if 1st lacks it", lambda g, n, r: third(g, n, r, False, True))
