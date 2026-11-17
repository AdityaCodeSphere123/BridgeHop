import os, sys, re
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
    A.run(g["question"], Tools(engine, ablate.CacheLLM())); rows.append((g,) + cap[g["question"]])
base = [evidence_f1(r["citations"], g["supporting_doc_ids"]) for g, n, r in rows]
def strat(name, fn):
    ev = [evidence_f1(fn(g, n, r), g["supporting_doc_ids"]) for g, n, r in rows]
    d = sum(ev) / 300 - sum(base) / 300
    print(f"{name:62s} EvF1={sum(ev)/300:.4f} dS={0.2*d:+.4f} better={sum(a>b for a,b in zip(ev,base))} worse={sum(a<b for a,b in zip(ev,base))}")
strip = lambda t: A.norm(re.sub(r"\s*\([^)]*\)\s*$", "", t))
def q2words(n, g): return A.content_words(n.subqs[1].replace("#1", "")) | A.content_words(g["question"])
def bridge_best(n, g, exclude):
    b = n.answers[0]
    if not b: return None
    pool = [p for p in n.ranked(1) if p.doc_id not in exclude and A.mentions(p, b)]
    if not pool: return None
    w = q2words(n, g)
    return max(enumerate(pool), key=lambda it: (A.overlap(it[1], w), -it[0]))[1].doc_id
def v1(g, n, r):
    c = list(r["citations"]); b, fin = n.answers[0], r["answer"]
    if len(c) == 2 and b:
        own = next((p.doc_id for p in n.ranked(1) if strip(p.title) == A.norm(b) and A.mentions(p, fin) and p.doc_id != c[0]), None)
        if own: c[1] = own
    return c
def v2(g, n, r):
    c = list(r["citations"]); fin = r["answer"]
    if len(c) == 2 and not A.mentions(n.passages[c[1]], fin):
        alt = bridge_best(n, g, {c[0]})
        if alt: c[1] = alt
    return c
def v3(g, n, r):
    c = list(r["citations"]); fin = r["answer"]
    if len(c) == 2 and not A.mentions(n.passages[c[1]], fin):
        alt = bridge_best(n, g, set(c))
        if alt: c = c + [alt]
    return c
strat("current", lambda g, n, r: r["citations"])
strat("V1 bridge's own article as answer citation (if it has the answer)", v1)
strat("V2 unsupported answer citation -> bridge passage matching Q2", v2)
strat("V3 unsupported answer citation -> add that as 3rd", v3)
