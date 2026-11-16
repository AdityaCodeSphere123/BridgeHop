"""Offline hop-2 retrieval simulation with the final agent's real plans/bridges (0 LLM tokens)."""
import os, sys, json, importlib.util
ROOT = "/Users/aditya/Desktop/classes/COL764-Assignment4"
sys.path.insert(0, ROOT + "/dev/exp/r10"); sys.path.insert(0, ROOT); os.chdir(ROOT)
import ablate
from a4.data import SearchEngine, read_jsonl
from a4.evaluate import answer_scores
from a4.tools import Tools
spec = importlib.util.spec_from_file_location("an", "dev/analyze.py"); an = importlib.util.module_from_spec(spec); spec.loader.exec_module(an)
gold = list(read_jsonl("data/dev.jsonl")); engine = SearchEngine.load("data")
texts = an.load_corpus_texts("data/corpus.jsonl", {d for g in gold for d in g["supporting_doc_ids"]})
os.environ["AGENT"] = "agent.py"; A = ablate.load_agent()
cap = {}; orig = A.next_query
def nq(q, notes, tools):
    if notes.hop == 1 and notes.attempts[1] == 0 and q not in cap:
        cap[q] = (list(notes.subqs), notes.answers[0], [list(h) for h in notes.hits], list(notes.hop_of_query))
    return orig(q, notes, tools)
A.next_query = nq
rows = []
for g in gold:
    A.run(g["question"], Tools(engine, ablate.CacheLLM()))
    if g["question"] not in cap: continue
    subqs, bridge, hits, hops = cap[g["question"]]
    golds = [g["answer"]] + list(g.get("answer_aliases", []))
    hop2 = next((d for d in g["supporting_doc_ids"] if any(an.contains(texts.get(d, ""), a) for a in golds)), None)
    if not hop2: continue
    rows.append(dict(q=g["question"], subqs=subqs, bridge=bridge, hits=hits, hops=hops, hop2=hop2,
                     bok=answer_scores(bridge, [an.gold_bridge(g)])[1] >= 0.5))
json.dump(rows, open("dev/exp/fork1/h2rows.json", "w"))
def ranked(hits, hops, passages_len=12):
    order = []
    for group in (1, 0):
        sc = {}
        for h, hp in zip(hits, hops):
            if hp == group:
                for r, d in enumerate(h): sc[d] = sc.get(d, 0) + 1 / (A.RRF_C + r + 1)
        order += [d for d in sorted(sc, key=lambda d: -sc[d]) if d not in order]
    return order[:passages_len]
S = lambda q: [p.doc_id for p in engine.search(q, 10)]
for name, extra in [("current (Q2 kw, bridge+question)", lambda r: []),
                    ("+ bridge alone", lambda r: [r["bridge"]]),
                    ("+ bridge + Q2 kw", lambda r: [f"{r['bridge']} {A.keywords(r['subqs'][1].replace('#1', ''))}"])]:
    found = pos0 = 0; n = 0; fb = 0; nb = 0
    for r in rows:
        bridge, q = r["bridge"], r["q"]
        sub = r["subqs"][1].replace("#1", bridge)
        qs = ([f"{bridge} {A.keywords(q)}", bridge] if "#1" not in r["subqs"][1] else [A.keywords(sub), f"{bridge} {A.keywords(q)}"]) + extra(r)
        hits = r["hits"] + [S(x) for x in qs]; hops = r["hops"] + [1] * len(qs)
        shown = ranked(hits, hops)
        n += 1; found += r["hop2"] in shown; pos0 += shown[:3].count(r["hop2"])
        if r["bok"]: nb += 1; fb += r["hop2"] in shown
    print(f"{name:36s} hop-2 gold in the 12 shown: {found}/{n} ({100*found/n:.1f}%), in top 3: {pos0} | right-bridge subset: {fb}/{nb} ({100*fb/nb:.1f}%)")
