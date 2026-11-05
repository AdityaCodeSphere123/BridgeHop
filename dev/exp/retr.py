"""Retrieval-only experiments (no LLM): recall of gold paragraphs under different query strategies."""
import json, os, sys, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from a4.data import SearchEngine, read_jsonl
from a4.evaluate import normalize_answer as N

eng = SearchEngine.load("data")
dev = read_jsonl("data/dev.jsonl")
text = {}
def T(d):
    i = eng.corpus.index_of[d]; return eng.corpus.titles[i] + " " + eng.corpus.texts[i]
def has(d, a): a = N(a); return a and f" {a} " in f" {N(T(d))} "
rows = []
for g in dev:
    golds = [g["answer"]] + g["answer_aliases"]
    gids = g["supporting_doc_ids"]
    h2 = next((d for d in gids if any(has(d, a) for a in golds)), gids[1])
    h1 = next(d for d in gids if d != h2)
    dec = g["decomposition"]
    rows.append(dict(q=g["question"], q1=dec[0]["question"], b=dec[0]["answer"],
                     q2=dec[1]["question"].replace("#1", dec[0]["answer"]), h1=h1, h2=h2))
def rec(fn, k, which):
    hit = 0
    for r in rows:
        ids = {p.doc_id for p in eng.search(fn(r), k)}
        hit += r[which] in ids
    return hit / len(rows)
for k in (5, 10):
    print("k", k)
    print("  Q -> h1", rec(lambda r: r["q"], k, "h1"), " Q -> h2", rec(lambda r: r["q"], k, "h2"))
    print("  goldQ1 -> h1", rec(lambda r: r["q1"], k, "h1"))
    print("  goldQ2(bridge) -> h2", rec(lambda r: r["q2"], k, "h2"))
    print("  bridge -> h2", rec(lambda r: r["b"], k, "h2"))
    print("  bridge+Q -> h2", rec(lambda r: r["b"] + " " + r["q"], k, "h2"))
    print("  Q2+Q -> h2", rec(lambda r: r["q2"] + " " + r["q"], k, "h2"))
# how often does h1 title equal / contain something
print("h1 title in Q:", sum(N(eng.corpus.titles[eng.corpus.index_of[r['h1']]]) in N(r['q']) for r in rows)/len(rows))
print("h2 title == bridge:", sum(N(eng.corpus.titles[eng.corpus.index_of[r['h2']]]) == N(r['b']) for r in rows)/len(rows))
print("h2 title contains bridge:", sum(N(r['b']) in N(eng.corpus.titles[eng.corpus.index_of[r['h2']]]) for r in rows)/len(rows))

if len(sys.argv) > 1:
    n = 0
    for r in rows:
        ids = [p.doc_id for p in eng.search(r["q2"], 10)]
        if r["h2"] not in ids:
            i = eng.corpus.index_of[r["h2"]]
            print("Q :", r["q"]); print("Q2:", r["q2"]); print("H2:", eng.corpus.titles[i], "|", eng.corpus.texts[i][:250]); print()
            n += 1
            if n > 25: break
