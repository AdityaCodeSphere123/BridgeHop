import json, os, re, sys, itertools
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from a4.data import SearchEngine, read_jsonl
from a4.evaluate import normalize_answer as N
eng = SearchEngine.load("data")
dev = {g["id"]: g for g in read_jsonl("data/dev.jsonl")}
tr = {t["id"]: t for t in read_jsonl("runs/v1b/traces.jsonl")}
def T(d): i = eng.corpus.index_of[d]; return eng.corpus.titles[i] + " " + eng.corpus.texts[i]
def has(d, a): a = N(a); return a and f" {a} " in f" {N(T(d))} "
rows = []
for qid, g in dev.items():
    golds = [g["answer"]] + g["answer_aliases"]; gids = g["supporting_doc_ids"]
    h2 = next((d for d in gids if any(has(d, a) for a in golds)), gids[1]); h1 = next(d for d in gids if d != h2)
    plan = next(c["output"] for c in tr[qid]["calls"] if c["type"] == "llm")
    q1 = re.search(r"Q1\s*:\s*(.+)", plan); q2 = re.search(r"Q2\s*:\s*(.+)", plan)
    q1 = q1.group(1).strip() if q1 else g["question"]; q2 = q2.group(1).strip() if q2 and "#1" in q2.group(1) else g["question"]
    b = g["decomposition"][0]["answer"]
    rows.append(dict(q=g["question"], q1=q1, q2=q2, q2b=q2.replace("#1", b), b=b, h1=h1, h2=h2))
S = {
 "q2b": lambda r: r["q2b"], "b": lambda r: r["b"], "b+q": lambda r: r["b"] + " " + r["q"],
 "q2b+q": lambda r: r["q2b"] + " " + r["q"], "q2-": lambda r: r["q2"].replace("#1", ""),
 "b+q2b": lambda r: r["b"] + " " + r["q2b"],
}
cache = {}
def res(name, r, k):
    key = (name, r["q"], k)
    if key not in cache: cache[key] = [p.doc_id for p in eng.search(S[name](r), k)]
    return cache[key]
for name in S:
    for k in (5, 10):
        print(f"{name:8s} k={k:2d} h2={sum(r['h2'] in res(name, r, k) for r in rows)/len(rows):.3f}")
print("--- unions (k=10 each)")
for combo in list(itertools.combinations(S, 2)) + list(itertools.combinations(S, 3)):
    hit = sum(any(r["h2"] in res(n, r, 10) for n in combo) for r in rows) / len(rows)
    print(f"{'+'.join(combo):30s} {hit:.3f}")
print("--- hop1")
S["q1"] = lambda r: r["q1"]; S["q"] = lambda r: r["q"]; S["q1+q"] = lambda r: r["q1"] + " " + r["q"]
for name in ("q1", "q", "q1+q"):
    for k in (5, 10): print(f"{name:6s} k={k} h1={sum(r['h1'] in res(name, r, k) for r in rows)/len(rows):.3f}")
print("q1(10)+q(10) union h1", sum(r["h1"] in res("q1", r, 10) + res("q", r, 10) for r in rows)/len(rows))
print("q1(6)+q(4) union h1", sum(r["h1"] in res("q1", r, 6) + res("q", r, 4) for r in rows)/len(rows))
print("q(10) h2 bonus: q1(10)+q(10) union h2", sum(r["h2"] in res("q1", r, 10) + res("q", r, 10) for r in rows)/len(rows))
