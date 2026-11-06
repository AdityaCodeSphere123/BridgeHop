"""Where does the gold paragraph land in the passage order the reader sees? (no LLM)
Uses the agent's real plans and bridges from a recorded server run."""
import json, os, re, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import retrieval as R
ROOT = R.ROOT
from a4.data import SearchEngine, read_jsonl
from a4.evaluate import normalize_answer as N


def main():
    eng = SearchEngine.load("data")
    S = lambda q, k: [p.doc_id for p in eng.search(q, k)]
    title = lambda d: eng.corpus.titles[eng.corpus.index_of[d]]
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "retrieval.py")).read()
    body = src.split("def main():")[1].split("    def rate(fn, which):")[0]
    ns = {}
    exec("def build():\n" + body + "    return rows\n", R.__dict__, ns)
    rows = ns["build"]()
    W = re.compile(r"\b(what|which|who|whom|whose|when|where|why|how|is|are|was|were|did|does|do)\b", re.I)
    kw = lambda q: " ".join(W.sub(" ", q).replace("?", " ").split())
    strip = lambda t: N(re.sub(r"\s*\([^)]*\)\s*$", "", t))
    t_in_q = lambda d, q: bool(strip(title(d))) and f" {strip(title(d))} " in f" {N(q)} "
    t_has_b = lambda d, b: bool(N(b)) and (f" {N(b)} " in f" {N(title(d))} " or f" {strip(title(d))} " in f" {N(b)} ")
    def boost(order, pred):
        return [d for d in order if pred(d)] + [d for d in order if not pred(d)]
    def first_k(order, k): return order[:k]
    def dist(name, orders, gold_key, k):
        c = Counter()
        for r, o in zip(rows, orders):
            o = o[:k]
            c["pos1" if o[:1] == [r[gold_key]] else ("pos2-3" if r[gold_key] in o[1:3] else ("pos4+" if r[gold_key] in o else "absent"))] += 1
        n = len(rows)
        print(f"{name:42s} " + " ".join(f"{k}={c[k]/n:.3f}" for k in ("pos1", "pos2-3", "pos4+", "absent")))
    h1 = [R.rrf([S(r["q1"], 10), S(r["q"], 10)]) for r in rows]
    dist("hop1 fused (current)", h1, "h1", 10)
    dist("hop1 fused + title-in-question first", [boost(o, lambda d, r=r: t_in_q(d, r["q"])) for o, r in zip(h1, rows)], "h1", 10)
    def first_only(order, pred):
        hit = next((d for d in order if pred(d)), None)
        return order if hit is None else [hit] + [d for d in order if d != hit]
    dist("hop1 fused + best title-in-question first", [first_only(o, lambda d, r=r: t_in_q(d, r["q"])) for o, r in zip(h1, rows)], "h1", 10)
    if "--hop1" in sys.argv: return
    for bk in ("ab", "gb"):
        h2 = [R.rrf([S(kw(r["q2"].replace("#1", r[bk])), 10), S(r[bk] + " " + kw(r["q"]), 10)]) for r in rows]
        dist(f"hop2 fused (current) [{bk}]", h2, "h2", 12)
        ment = lambda d, b: bool(N(b)) and f" {N(b)} " in f" {N(title(d) + ' ' + eng.corpus.texts[eng.corpus.index_of[d]])} "
        dist(f"hop2 fused + mentions-bridge first [{bk}]", [boost(o, lambda d, r=r: ment(d, r[bk])) for o, r in zip(h2, rows)], "h2", 12)
        q2l = [S(kw(r["q2"].replace("#1", r[bk])), 10) for r in rows]
        dist(f"hop2 Q2-list first, then rest [{bk}]", [q + [d for d in o if d not in q] for q, o in zip(q2l, h2)], "h2", 12)
        dist(f"hop2 fused + bridge-titled first [{bk}]", [boost(o, lambda d, r=r: t_has_b(d, r[bk])) for o, r in zip(h2, rows)], "h2", 12)


if __name__ == "__main__":
    main()
