"""Offline retrieval comparison (no LLM): recall of the gold hop-1 / hop-2 paragraphs within the passages an
agent would SHOW the LLM, using the agent's real sub-questions (from a server run's plan outputs) and either
the gold bridge or the agent's own bridge.

    python3 dev/exp/retrieval.py runs/replay_agent
"""
import json, os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from a4.data import SearchEngine, read_jsonl
from a4.evaluate import normalize_answer as N

STOP_Q = re.compile(r"\b(what|which|who|whom|whose|when|where|why|how|is|are|was|were|did|does|do|the|of|in|a|an)\b", re.I)


def rr(lists):  # round robin (current agent)
    out = []
    for i in range(max(map(len, lists))):
        for L in lists:
            if i < len(L) and L[i] not in out: out.append(L[i])
    return out


def rrf(lists, c=3):
    sc = {}
    for L in lists:
        for i, d in enumerate(L): sc[d] = sc.get(d, 0) + 1 / (c + i + 1)
    return sorted(sc, key=lambda d: -sc[d])


def main():
    run = sys.argv[1]
    eng = SearchEngine.load("data")
    S = lambda q, k: [p.doc_id for p in eng.search(q, k)]
    def T(d): i = eng.corpus.index_of[d]; return eng.corpus.titles[i] + " " + eng.corpus.texts[i]
    has = lambda d, a: bool(N(a)) and f" {N(a)} " in f" {N(T(d))} "
    tr = {t["id"]: t for t in read_jsonl(f"{run}/traces.jsonl")}
    rows = []
    for g in read_jsonl("data/dev.jsonl"):
        golds = [g["answer"]] + g["answer_aliases"]; gids = g["supporting_doc_ids"]
        h2 = next((d for d in gids if any(has(d, a) for a in golds)), gids[1]); h1 = next(d for d in gids if d != h2)
        calls = tr[g["id"]]["calls"]
        plan = calls[0]["output"]
        q1 = re.search(r"Q1\s*:\s*(.+)", plan); q2 = re.search(r"Q2\s*:\s*(.+)", plan)
        ok = q1 and q2 and "#1" in q2.group(1)
        q1 = q1.group(1).strip() if ok else g["question"]; q2 = q2.group(1).strip() if ok else g["question"]
        rd = [c["output"] for c in calls if c["type"] == "llm" and "one step towards" in c.get("prompt", "")]
        ab = ""
        for o in rd:
            m = re.search(r"Answer\s*:\s*(.+)", o)
            if m and "unknown" not in m.group(1).lower(): ab = m.group(1).strip(" .")
        rows.append(dict(q=g["question"], q1=q1, q2=q2, gb=g["decomposition"][0]["answer"], ab=ab, h1=h1, h2=h2))
    def rate(fn, which):
        return sum(r[which] in fn(r) for r in rows) / len(rows)
    print("HOP 1 (passages shown)")
    print("  current  Q1(6)+Q(4) rr top10     ", round(rate(lambda r: rr([S(r["q1"], 6), S(r["q"], 4)])[:10], "h1"), 3))
    print("  proposed Q1(10)+Q(10) rrf top10  ", round(rate(lambda r: rrf([S(r["q1"], 10), S(r["q"], 10)])[:10], "h1"), 3))
    print("  proposed Q1(10)+Q(10) rrf top12  ", round(rate(lambda r: rrf([S(r["q1"], 10), S(r["q"], 10)])[:12], "h1"), 3))
    for bkey, name in (("gb", "gold bridge"), ("ab", "agent bridge")):
        print(f"HOP 2 with {name}")
        sub = lambda r: r["q2"].replace("#1", r[bkey])
        print("  current  Q2b(6)+b(4) rr top10        ", round(rate(lambda r: rr([S(sub(r), 6), S(r[bkey] or r["q"], 4)])[:10], "h2"), 3))
        for n in (10, 12, 15):
            print(f"  proposed Q2b(10)+bQ(10) rrf top{n:2d}    ", round(rate(lambda r: rrf([S(sub(r), 10), S(r[bkey] + ' ' + r['q'], 10)])[:n], "h2"), 3))
        print("  proposed + stripped wh-words top15    ", round(rate(lambda r: rrf([S(STOP_Q.sub(' ', sub(r)), 10), S(r[bkey] + ' ' + STOP_Q.sub(' ', r['q']), 10)])[:15], "h2"), 3))


if __name__ == "__main__":
    main()
