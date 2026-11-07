"""Query2doc / HyDE retrieval simulation for hop 2 (needs the LLM for the pseudo-passages, cached).

For each question, take the agent's hop-2 sub-question (bridge filled in) and hop-1 searches from a run,
ask the LLM for a short hypothetical passage, and count how often the gold hop-2 paragraph is among the
12 passages the agent would show, for several ways of adding that passage to the BM25 queries.

    python3 dev/exp/q2d_sim.py runs/S_m2final
"""
import json, os, re, sys, importlib.util
from concurrent.futures import ThreadPoolExecutor
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "dev", "exp")); os.chdir(ROOT)
import crun  # patches HTTPLLM with the disk cache
import a4.llm as L
from a4.config import load_config
from a4.data import SearchEngine
from a4.evaluate import normalize_answer as N
spec = importlib.util.spec_from_file_location("ag", "agent.py"); A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
llm = L.HTTPLLM(load_config())
eng = SearchEngine.load("data")
gold = {g["id"]: g for g in map(json.loads, open("data/dev.jsonl"))}
corpus = {}
for l in open("data/corpus.jsonl"):
    d = json.loads(l); corpus[d["doc_id"]] = d
T = {t["id"]: t for t in map(json.loads, open(sys.argv[1] + "/traces.jsonl"))}
PSEUDO = """Write a short Wikipedia-style paragraph (two or three sentences) that answers the question.
Question: {q}
Paragraph:"""
rows = []
for i, t in T.items():
    calls = t["calls"]; llmc = [c for c in calls if c["type"] == "llm"]
    h2 = [c for c in llmc if c["prompt"].startswith("Answer the question using the passages below.\n(Main")]
    if not h2: continue
    subq = re.search(r"\nQuestion: (.*)\n", h2[0]["prompt"]).group(1)
    bridge = re.search(r"#1 = (.*?)\.\)", h2[0]["prompt"]).group(1)
    h1q = []
    for c in calls:
        if c["type"] == "llm" and c is h2[0]: break
        if c["type"] == "search": h1q.append(c["query"])
    h1q = h1q[:2]
    g = gold[i]; sd = g["supporting_doc_ids"]; ga = N(g["answer"])
    d2 = [d for d in sd if ga and ga in N(corpus[d]["title"] + " " + corpus[d]["text"])]
    rows.append(dict(id=i, subq=subq, bridge=bridge, h1q=h1q, d2=d2[0] if d2 else sd[-1], q=g["question"],
                     bridge_ok=N(bridge) == N(g["decomposition"][0]["answer"]) or N(g["decomposition"][0]["answer"]) in N(bridge)))
def gen(r):
    try: return llm.complete(PSEUDO.format(q=r["subq"]), 96)[0]
    except Exception as e: return ""
with ThreadPoolExecutor(8) as ex: pseudo = list(ex.map(gen, rows))
cache = {}
def S(q):
    if q not in cache: cache[q] = eng.search(q, 10)
    return cache[q]
KW = A.keywords
V = {
  "current [kw(Q2), b+kw(Q)]":            lambda r, p: [KW(r["subq"]), f"{r['bridge']} {KW(r['q'])}"],
  "Q2D [kw(Q2), Q2 x2 + pseudo]":          lambda r, p: [KW(r["subq"]), f"{KW(r['subq'])} {KW(r['subq'])} {p}"],
  "Q2D [kw(Q2)+pseudo, b+kw(Q)]":          lambda r, p: [f"{KW(r['subq'])} {p}", f"{r['bridge']} {KW(r['q'])}"],
  "current + 3rd query pseudo":            lambda r, p: [KW(r["subq"]), f"{r['bridge']} {KW(r['q'])}", f"{r['bridge']} {p}"],
  "current + 3rd query b+Q2+pseudo":       lambda r, p: [KW(r["subq"]), f"{r['bridge']} {KW(r['q'])}", f"{KW(r['subq'])} {p}"],
}
for name, fn in V.items():
    for subset in ("all", "bridge ok"):
        rs = [(r, p) for r, p in zip(rows, pseudo) if subset == "all" or r["bridge_ok"]]
        found = shown = 0
        for r, p in rs:
            notes = A.Notes(r["q"])
            for q in r["h1q"]: notes.add(q, S(q))
            notes.hop_of_query = [0] * len(notes.hop_of_query); notes.hop = 1
            for q in fn(r, p): notes.add(q, S(q))
            ids = [x.doc_id for x in notes.ranked(1)]
            found += r["d2"] in notes.passages; shown += r["d2"] in ids[:12]
        print(f"{name:36s} {subset:9s} n={len(rs):3d}  found {found/len(rs):.3f}  in top-12 {shown/len(rs):.3f}")
print("example pseudo:", rows[0]["subq"], "->", pseudo[0][:200])
