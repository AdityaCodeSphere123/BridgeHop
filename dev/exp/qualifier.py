"""Offline: does a qualified bridge ("Lynn, Massachusetts" from the hop-1 passage) improve hop-2 retrieval?"""
import json, os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from a4.data import SearchEngine
from a4.evaluate import normalize_answer as N
sys.path.insert(0, "dev/exp"); import retrieval as R
eng = SearchEngine.load("data")
S = lambda q, k: [p.doc_id for p in eng.search(q, k)]
G = {json.loads(l)["id"]: json.loads(l) for l in open("data/dev.jsonl")}
T = {json.loads(l)["id"]: json.loads(l) for l in open("runs/replay_agent/traces.jsonl")}
W = re.compile(r"\b(what|which|who|whom|whose|when|where|why|how|is|are|was|were|did|does|do)\b", re.I)
kw = lambda q: " ".join(W.sub(" ", q).replace("?", " ").split())
def T_(d): i = eng.corpus.index_of[d]; return eng.corpus.titles[i] + " " + eng.corpus.texts[i]
QUAL = lambda b: re.compile(re.escape(b) + r",\s+((?:[A-Z][\w.'\-]*)(?:\s+[A-Z][\w.'\-]*){0,2})")
n = changed = base_hit = new_hit = 0; ex = []
for qid, t in T.items():
    g = G[qid]; golds = [g["answer"]] + g["answer_aliases"]; gids = g["supporting_doc_ids"]
    has = lambda d, a: bool(N(a)) and f" {N(a)} " in f" {N(T_(d))} "
    h2 = next((d for d in gids if any(has(d, a) for a in golds)), gids[1])
    reads1 = [c for c in t["calls"] if c["type"] == "llm" and "one step towards" in c["prompt"]]
    hop2 = [c for c in t["calls"] if c["type"] == "llm" and "We already know that #1" in c["prompt"]]
    if not reads1 or not hop2: continue
    r = next((c for c in reads1 if re.search(r"Answer\s*:\s*(?!unknown)(.+)", c["output"], re.I)), None)
    if r is None: continue
    b = re.search(r"Answer\s*:\s*(.+)", r["output"]).group(1).strip(" .")
    m = re.search(r"Passage\s*:\s*\[?(\d+)", r["output"])
    lines = [l for l in r["prompt"].splitlines() if re.match(r"^\[\d+\] ", l)]
    text = lines[int(m.group(1)) - 1] if m and int(m.group(1)) <= len(lines) else ""
    q2 = re.search(r"\nQuestion: (.+)", hop2[0]["prompt"].split("Passages:")[1]).group(1)  # Q2 with bridge
    qm = QUAL(b).search(text)
    n += 1
    base = R.rrf([S(kw(q2), 10), S(b + " " + kw(g["question"]), 10)])[:12]
    base_hit += h2 in base
    if qm and qm.group(1) not in b:
        qb = f"{b}, {qm.group(1)}"
        new = R.rrf([S(kw(q2.replace(b, qb)), 10), S(qb + " " + kw(g["question"]), 10)])[:12]
        changed += 1; new_hit += h2 in new
        ex.append((b, qb, h2 in base, h2 in new))
    else:
        new_hit += h2 in base
print(f"questions {n}, qualified bridges {changed}; hop-2 recall: plain {base_hit/n:.3f} -> qualified {new_hit/n:.3f}")
from collections import Counter
print(Counter((e[2], e[3]) for e in ex))
for e in ex[:25]: print(e)
