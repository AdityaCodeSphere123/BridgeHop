"""Replay the strict agent from cache (0 tokens), record each plan, and test plan-quality checks."""
import os, sys, re, json
ROOT = "/Users/aditya/Desktop/classes/COL764-Assignment4"
sys.path.insert(0, ROOT + "/dev/exp/r10"); sys.path.insert(0, ROOT); os.chdir(ROOT)
import ablate
from a4.data import SearchEngine, read_jsonl
from a4.evaluate import answer_scores
from a4.tools import Tools
gold = list(read_jsonl("data/dev.jsonl")); engine = SearchEngine.load("data")
os.environ["AGENT"] = sys.argv[1] if len(sys.argv) > 1 else "agent.py"
A = ablate.load_agent()
rows = []
for g in gold:
    t = Tools(engine, ablate.CacheLLM()); n = A.Notes(g["question"]); A.plan(g["question"], n, t)
    r = A.run(g["question"], Tools(engine, ablate.CacheLLM()))
    f1 = answer_scores(r["answer"], [g["answer"]] + g.get("answer_aliases", []))[1]
    rows.append(dict(id=g["id"], q=g["question"], q1=n.subqs[0], q2=n.subqs[1], gb=g["decomposition"][0]["answer"], f1=f1))
json.dump(rows, open("dev/exp/fork1/plans.json", "w"), indent=0)
def ents(q):  # capitalised phrases not at the start of the question
    words = q.rstrip("?").split()
    out, cur = [], []
    for i, w in enumerate(words):
        if i > 0 and w[:1].isupper() or (cur and w.lower() in {"of", "the", "and", "&", "de", "in"}):
            cur.append(w)
        else:
            if cur: out.append(" ".join(cur).strip(" ,'\"")); cur = []
    if cur: out.append(" ".join(cur).strip(" ,'\""))
    return [e for e in out if e and e[:1].isupper()]
bad1 = [r for r in rows if ents(r["q"]) and not any(A.norm(e) in A.norm(r["q1"]) or any(w in A.norm(r["q1"]).split() for w in A.norm(e).split() if len(w) > 3) for e in ents(r["q"]))]
bad2 = [r for r in rows if re.match(r"\s*(who|what) (is|was|are|were) [^?]*\?$", r["q1"], re.I) and len(r["q1"].split()) <= 5]
for name, b in (("Q1 drops every named entity", bad1), ("Q1 is a bare 'who is X?'", bad2)):
    print(f"\n{name}: {len(b)} questions, mean F1 {sum(r['f1'] for r in b)/max(1,len(b)):.2f}")
    for r in b[:12]: print("   ", round(r["f1"], 2), "|", r["q"][:70], "|| Q1:", r["q1"][:60], "|| gold bridge:", r["gb"])
