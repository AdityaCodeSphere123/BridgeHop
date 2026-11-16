import os, sys, json
ROOT = "/Users/aditya/Desktop/classes/COL764-Assignment4"
sys.path.insert(0, ROOT + "/dev/exp/r10"); sys.path.insert(0, ROOT); os.chdir(ROOT)
import ablate
from a4.data import SearchEngine, read_jsonl
from a4.evaluate import answer_scores
from a4.tools import Tools
gold = list(read_jsonl("data/dev.jsonl")); engine = SearchEngine.load("data")
os.environ["AGENT"] = sys.argv[1]; A = ablate.load_agent()
cap = {}
orig_answer = A.answer
def answer(q, notes, tools):
    r = orig_answer(q, notes, tools); left = tools.budget_left()
    cap[q] = dict(notes=notes, llm=left["llm_calls"], search=left["search_calls"]); return r
A.answer = answer
groups = {}
for g in gold:
    t = Tools(engine, ablate.CacheLLM()); r = A.run(g["question"], t)
    f1 = answer_scores(r["answer"], [g["answer"]] + g.get("answer_aliases", []))[1]
    c = cap[g["question"]]; n = c["notes"]
    if not n.answers[1]:
        key = "no hop-2 answer"
    else:
        cited = n.cited[1]
        has_b = cited is not None and A.mentions(cited, n.answers[0])
        key = f"hop-2 cited passage {'MENTIONS' if has_b else 'LACKS'} bridge" + (" | spare llm+search" if c["llm"] >= 1 and c["search"] >= 1 else " | no spare")
    groups.setdefault(key, []).append(f1)
for k, v in sorted(groups.items()):
    print(f"{k:60s} n={len(v):3d} meanF1={sum(v)/len(v):.2f} F1=0: {sum(1 for x in v if x == 0)}")
