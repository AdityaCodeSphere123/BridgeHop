"""F1 by where the final answer came from (cache replay, 0 tokens)."""
import os, sys, re, collections
ROOT = "/Users/aditya/Desktop/classes/COL764-Assignment4"
sys.path.insert(0, ROOT + "/dev/exp/r10"); sys.path.insert(0, ROOT); os.chdir(ROOT)
import ablate
from a4.data import SearchEngine, read_jsonl
from a4.evaluate import answer_scores
from a4.tools import Tools
gold = list(read_jsonl("data/dev.jsonl")); engine = SearchEngine.load("data")
os.environ["AGENT"] = "agent.py"; A = ablate.load_agent()
info = {}
orig_answer = A.answer
def answer(q, notes, tools):
    left = tools.budget_left()
    src = "hop-2 read" if notes.answers[1] else ("hop-2 none, bridge known" if notes.answers[0] else "no bridge")
    info[q] = (src, left["llm_calls"], left["search_calls"], notes.attempts[:], notes.memory_bridge)
    return orig_answer(q, notes, tools)
A.answer = answer
groups = collections.defaultdict(list)
for g in gold:
    r = A.run(g["question"], Tools(engine, ablate.CacheLLM()))
    f1 = answer_scores(r["answer"], [g["answer"]] + g.get("answer_aliases", []))[1]
    src, llm, srch, att, mb = info[g["question"]]
    groups[src].append(f1)
    groups[f"{src} | hop-2 retries={att[1]}"].append(f1)
    if mb: groups["bridge from memory"].append(f1)
    groups[f"{src} | spare llm calls at answer()={llm}"].append(f1)
for k in sorted(groups):
    v = groups[k]; print(f"{k:55s} n={len(v):3d} meanF1={sum(v)/len(v):.2f}")
