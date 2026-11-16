import os, sys, re
ROOT = "/Users/aditya/Desktop/classes/COL764-Assignment4"
sys.path.insert(0, ROOT + "/dev/exp/r10"); sys.path.insert(0, ROOT); os.chdir(ROOT)
import ablate
from a4.data import SearchEngine, read_jsonl
from a4.evaluate import answer_scores
from a4.tools import Tools
gold = list(read_jsonl("data/dev.jsonl")); engine = SearchEngine.load("data")
os.environ["AGENT"] = "agent.py"; A = ablate.load_agent()
outs = {}; orig_llm = None
cap = {}; orig_answer = A.answer
def answer(q, notes, tools):
    cap[q] = notes; return orig_answer(q, notes, tools)
A.answer = answer
for g in gold:
    t = Tools(engine, ablate.CacheLLM()); r = A.run(g["question"], t)
    n = cap[g["question"]]
    if A.norm(n.subqs[0]) != A.norm(g["question"]): continue
    f1 = answer_scores(r["answer"], [g["answer"]] + g.get("answer_aliases", []))[1]
    plans = [c["output"].replace("\n", " | ")[:150] for c in t.trace if c["type"] == "llm" and c["prompt"].startswith(("Split", "Each question"))]
    print(f"F1 {f1:.2f} | {g['question'][:80]}")
    for p in plans:
        q1 = re.search(r"Q1\s*:\s*(.+?)(?:\||$)", p); 
        why = []
        out = p.replace(" | ", "\n")
        parsed = A.parse_plan(out, g["question"])
        if q1 and A.NOT_AN_ENTITY.match(q1.group(1)): why.append("NOT_AN_ENTITY")
        elif parsed is None: why.append("parse_plan rejects")
        else:
            if A.wrong_place_type(g["question"], parsed[0]): why.append("wrong_place_type")
            if A.loses_entity(g["question"], parsed[0]): why.append("loses_entity")
        print(f"     plan: {p[:130]}  -> {why}")
