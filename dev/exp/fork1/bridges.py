import os, sys, json, re
ROOT = "/Users/aditya/Desktop/classes/COL764-Assignment4"
sys.path.insert(0, ROOT + "/dev/exp/r10"); sys.path.insert(0, ROOT); os.chdir(ROOT)
import ablate
from a4.data import SearchEngine, read_jsonl
from a4.evaluate import answer_scores
from a4.tools import Tools
gold = list(read_jsonl("data/dev.jsonl")); engine = SearchEngine.load("data")
os.environ["AGENT"] = sys.argv[1]; A = ablate.load_agent()
cap = {}; orig = A.answer
def answer(q, notes, tools):
    cap[q] = notes; return orig(q, notes, tools)
A.answer = answer
rows = []
for g in gold:
    r = A.run(g["question"], Tools(engine, ablate.CacheLLM())); n = cap[g["question"]]
    f1 = answer_scores(r["answer"], [g["answer"]] + g.get("answer_aliases", []))[1]
    rows.append(dict(q=g["question"], b=n.answers[0], gb=g["decomposition"][0]["answer"], q1=n.subqs[0], f1=f1, mem=n.memory_bridge,
                     bok=answer_scores(n.answers[0], [g["decomposition"][0]["answer"]])[1]))
json.dump(rows, open("dev/exp/fork1/bridges.json", "w"))
desc = [r for r in rows if r["b"] and not re.search(r"[A-Z0-9]", r["b"])]
print(f"bridges with no capital letter or digit: {len(desc)}, mean F1 {sum(r['f1'] for r in desc)/max(1,len(desc)):.2f}, bridge right {sum(r['bok']>0.5 for r in desc)}")
for r in desc: print("   ", round(r["f1"],2), "| bridge:", r["b"], "| gold:", r["gb"], "| Q1:", r["q1"][:60])
