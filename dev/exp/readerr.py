import json, os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from a4.data import read_jsonl
from a4.evaluate import answer_scores, normalize_answer as N
run = sys.argv[1]
gold = {g["id"]: g for g in read_jsonl("data/dev.jsonl")}
P = {r["id"]: r for r in read_jsonl(f"{run}/predictions.jsonl")}
T = {t["id"]: t for t in read_jsonl(f"{run}/traces.jsonl")}
n = 0
for qid, p in P.items():
    g = gold[qid]; golds = [g["answer"]] + g["answer_aliases"]
    ret = {d for c in T[qid]["calls"] if c["type"] == "search" for d in c["doc_ids"]}
    if not set(g["supporting_doc_ids"]) <= ret: continue
    if answer_scores(p["answer"], golds)[1] >= 0.5: continue
    outs = [c["output"].replace("\n", " | ")[:70] for c in T[qid]["calls"] if c["type"] == "llm"]
    print(f"Q: {g['question']}\n   gold: {g['decomposition'][0]['answer']} -> {g['answer']}   pred: {p['answer']}\n   llm: {outs}")
    n += 1
print(n)
