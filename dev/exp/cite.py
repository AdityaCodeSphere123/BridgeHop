import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from a4.data import read_jsonl
from a4.evaluate import evidence_f1, answer_scores
run = sys.argv[1]
gold = {g["id"]: g for g in read_jsonl("data/dev.jsonl")}
P = {r["id"]: r for r in read_jsonl(f"{run}/predictions.jsonl")}
T = {t["id"]: t for t in read_jsonl(f"{run}/traces.jsonl")}
pol = {"cur": 0, "add_top_lastsearch": 0, "add_top_firstsearch": 0, "only_first": 0, "add_top_any_unc": 0}
for qid, p in P.items():
    g = gold[qid]["supporting_doc_ids"]; c = p["citations"]
    srch = [x["doc_ids"] for x in T[qid]["calls"] if x["type"] == "search"]
    pol["cur"] += evidence_f1(c, g)
    pol["only_first"] += evidence_f1(c[:1], g)
    def add(L):
        for d in L:
            if d not in c: return c + [d]
        return c
    pol["add_top_lastsearch"] += evidence_f1(add(srch[-1]), g)
    pol["add_top_firstsearch"] += evidence_f1(add(srch[0]), g)
for k, v in pol.items(): print(f"{k:22s} {v/len(P):.4f}")
