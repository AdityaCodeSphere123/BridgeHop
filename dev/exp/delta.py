"""Per-question differences between two replay/run folders: python3 dev/exp/delta.py runs/A runs/B"""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from a4.evaluate import answer_scores, evidence_f1
G = {json.loads(l)["id"]: json.loads(l) for l in open("data/dev.jsonl")}
def load(r): return {json.loads(l)["id"]: json.loads(l) for l in open(f"{r}/predictions.jsonl")}
A, B = load(sys.argv[1]), load(sys.argv[2])
for qid in A:
    g = G[qid]; golds = [g["answer"]] + g["answer_aliases"]
    fa, fb = answer_scores(A[qid]["answer"], golds)[1], answer_scores(B[qid]["answer"], golds)[1]
    ea, eb = evidence_f1(A[qid]["citations"], g["supporting_doc_ids"]), evidence_f1(B[qid]["citations"], g["supporting_doc_ids"])
    if abs(fa - fb) > 1e-9 or abs(ea - eb) > 1e-9:
        print(f"F1 {fa:.2f}->{fb:.2f} Ev {ea:.2f}->{eb:.2f} gold={g['answer']!r} | {A[qid]['answer'][:50]!r} -> {B[qid]['answer'][:50]!r}")
