"""Score runs on the same question subset: python dev/exp/sub.py runs/v1b runs/b [--n 150]"""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from a4.data import read_jsonl
from a4.evaluate import score
args = sys.argv[1:]
n = int(args[args.index("--n") + 1]) if "--n" in args else None
runs = [a for a in args if a.startswith("runs/")]
gold = read_jsonl("data/dev.jsonl")
ids = set.intersection(*[{r["id"] for r in read_jsonl(f"{r}/predictions.jsonl")} for r in runs])
g = [x for x in gold if x["id"] in ids][: n or None]
for r in runs:
    m = score(read_jsonl(f"{r}/predictions.jsonl"), g)
    print(f"{r:22s} n={m['n']} S={m['S']:.4f} F1={m['answer_f1']:.4f} EM={m['answer_em']:.4f} EvF1={m['evidence_f1']:.4f} llm={m['mean_llm_calls']}")
