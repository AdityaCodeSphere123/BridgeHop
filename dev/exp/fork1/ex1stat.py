import os, sys, importlib.util
ROOT = "/Users/aditya/Desktop/classes/COL764-Assignment4"; sys.path.insert(0, ROOT); os.chdir(ROOT)
spec = importlib.util.spec_from_file_location("an", "dev/analyze.py"); an = importlib.util.module_from_spec(spec); spec.loader.exec_module(an)
from a4.data import read_jsonl
from a4.evaluate import answer_scores
gold = {g["id"]: g for g in read_jsonl("data/dev.jsonl")}
for run in sys.argv[1:]:
    preds = {r["id"]: r for r in read_jsonl(f"{run}/predictions.jsonl")}
    traces = {t["id"]: t for t in read_jsonl(f"{run}/traces.jsonl")}
    texts = an.load_corpus_texts("data/corpus.jsonl", {d for i in preds for d in gold[i]["supporting_doc_ids"]})
    rows = []
    for qid, r in preds.items():
        g, t = gold[qid], traces[qid]; golds = [g["answer"]] + list(g.get("answer_aliases", []))
        f1 = answer_scores(r["answer"], golds)[1]
        ret = {d for c in t["calls"] if c["type"] == "search" for d in c["doc_ids"]}
        hop2 = next((d for d in g["supporting_doc_ids"] if any(an.contains(texts.get(d, ""), a) for a in golds)), None)
        gb, ab = an.gold_bridge(g), an.agent_bridge(t)
        rows.append((answer_scores(ab, [gb])[1] >= 0.5 if gb else None, (hop2 in ret) if hop2 else None, f1, r["answer"], qid))
    bad = [x for x in rows if x[0] is False and not x[1]]; good = [x for x in rows if x[0] and x[1]]
    print(f"{run}: wrong bridge & no hop-2 para F1={sum(x[2] for x in bad)/len(bad):.2f} n={len(bad)} | right bridge & para F1={sum(x[2] for x in good)/len(good):.2f} n={len(good)}")
