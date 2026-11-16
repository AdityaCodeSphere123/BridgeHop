"""Design-section statistics for the final agent, from a cache replay (0 tokens)."""
import os, sys, re, importlib.util
ROOT = "/Users/aditya/Desktop/classes/COL764-Assignment4"
sys.path.insert(0, ROOT + "/dev/exp/r10"); sys.path.insert(0, ROOT); os.chdir(ROOT)
import ablate
from a4.data import SearchEngine, read_jsonl
from a4.evaluate import answer_scores
from a4.tools import Tools
spec = importlib.util.spec_from_file_location("an", "dev/analyze.py"); an = importlib.util.module_from_spec(spec); spec.loader.exec_module(an)
gold = list(read_jsonl("data/dev.jsonl")); engine = SearchEngine.load("data")
texts = an.load_corpus_texts("data/corpus.jsonl", {d for g in gold for d in g["supporting_doc_ids"]})
os.environ["AGENT"] = "agent.py"; A = ablate.load_agent()
rec = {}
orig_call, orig_read, orig_weak, orig_answer = A.call_llm, A.read, A.recheck_weak_bridge, A.answer
def call_llm(tools, build, passages, max_tokens, limit):
    out, shown = orig_call(tools, build, passages, max_tokens, limit)
    rec.setdefault("calls", []).append((shown, out)); return out, shown
def read(q, notes, tools):
    first_h1 = notes.hop == 0 and notes.attempts[0] == 0 and "h1" not in rec
    n0 = len(rec.get("calls", []))
    r = orig_read(q, notes, tools)
    if first_h1 and len(rec.get("calls", [])) > n0:
        rec["h1"] = rec["calls"][n0]
    return r
def weak(q, notes, tools, context, subq, ans, cited):
    rec["weak"] = ans; return orig_weak(q, notes, tools, context, subq, ans, cited)
def answer(q, notes, tools):
    rec["notes"] = notes; return orig_answer(q, notes, tools)
A.call_llm, A.read, A.recheck_weak_bridge, A.answer = call_llm, read, weak, answer
plan_f1, noplan_f1, pos, weak_ok = [], [], {}, []
for g in gold:
    rec.clear()
    r = A.run(g["question"], Tools(engine, ablate.CacheLLM()))
    golds = [g["answer"]] + list(g.get("answer_aliases", []))
    f1 = answer_scores(r["answer"], golds)[1]
    n = rec["notes"]
    (noplan_f1 if A.norm(n.subqs[0]) == A.norm(g["question"]) else plan_f1).append(f1)
    gb = an.gold_bridge(g)
    hop2 = next((d for d in g["supporting_doc_ids"] if any(an.contains(texts.get(d, ""), a) for a in golds)), None)
    hop1 = next((d for d in g["supporting_doc_ids"] if d != hop2), None) if hop2 else None
    if "h1" in rec and gb and hop1:
        shown, out = rec["h1"]
        m = re.search(r"Answer\s*:\s*(.+)", out); a1 = A.clean_answer(m.group(1) if m else out)
        ids = [p.doc_id for p in shown]
        k = ids.index(hop1) if hop1 in ids else None
        pos.setdefault(k if k is None or k < 2 else "2+", []).append(answer_scores(a1, [gb])[1] >= 0.5)
    if "weak" in rec and gb:
        weak_ok.append(answer_scores(rec["weak"], [gb])[1] >= 0.5)
m = lambda v: sum(v) / max(1, len(v))
print(f"plan used: n={len(plan_f1)} F1={m(plan_f1):.2f} | no usable plan: n={len(noplan_f1)} F1={m(noplan_f1):.2f}")
for k in [0, 1, "2+", None]:
    v = pos.get(k, []); print(f"first hop-1 read, gold hop-1 passage at {k}: n={len(v)} bridge right {100*m(v):.0f}%")
print(f"weak bridge (cited passage neither first nor named article): n={len(weak_ok)} right {100*m(weak_ok):.0f}%")
