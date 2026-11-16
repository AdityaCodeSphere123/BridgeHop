"""What each step of the FINAL agent did on dev (cache replay of agent.py, 0 tokens)."""
import os, sys, re, json, importlib.util, collections
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
ev = []
O = {k: getattr(A, k) for k in ["recheck_weak_bridge", "second_opinion", "memory_check", "answer"]}
def weak(q, notes, tools, context, subq, ans, cited):
    c = tools.budget_left()["llm_calls"]; r = O["recheck_weak_bridge"](q, notes, tools, context, subq, ans, cited)
    if tools.budget_left()["llm_calls"] < c: ev.append(("weak", ans, r[0]))
    return r
def second(q, notes, tools, context, shown, ans, cited, lc):
    mis = cited is not None and not A.mentions(cited, ans)
    c = tools.budget_left()["llm_calls"]; r = O["second_opinion"](q, notes, tools, context, shown, ans, cited, lc)
    if tools.budget_left()["llm_calls"] < c: ev.append(("mis" if mis else "second", ans, r[0]))
    return r
def mem(q, notes, tools, context, asked, jq, ans, cited):
    c = tools.budget_left()["llm_calls"]; r = O["memory_check"](q, notes, tools, context, asked, jq, ans, cited)
    if tools.budget_left()["llm_calls"] < c: ev.append(("memory", ans, r[0]))
    return r
cap = {}
def answer(q, notes, tools):
    cap["notes"] = notes; return O["answer"](q, notes, tools)
A.recheck_weak_bridge, A.second_opinion, A.memory_check, A.answer = weak, second, mem, answer
plan_groups = collections.defaultdict(list); step = collections.defaultdict(lambda: [0, 0, 0, 0, 0.0])  # fired, changed, better, worse
other = collections.defaultdict(list); late = [0, 0]
for g in gold:
    ev.clear(); t = Tools(engine, ablate.CacheLLM()); r = A.run(g["question"], t)
    golds = [g["answer"]] + list(g.get("answer_aliases", [])); gb = an.gold_bridge(g)
    f1 = answer_scores(r["answer"], golds)[1]; n = cap["notes"]; q = g["question"]
    # plan path
    plans = [c["output"] for c in t.trace if c["type"] == "llm" and c["prompt"].startswith(("Split", "Each question"))]
    ps = A.phrase_split(q)
    if A.norm(n.subqs[0]) == A.norm(q): grp = "no split"
    elif ps and list(ps) == n.subqs: grp = "grammar split"
    elif plans and A.parse_plan(plans[0], q) and list(A.parse_plan(plans[0], q)) == n.subqs: grp = "first plan" if len(plans) == 1 else "first plan (kept after a failed retry)"
    else: grp = "second plan prompt"
    plan_groups[grp].append(f1)
    for kind, before, after in ev:
        s = step[kind]; s[0] += 1
        if A.norm(before) != A.norm(after):
            s[1] += 1
            ref = [gb] if kind == "weak" else golds
            b, a = answer_scores(before, ref)[1], answer_scores(after, ref)[1]
            s[2] += a > b; s[3] += a < b; s[4] += (a - b) if kind != 'weak' else 0
    if n.memory_bridge: other["bridge from memory"].append(f1)
    if not n.answers[1]: other["no hop-2 answer: memory guess"].append(f1)
    hop2 = next((d for d in g["supporting_doc_ids"] if any(an.contains(texts.get(d, ""), a) for a in golds)), None)
    if hop2:
        txt = texts[hop2].lower(); pos = min([txt.find(a.lower()) for a in golds if txt.find(a.lower()) >= 0] or [-1])
        late[0] += 1; late[1] += pos > 1200
print("PLAN PATHS"); [print(f"  {k:40s} n={len(v):3d} F1={sum(v)/len(v):.2f}") for k, v in sorted(plan_groups.items(), key=lambda x: -len(x[1]))]
print("STEPS (ran, changed, better, worse, dS)"); [print(f"  {k:10s} ran={v[0]} changed={v[1]} better={v[2]} worse={v[3]} dS={0.8*v[4]/300:+.4f}") for k, v in step.items()]
print("OTHER"); [print(f"  {k:32s} n={len(v):3d} F1={sum(v)/len(v):.2f}") for k, v in other.items()]
print(f"answer after character 1,200 of its passage: {late[1]} of {late[0]}")
