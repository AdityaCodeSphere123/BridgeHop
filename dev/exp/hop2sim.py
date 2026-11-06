"""Hop-2 retrieval simulation (no LLM): uses the agent's own plan, bridge and hop-1 searches from a run,
then tries different hop-2 query sets and counts how often the gold hop-2 passage is among those shown.

    python3 dev/exp/hop2sim.py runs/kreplay_agent
"""
import json, os, re, sys, importlib.util
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from a4.data import SearchEngine
from a4.evaluate import normalize_answer as N
spec = importlib.util.spec_from_file_location("ag", "agent.py"); A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
eng = SearchEngine.load("data")
gold = {g['id']: g for g in map(json.loads, open('data/dev.jsonl'))}
corpus = {}
for l in open('data/corpus.jsonl'):
    d = json.loads(l); corpus[d['doc_id']] = d
run = sys.argv[1]
T = {t['id']: t for t in map(json.loads, open(run + '/traces.jsonl'))}
cache = {}
def S(q, k=10):
    if (q, k) not in cache: cache[(q, k)] = eng.search(q, k)
    return cache[(q, k)]
def hop2_doc(g):
    sd = g['supporting_doc_ids']
    d = [d for d in sd if N(g['answer']) and N(g['answer']) in N(corpus[d]['title'] + ' ' + corpus[d]['text'])]
    return d[0] if d else sd[-1]
KW = A.keywords
VARIANTS = {
    'V0 current  [kw(Q2b), b+kw(Q)]':      lambda b, q2, q: [KW(q2), f"{b} {KW(q)}"],
    'V1 [kw(Q2b), b]':                      lambda b, q2, q: [KW(q2), b],
    'V2 [b+kw(Q), b]':                      lambda b, q2, q: [f"{b} {KW(q)}", b],
    'V3 [kw(Q2b), b+kw(Q), b] (3 searches)': lambda b, q2, q: [KW(q2), f"{b} {KW(q)}", b],
    'V4 [kw(Q2b), b+kw(Q2b)]':              lambda b, q2, q: [KW(q2), f"{b} {KW(q2)}"],
}
res = {v: [0, 0] for v in VARIANTS}; n = 0
for i, t in T.items():
    g = gold[i]; calls = t['calls']; llm = [c for c in calls if c['type'] == 'llm']
    bridge = ''
    for c in llm:
        m = re.search(r"We already know that #1 = (.*?)\.\)", c['prompt'])
        if m: bridge = m.group(1); break
    if not bridge or N(bridge) != N(g['decomposition'][0]['answer']) and A.norm(g['decomposition'][0]['answer']) not in A.norm(bridge):
        continue  # only questions where the agent's bridge is right
    m = re.search(r"Question: (.*)\n", [c for c in llm if '#1 =' in c['prompt']][0]['prompt'])
    q2b = m.group(1)  # hop-2 sub-question with the bridge filled in
    q2 = q2b  # keywords() is applied inside the variants
    # hop-1 searches = the searches before the first hop-2 read prompt
    h1 = []
    for c in calls:
        if c['type'] == 'llm' and '#1 =' in c['prompt']: break
        if c['type'] == 'search': h1.append(c['query'])
    d2 = hop2_doc(g); n += 1
    for v, fn in VARIANTS.items():
        notes = A.Notes(g['question'])
        for q in h1: notes.add(q, S(q)); 
        notes.hop_of_query = [0] * len(h1)
        notes.hop = 1
        for q in fn(bridge, q2, g['question']):
            notes.add(q, S(q))
        ranked = notes.ranked(1)
        found = d2 in notes.passages
        shown = d2 in [p.doc_id for p in ranked[:12]]
        res[v][0] += found; res[v][1] += shown
print(f"questions with a correct bridge: {n}")
for v, (f, s) in res.items(): print(f"  {v:40s} found {f/n:.3f}  in top-12 shown {s/n:.3f}")
