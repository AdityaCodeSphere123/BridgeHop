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
        'V4 [kw(Q2b), b+kw(Q2b)]':              lambda b, q2, q: [KW(q2), f"{b} {KW(q2)}"],
}
res = {}; n = 0
def by_mention(notes, b):
    r = notes.ranked(1)
    return [p for p in r if A.mentions(p, b)] + [p for p in r if not A.mentions(p, b)]
def by_mention_hop2only(notes, b):
    r = notes.ranked(1); h2 = set(d for hits, h in zip(notes.hits, notes.hop_of_query) if h == 1 for d in hits)
    top = [p for p in r if p.doc_id in h2]
    return [p for p in top if A.mentions(p, b)] + [p for p in r if p.doc_id not in h2 and A.mentions(p, b)] + [p for p in top if not A.mentions(p, b)] + [p for p in r if p.doc_id not in h2 and not A.mentions(p, b)]
def unified(notes, b, w=0.5):
    sc = {}
    for hits, h in zip(notes.hits, notes.hop_of_query):
        for i, d in enumerate(hits): sc[d] = sc.get(d, 0) + (1 if h == 1 else w) / (A.RRF_C + i + 1)
    return [notes.passages[d] for d in sorted(sc, key=lambda d: -sc[d])]
def unified_mention(notes, b):
    r = unified(notes, b)
    return [p for p in r if A.mentions(p, b)] + [p for p in r if not A.mentions(p, b)]
def keep(k):
    def f(notes, b):
        r = notes.ranked(1); head, tail = r[:k], r[k:]
        return head + [p for p in tail if A.mentions(p, b)] + [p for p in tail if not A.mentions(p, b)]
    return f
def swap_in(notes, b, n=12):
    r = notes.ranked(1); top, rest = r[:n], r[n:]
    extra = [p for p in rest if A.mentions(p, b)]
    if not extra: return r
    keep = list(top)
    for p in extra:
        idx = [j for j in range(len(keep) - 1, 0, -1) if not A.mentions(keep[j], b)]
        if not idx: break
        keep[idx[0]] = p
    used = {p.doc_id for p in keep}
    return keep + [p for p in r if p.doc_id not in used]
CHANGED = [0]
RANKS = {'swap-in bottom': swap_in, **{f'keep top {k} + mention': keep(k) for k in (1, 2, 3, 4, 6)}, 'current': lambda n, b: n.ranked(1), 'mention-first': by_mention, 'mention, hop2 lists first': by_mention_hop2only,
         'unified rrf w=.5': unified, 'unified + mention': unified_mention}
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
        for rk, rfn in RANKS.items():
            ranked = rfn(notes, bridge)
            key = (v, rk); res.setdefault(key, [0, 0, 0])
            res[key][0] += d2 in notes.passages
            res[key][1] += d2 in [p.doc_id for p in ranked[:12]]
            res[key][2] += bool(ranked) and ranked[0].doc_id == d2
print(f"questions with a correct bridge: {n}")
for (v, rk), (f, s, t1) in res.items(): print(f"  {v:40s} {rk:26s} found {f/n:.3f}  top-12 {s/n:.3f}  top-1 {t1/n:.3f}")
