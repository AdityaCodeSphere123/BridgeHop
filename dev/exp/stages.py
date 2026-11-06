"""Sort every question into the stage where it failed (uses gold bridge and gold doc ids).

    python3 dev/exp/stages.py runs/kreplay_agent [--show STAGE]
"""
import json, re, sys, collections, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from a4.evaluate import answer_scores, normalize_answer as N
gold = {g['id']: g for g in map(json.loads, open('data/dev.jsonl'))}
run = sys.argv[1]
show = sys.argv[sys.argv.index('--show') + 1] if '--show' in sys.argv else None
P = {r['id']: r for r in map(json.loads, open(run + '/predictions.jsonl'))}
T = {t['id']: t for t in map(json.loads, open(run + '/traces.jsonl'))}
corpus = {}
for l in open('data/corpus.jsonl'):
    d = json.loads(l); corpus[d['doc_id']] = d
def f1(a, b): return answer_scores(a, [b])[1]
stages = collections.Counter(); loss = collections.Counter()
for i, p in P.items():
    g = gold[i]; F = answer_scores(p['answer'], [g['answer']] + g['answer_aliases'])[1]
    if F >= 0.99: stages['ok'] += 1; continue
    gb = g['decomposition'][0]['answer']
    calls = T[i]['calls']
    llm = [c for c in calls if c['type'] == 'llm']
    found = set(d for c in calls if c['type'] == 'search' for d in c['doc_ids'])
    # gold doc for hop 1 = the supporting doc that mentions the bridge and the question's entity; approximate:
    # hop-2 gold doc = the one containing the gold answer
    sd = g['supporting_doc_ids']
    d_ans = [d for d in sd if N(g['answer']) and N(g['answer']) in N(corpus[d]['title'] + ' ' + corpus[d]['text'])]
    d2 = d_ans[0] if d_ans else sd[-1]
    d1 = [d for d in sd if d != d2][0] if len(sd) > 1 else sd[0]
    # bridge the agent used: from hop-2 read prompt "#1 = X."
    bridge = ''
    for c in llm:
        m = re.search(r"We already know that #1 = (.*?)\.\)", c['prompt'])
        if m: bridge = m.group(1); break
    shown = lambda d: any(f"{corpus[d]['title']}:" in c['prompt'] for c in llm[1:])
    if not bridge:
        st = 'A no bridge' + (' (hop1 doc not found)' if d1 not in found else ' (hop1 doc found)')
    elif f1(bridge, gb) < 0.5:
        st = 'B wrong bridge' + (' (hop1 doc not found)' if d1 not in found else ' (hop1 doc found)')
    elif d2 not in found:
        st = 'C bridge ok, hop2 doc not found'
    else:
        st = 'D bridge ok, hop2 doc found, read wrong'
    if F > 0: st += ' [partial]'
    stages[st] += 1; loss[st] += 1 - F
    if show and st.startswith(show):
        print(f"--- {g['question']}\n    gold={g['answer']!r} pred={p['answer']!r} gold_bridge={gb!r} bridge={bridge!r} F1={F:.2f}")
        print(f"    plan: {llm[0]['output'].strip()[:150]!r}")
        print(f"    queries: {[c['query'] for c in calls if c['type']=='search']}")
        print(f"    hop1 doc {d1} {corpus[d1]['title']!r} found={d1 in found}  hop2 doc {d2} {corpus[d2]['title']!r} found={d2 in found}")
for k in sorted(stages): print(f"{stages[k]:4d}  F1 lost {loss[k]:5.1f}  {k}")
