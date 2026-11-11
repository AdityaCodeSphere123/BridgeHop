"""Bridge accuracy of a replay run vs. gold decomposition: python3 dev/exp/r8/bridges.py runs/replay_X"""
import json,sys,re,ast,collections
sys.path.insert(0,'.')
from a4.evaluate import answer_scores,_f1
gold={g['id']:g for g in map(json.loads,open('data/dev.jsonl'))}
run=sys.argv[1]
pred={p['id']:p for p in map(json.loads,open(run+'/predictions.jsonl'))}
tr={t['id']:t for t in map(json.loads,open(run+'/traces.jsonl'))}
def bridge_of(t):
    # the first hop-2 search is keywords(Q2 with bridge) — easier: take bridge from the hop-1 read that was accepted:
    b=None
    for c in t['calls']:
        if c['type']=='llm' and 'Main question:' in c['prompt']:
            m=re.search(r'We already know that #1 = (.*?)\.\)',c['prompt'])
            if m: return m.group(1)
    return ''
stat=collections.defaultdict(list); out=[]
for i,g in gold.items():
    dec=g['decomposition']; gb=dec[0]['answer']
    b=bridge_of(tr[i]); ok=_f1(b,gb)>=0.5 if b else False
    f=answer_scores(pred[i]['answer'],[g['answer']]+list(g.get('answer_aliases',[])))[1]
    stat['right' if ok else ('none' if not b else 'wrong')].append(f)
    out.append((i,ok,b,gb,f))
for k,v in stat.items(): print(f"bridge {k:5s}: {len(v):3d} questions, mean answer F1 {sum(v)/len(v):.3f}")
json.dump(out,open(run+'/bridges.json','w'))
