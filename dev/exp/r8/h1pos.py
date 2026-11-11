"""Position of the gold hop-1 passage in the first hop-1 read, vs. bridge correctness (offline)."""
import json,re,sys,collections,importlib.util
sys.path.insert(0,'.')
spec=importlib.util.spec_from_file_location('ag','agent.py'); A=importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
from a4 import Passage
gold={g['id']:g for g in map(json.loads,open('data/dev.jsonl'))}
tr={t['id']:t for t in map(json.loads,open('runs/replay_agent/traces.jsonl'))}
B={r[0]:r for r in json.load(open('runs/replay_agent/bridges.json'))}
corpus={}
for l in open('data/corpus.jsonl'):
    r=json.loads(l); corpus[r['doc_id']]=r
bytext={(r['title'],r['text'][:200]):r['doc_id'] for r in corpus.values()}
pos=collections.defaultdict(list); newpos=collections.Counter(); oldpos=collections.Counter()
for i,g in gold.items():
    gb=g['decomposition'][0]['answer'].lower()
    docs=[corpus[d] for d in g['supporting_doc_ids']]
    h1=next((d for d in docs if gb in (d['title']+' '+d['text']).lower()),None)
    if not h1: continue
    reads=[c for c in tr[i]['calls'] if c['type']=='llm' and 'This is one step' in c['prompt'] and 'Passages:' in c['prompt']]
    if not reads: continue
    p=reads[0]['prompt']; block=p.split('Passages:\n',1)[1].split('\n\nQuestion:',1)[0]
    items=re.findall(r'^\[(\d+)\] (.*?): (.*)$',block,re.M)
    ps=[Passage(bytext.get((t,x[:200]),'?'),t,x,0.0) for _,t,x in items]
    ids=[q.doc_id for q in ps]
    q1=re.search(r'\nQuestion: (.*)\n',p).group(1); Q=g['question']
    words=A.content_words(q1)
    reo=sorted(range(len(ps)),key=lambda k:(-2*A.title_in(ps[k],Q)-A.overlap(ps[k],words),k))
    r=ids.index(h1['doc_id'])+1 if h1['doc_id'] in ids else 0
    r2=[ids[k] for k in reo].index(h1['doc_id'])+1 if r else 0
    pos[min(r,4) if r else 'absent'].append(B[i][1]); oldpos[min(r,4) if r else 'absent']+=1; newpos[min(r2,4) if r2 else 'absent']+=1
print({k:(len(v),round(sum(v)/len(v),2)) for k,v in sorted(pos.items(),key=str)})
print('now      ',sorted(oldpos.items(),key=str)); print('reordered',sorted(newpos.items(),key=str))
