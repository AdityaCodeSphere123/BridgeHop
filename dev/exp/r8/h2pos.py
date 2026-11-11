"""Where is the gold hop-2 passage in the first hop-2 read, now vs. with a re-ordering (offline)."""
import json,re,sys,collections,importlib.util
sys.path.insert(0,'.')
from a4.evaluate import answer_scores
spec=importlib.util.spec_from_file_location('ag','agent.py'); A=importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
from a4 import Passage
gold={g['id']:g for g in map(json.loads,open('data/dev.jsonl'))}
pred={p['id']:p for p in map(json.loads,open('runs/replay_agent/predictions.jsonl'))}
tr={t['id']:t for t in map(json.loads,open('runs/replay_agent/traces.jsonl'))}
B={r[0]:r for r in json.load(open('runs/replay_agent/bridges.json'))}
corpus={}
for l in open('data/corpus.jsonl'):
    r=json.loads(l); corpus[r['doc_id']]=r
bytext={}
for r in corpus.values(): bytext.setdefault((r['title'],r['text'][:200]),r['doc_id'])
cur=collections.Counter(); new=collections.Counter(); byf=collections.defaultdict(list)
for i,g in gold.items():
    if not B[i][1]: continue
    docs=[corpus[d] for d in g['supporting_doc_ids']]; ans=g['answer'].lower()
    h2=next((d for d in docs if ans in (d['title']+' '+d['text']).lower()),docs[1])
    reads=[c for c in tr[i]['calls'] if c['type']=='llm' and 'Main question:' in c['prompt'] and 'Passages:' in c['prompt']]
    if not reads: continue
    p=reads[0]['prompt']; block=p.split('Passages:\n',1)[1].split('\n\nQuestion:',1)[0]
    items=re.findall(r'^\[(\d+)\] (.*?): (.*)$',block,re.M)
    ps=[Passage(bytext.get((t,x[:200]),'?'),t,x,0.0) for _,t,x in items]
    ids=[q.doc_id for q in ps]
    q2=re.search(r'\nQuestion: (.*)\n',p).group(1); bridge=B[i][2]
    words=A.content_words(q2)-A.content_words(bridge)
    reo=sorted(range(len(ps)),key=lambda k:(-(A.mentions(ps[k],bridge)+0.0)-0.5*A.overlap(ps[k],words),k))
    ids2=[ids[k] for k in reo]
    f=answer_scores(pred[i]['answer'],[g['answer']]+g.get('answer_aliases',[]))[1]
    r1=ids.index(h2['doc_id'])+1 if h2['doc_id'] in ids else 0
    r2=ids2.index(h2['doc_id'])+1 if h2['doc_id'] in ids2 else 0
    cur[min(r1,4) if r1 else 'absent']+=1; new[min(r2,4) if r2 else 'absent']+=1
    byf[min(r1,4) if r1 else 'absent'].append(f)
print('current position of gold hop-2 passage (4 = 4th or later):',sorted(cur.items(),key=str))
print('mean F1 by current position:',{k:round(sum(v)/len(v),2) for k,v in sorted(byf.items(),key=str)})
print('reordered position:',sorted(new.items(),key=str))
