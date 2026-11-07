"""Compare a server run against runs/S_final: halves, wins/losses, changed answers, tokens since a cache mark.
    python3 dev/exp/r7/cmp.py RUN [cache_mark_file]"""
import json,sys,os
sys.path.insert(0,'.')
import a4.evaluate as E
run=sys.argv[1]
if len(sys.argv)>2:
    n0=int(open(sys.argv[2]).read().split()[0]); ls=open('runs/llm_cache.jsonl').read().splitlines()[n0:]
    print(f"tokens: {sum(v[1]+v[2] for v in (json.loads(l)['v'] for l in ls) if isinstance(v,list) and len(v)==3):,}")
gold={g['id']:g for g in map(json.loads,open('data/dev.jsonl'))}; order=list(gold)
L=lambda n:{r['id']:r for r in map(json.loads,open(f'runs/{n}/predictions.jsonl'))}
b,r=L(os.environ.get('BASE','S_final')),L(run)
h=[E.score([r[i] for i in order[k:k+150]],[gold[i] for i in order[k:k+150]])['S']-E.score([b[i] for i in order[k:k+150]],[gold[i] for i in order[k:k+150]])['S'] for k in (0,150)]
F=lambda x,g: E.answer_scores(x['answer'],[g['answer']]+g['answer_aliases'])[1]
print("halves",[round(x,4) for x in h],"wins",sum(F(r[i],gold[i])>F(b[i],gold[i])+1e-9 for i in order),"losses",sum(F(r[i],gold[i])<F(b[i],gold[i])-1e-9 for i in order))
for i in order:
    if abs(F(r[i],gold[i])-F(b[i],gold[i]))>1e-9: print(f"  {F(b[i],gold[i]):.2f}->{F(r[i],gold[i]):.2f} gold={gold[i]['answer'][:26]!r:28} cur={b[i]['answer'][:26]!r:28} new={r[i]['answer'][:26]!r}")
