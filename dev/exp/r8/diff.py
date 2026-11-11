"""Per-question answer diff between two replay runs: python3 dev/exp/r8/diff.py runs/replay_A runs/replay_B"""
import json,sys
sys.path.insert(0,'.')
from a4.evaluate import answer_scores
gold={g['id']:g for g in map(json.loads,open('data/dev.jsonl'))}
a={p['id']:p for p in map(json.loads,open(sys.argv[1]+'/predictions.jsonl'))}
b={p['id']:p for p in map(json.loads,open(sys.argv[2]+'/predictions.jsonl'))}
for i in a:
    if a[i]['answer']!=b[i]['answer']:
        g=gold[i]; G=[g['answer']]+g.get('answer_aliases',[])
        fa,fb=answer_scores(a[i]['answer'],G)[1],answer_scores(b[i]['answer'],G)[1]
        print(f"{fa:.2f}->{fb:.2f} {a[i]['answer'][:35]!r} -> {b[i]['answer'][:35]!r} | gold {g['answer'][:30]!r} | {g['question'][:50]}")
