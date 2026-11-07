"""Score a post-processing patch on all server runs at once (zero LLM calls, server cache).

    python3 dev/exp/smultibed.py PATCHFILE.py      (PATCH = [(old, new), ...] applied to every dev/exp/sbeds/*.py)
"""
import hashlib, json, os, sys, glob, importlib.util, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from a4.budget import LLMError
from a4.data import SearchEngine, read_jsonl
from a4.evaluate import score
from a4.runner import run_agent
SC = {}
for line in open("runs/llm_cache.jsonl"):
    r = json.loads(line); SC[r["k"]] = r["v"]
sha = lambda *p: hashlib.sha1(json.dumps(p).encode()).hexdigest()
class Server:
    def count_tokens(self, prompt):
        v = SC.get(sha("t", prompt)); return v if v is not None else len(prompt) // 4
    def complete(self, prompt, mt):
        v = SC.get(sha("c", prompt, mt))
        if v is None: raise LLMError("miss")
        return tuple(v)
def main():
    spec = importlib.util.spec_from_file_location("patch", sys.argv[1]); P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
    eng = SearchEngine.load("data"); gold = read_jsonl("data/dev.jsonl")
    qs = [{"id": g["id"], "question": g["question"]} for g in gold]
    tmp = tempfile.mkdtemp(); deltas = []
    for f in sorted(glob.glob("dev/exp/sbeds/*.py")):
        src = open(f).read(); new = src
        for old, rep in P.PATCH:
            if old not in new: print(f"   [{os.path.basename(f)}] patch text not found: {old[:60]!r}")
            new = new.replace(old, rep)
        pf = os.path.join(tmp, "p_" + os.path.basename(f)); open(pf, "w").write(new)
        res = []
        for a in (f, pf):
            recs, _ = run_agent(a, qs, eng, Server(), workers=8, progress=False)
            res.append(score(recs, gold))
        d = res[1]["S"] - res[0]["S"]; deltas.append(d)
        print(f"{os.path.basename(f):26s} S {res[0]['S']:.4f} -> {res[1]['S']:.4f}  dS={d:+.4f}")
    print(f"MEAN dS={sum(deltas)/len(deltas):+.4f}  wins={sum(d>1e-9 for d in deltas)} losses={sum(d<-1e-9 for d in deltas)} of {len(deltas)}")
if __name__ == "__main__":
    main()
