"""Exact offline scoring against the Kaggle cache (local Qwen3-4B outputs), ZERO LLM calls.

    python3 dev/exp/kreplay.py agent.py [more agents ...]

Same idea as replay.py, but reads runs/kaggle_cache.jsonl (written by kaggle_run.py on Kaggle).
A prompt that is not in the cache counts as a miss; the number of affected questions is printed.
"""
import hashlib, json, os, sys, threading
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from a4.budget import LLMError
from a4.data import SearchEngine, read_jsonl
from a4.evaluate import score
from a4.runner import run_agent

cache = {}
for line in open("runs/kaggle_cache.jsonl"):
    r = json.loads(line); cache[r["k"]] = r["v"]
key = lambda prompt, mt: hashlib.sha1(json.dumps(["qwen", "float16", prompt, mt]).encode()).hexdigest()

class ReplayLLM:
    def count_tokens(self, prompt):
        return len(prompt) // 4  # no tokenizer here; prompts are kept far below 3000 tokens by the char budget
    def complete(self, prompt, max_tokens):
        v = cache.get(key(prompt, max_tokens))
        if v is None:
            raise LLMError("replay miss")
        return tuple(v)

def main():
    eng = SearchEngine.load("data")
    gold = read_jsonl("data/dev.jsonl")
    qs = [{"id": g["id"], "question": g["question"]} for g in gold]
    for agent in sys.argv[1:]:
        recs, traces = run_agent(agent, qs, eng, ReplayLLM(), workers=8, progress=False)
        miss_q = [r["id"] for r in recs if r.get("infra_error")]
        m = score(recs, gold)
        print(f"{agent:34s} S={m['S']:.4f} F1={m['answer_f1']:.4f} EM={m['answer_em']:.4f} EvF1={m['evidence_f1']:.4f} misses={len(miss_q)}")
        out = os.path.join("runs", "kreplay_" + os.path.basename(agent).replace(".py", ""))
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "predictions.jsonl"), "w") as f:
            for r in recs: f.write(json.dumps(r) + "\n")
        with open(os.path.join(out, "traces.jsonl"), "w") as f:
            for t in traces: f.write(json.dumps(t) + "\n")


if __name__ == "__main__":
    main()
