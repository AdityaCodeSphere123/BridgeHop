"""Exact offline scoring with ZERO LLM calls: replays the server responses cached by crun.py.

    python3 dev/exp/replay.py agent.py [more agents ...]

Any change that keeps the agent's prompts identical (answer cleanup, fallbacks, citations) is scored
exactly as the server would score it. A prompt that is not in the cache counts as a miss and that
question's score is unreliable; the number of affected questions is printed.
"""
import hashlib, json, os, sys, threading
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from a4.budget import LLMError
from a4.data import SearchEngine, read_jsonl
from a4.evaluate import score
from a4.runner import run_agent

cache = {}
for line in open("runs/llm_cache.jsonl"):
    r = json.loads(line); cache[r["k"]] = r["v"]
key = lambda *p: hashlib.sha1(json.dumps(p).encode()).hexdigest()

class ReplayLLM:
    def __init__(self): self.misses = set(); self.lock = threading.Lock(); self.cur = threading.local()
    def count_tokens(self, prompt):
        v = cache.get(key("t", prompt))
        return v if v is not None else len(prompt) // 4  # unseen prompt: estimate (call will miss anyway)
    def complete(self, prompt, max_tokens):
        v = cache.get(key("c", prompt, max_tokens))
        if v is None:
            raise LLMError("replay miss")
        return tuple(v)

def main():
    eng = SearchEngine.load("data")
    gold = read_jsonl("data/dev.jsonl")
    qs = [{"id": g["id"], "question": g["question"]} for g in gold]
    for agent in sys.argv[1:]:
        llm = ReplayLLM()
        recs, traces = run_agent(agent, qs, eng, llm, workers=8, progress=False)
        miss_q = [r["id"] for r in recs if r.get("infra_error")]
        m = score(recs, gold)
        print(f"{agent:28s} S={m['S']:.4f} F1={m['answer_f1']:.4f} EM={m['answer_em']:.4f} EvF1={m['evidence_f1']:.4f} misses={len(miss_q)}")
        out = os.path.join("runs", "replay_" + os.path.basename(agent).replace(".py", ""))
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "predictions.jsonl"), "w") as f:
            for r in recs: f.write(json.dumps(r) + "\n")
        with open(os.path.join(out, "traces.jsonl"), "w") as f:
            for t in traces: f.write(json.dumps(t) + "\n")


if __name__ == "__main__":
    main()
