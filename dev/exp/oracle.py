"""Offline end-to-end check of retrieval + control flow + citations, with NO LLM.

The "oracle reader" answers a read correctly exactly when the gold paragraph of that hop is in the prompt,
and says unknown (or a wrong guess, when asked for a best guess) otherwise. Plans are the agent's real plans
from a server run. So the score measures how well an agent puts the right evidence in front of the reader;
it is an upper bound on what reading can achieve, not a prediction of the real score.

    python3 dev/exp/oracle.py agent.py dev/exp/agent_c.py
"""
import json, os, re, sys, threading
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from a4.data import SearchEngine, read_jsonl
from a4.evaluate import score, normalize_answer as N
from a4.runner import run_agent

TOK_PER_CHAR = 0.30   # pessimistic (real median 0.245, p99 0.32): stresses the 3,000-token limit


def main():
    eng = SearchEngine.load("data")
    gold = read_jsonl("data/dev.jsonl")
    plans = {}
    for t in read_jsonl("runs/v1b/traces.jsonl"):
        plans[t["id"]] = t["calls"][0]["output"]
    def text(d): i = eng.corpus.index_of[d]; return eng.corpus.titles[i] + " " + eng.corpus.texts[i]
    info = {}
    for g in gold:
        golds = [g["answer"]] + g["answer_aliases"]; gids = g["supporting_doc_ids"]
        has = lambda d, a: bool(N(a)) and f" {N(a)} " in f" {N(text(d))} "
        h2 = next((d for d in gids if any(has(d, a) for a in golds)), gids[1]); h1 = next(d for d in gids if d != h2)
        snip = lambda d: eng.corpus.texts[eng.corpus.index_of[d]][:80]
        info[g["question"]] = dict(id=g["id"], b=g["decomposition"][0]["answer"], a=g["answer"], s1=snip(h1), s2=snip(h2))

    class Oracle:
        def __init__(self): self.max_tok = 0; self.lock = threading.Lock()
        def count_tokens(self, prompt):
            n = int(len(prompt) * TOK_PER_CHAR)
            with self.lock: self.max_tok = max(self.max_tok, n)
            return n
        def complete(self, prompt, max_tokens):
            q = next((q for q in info if q in prompt), None)
            if q is None: return "Answer: unknown", 0, 0
            g = info[q]
            if prompt.startswith(("Split the question", "Every question below")):
                return plans[g["id"]], 0, 0
            blocks = re.findall(r"^\[(\d+)\] .*$", prompt, flags=re.M)
            def pos(snip):
                for line in prompt.splitlines():
                    m = re.match(r"^\[(\d+)\] ", line)
                    if m and snip[:60] in line: return m.group(1)
            guess = "best guess" in prompt or "never" in prompt
            if "This is one step towards answering" in prompt:
                i = pos(g["s1"]); return (f"Passage: {i}\nAnswer: {g['b']}" if i else ("Passage: 1\nAnswer: wrong guess" if guess else "Passage: none\nAnswer: unknown")), 0, 0
            if "We already know that #1" in prompt:
                ok = g["b"].lower() in prompt.split("We already know that #1 =")[1][:200].lower()
                i = pos(g["s2"]); return (f"Passage: {i}\nAnswer: {g['a']}" if i and ok else ("Passage: 1\nAnswer: wrong guess" if guess else "Passage: none\nAnswer: unknown")), 0, 0
            i = pos(g["s2"]); return (g["a"] if i else "wrong guess"), 0, 0

    qs = [{"id": g["id"], "question": g["question"]} for g in gold]
    for agent in sys.argv[1:]:
        llm = Oracle()
        recs, traces = run_agent(agent, qs, eng, llm, workers=8, progress=False)
        m = score(recs, gold)
        out = os.path.join("runs", "oracle_" + os.path.basename(agent).replace(".py", ""))
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "predictions.jsonl"), "w") as f:
            for r in recs: f.write(json.dumps(r) + "\n")
        with open(os.path.join(out, "traces.jsonl"), "w") as f:
            for t in traces: f.write(json.dumps(t) + "\n")
        rej = sum(1 for t in traces for c in t["calls"] if c["type"] == "llm_rejected")
        print(f"{agent:26s} oracle S={m['S']:.4f} F1={m['answer_f1']:.4f} EvF1={m['evidence_f1']:.4f} "
              f"search={m['mean_search_calls']} llm={m['mean_llm_calls']} status={m['status_counts']} "
              f"max_prompt_tok~{llm.max_tok} too_long_retries={rej}")


if __name__ == "__main__":
    main()
